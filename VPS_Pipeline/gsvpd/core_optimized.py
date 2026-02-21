"""
OPTIMIZED Core module for downloading, processing, and stitching Google Street View panoramas.

Key optimizations:
1. Batch processing instead of all-at-once to reduce memory pressure
2. Offload CPU-intensive operations (stitching, view extraction) to ThreadPoolExecutor
3. Use more workers for parallel processing
4. Better connection pooling with increased limits
5. Release memory immediately after processing
6. Parallel GCS uploads using ThreadPoolExecutor
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from aiohttp import ClientTimeout  # used for session-level timeout
import cv2
import math
import numpy as np
from rich import print
from typing import Tuple, Union, Optional, List
import os

from .constants import (
    ZOOM_SIZES,
    OLD_ZOOM_SIZES,
    TILE_COUNT_TO_SIZE,
    TILES_AXIS_COUNT,
    TILE_SIZE,
    X_COUNT_TO_SIZE,
    ZOOM_HEIGHTS,
)
from .my_utils import save_img
from .directional_views import DirectionalViewExtractor, DirectionalViewConfig, DirectionalViewResult
from .augmentations import apply_pixel_augmentations
from .gcs_uploader import GCSUploader, GCSConfig, GCSUploadResult
from .progress_bar import ProgressBar
from .zip_batcher import ZipBatcher



# ─── Tile Row Skip Optimization ──────────────────────────────────────────────

def _compute_phi_bounds(fov_deg, max_pitch_deg=0.0, max_roll_deg=0.0):
    """
    Compute the min/max latitude (phi) in degrees that any pixel in a
    perspective view can reach, given FOV and worst-case pitch/roll.
    """
    fov_rad = math.radians(fov_deg)
    hfov_rad = 2.0 * math.atan(math.tan(fov_rad / 2.0))
    tan_h = math.tan(hfov_rad / 2.0)
    tan_v = math.tan(fov_rad / 2.0)

    # 8 extreme points on the view perimeter (corners + edge midpoints)
    perimeter = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),           (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    # Test all worst-case rotation combos
    configs = [(0.0, 0.0)]
    if max_pitch_deg > 0 or max_roll_deg > 0:
        for p in (max_pitch_deg, -max_pitch_deg):
            for r in (max_roll_deg, -max_roll_deg):
                configs.append((p, r))

    phi_min = 90.0
    phi_max = -90.0

    for pitch_d, roll_d in configs:
        pitch = math.radians(pitch_d)
        roll = math.radians(roll_d)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        for nx, ny in perimeter:
            rx = tan_h * nx
            ry = tan_v * ny
            rz = 1.0
            inv_len = 1.0 / math.sqrt(rx * rx + ry * ry + rz * rz)
            rx *= inv_len
            ry *= inv_len
            rz *= inv_len

            if roll != 0:
                rx, ry = rx * cr - ry * sr, rx * sr + ry * cr
            if pitch != 0:
                ry, rz = ry * cp - rz * sp, ry * sp + rz * cp

            phi = math.degrees(math.asin(max(-1.0, min(1.0, ry))))
            if phi < phi_min:
                phi_min = phi
            if phi > phi_max:
                phi_max = phi

    return phi_min, phi_max


def compute_required_tile_rows(zoom_level, fov_degrees, augment=False):
    """
    Compute which tile Y rows are needed for view extraction at the given
    zoom level and FOV. Returns a frozenset of y indices, or None if all
    rows should be fetched (no optimization possible).

    Computed once per scraping run — FOV, zoom, and view angles are constant.
    Only effective for zoom >= 3.
    """
    if zoom_level < 3:
        return None

    tiles_x, tiles_y = TILES_AXIS_COUNT[zoom_level]
    total_y = tiles_y + 1

    # Worst-case FOV and rotations for augmentation mode
    if augment:
        effective_fov = max(fov_degrees, 100.0)  # augment uses up to 100°
        phi_min, phi_max = _compute_phi_bounds(effective_fov, 5.0, 5.0)
    else:
        phi_min, phi_max = _compute_phi_bounds(fov_degrees)

    # Compute for both old and new pano heights, take union (conservative)
    heights = ZOOM_HEIGHTS.get(zoom_level, ())
    if not heights:
        return None

    required = set()
    for pano_h in heights:
        # phi_max -> topmost latitude -> smallest y (top of image)
        min_y = (0.5 - phi_max / 180.0) * pano_h
        max_y = (0.5 - phi_min / 180.0) * pano_h

        for ty in range(total_y):
            tile_top = ty * TILE_SIZE
            tile_bottom = (ty + 1) * TILE_SIZE
            if tile_bottom > min_y and tile_top < max_y:
                required.add(ty)

    if len(required) >= total_y:
        return None

    return frozenset(required)


# Helper functions for dimension detection (zoom 0-2 only)
def _check_black_bottom(tile_data: bytes) -> bool:
    """Check if tile has black bottom (executed in subprocess)."""
    import numpy as np
    from PIL import Image
    from io import BytesIO
    
    tile = Image.open(BytesIO(tile_data))
    arr = np.array(tile)
    bottom = arr[-5:]
    return bool(np.all(bottom <= 10))


def _check_black_percentage(tile_data: bytes) -> float:
    """Calculate black percentage (executed in subprocess)."""
    import numpy as np
    from PIL import Image
    from io import BytesIO
    
    tile = Image.open(BytesIO(tile_data))
    img_np = np.array(tile)
    black_pixels = np.all(img_np <= 10, axis=2)
    return float(np.sum(black_pixels) / black_pixels.size * 100)


def _stitch_and_process_tiles(
    tile_data_list: List[Tuple[int, int, bytes]],
    width: int,
    height: int,
    config: dict,
    panoid: str,
    zoom_level: int,
    heading_deg: float = None
) -> dict:
    """
    Stitch tiles and extract views in a thread.
    Uses numpy/cv2 (GIL-releasing) for all heavy operations.
    """
    import cv2
    import numpy as np
    import os

    result = {
        "success": False,
        "error": "",
        "size": (width, height),
        "tiles": (0, 0),
        "views": [],
        "view_filenames": [],
        "panorama_bytes": None,
        "timings": {},
    }

    try:
        import time as _time

        # ── Stitch tiles (numpy-based, GIL-friendly) ──
        t0 = _time.perf_counter()
        full_img_cv = np.zeros((height, width, 3), dtype=np.uint8)
        x_values = set()
        y_values = set()

        for x, y, tile_bytes in tile_data_list:
            tile = cv2.imdecode(
                np.frombuffer(tile_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if tile is not None:
                th, tw = tile.shape[:2]
                
                # Calculate target coordinates on the main canvas
                y_start = y * 512
                y_end = min(y_start + th, height)
                x_start = x * 512
                x_end = min(x_start + tw, width)
                
                # Calculate source coordinates from the tile
                # (How much of the tile do we need to copy?)
                h_target = y_end - y_start
                w_target = x_end - x_start
                
                if h_target > 0 and w_target > 0:
                     full_img_cv[y_start:y_end, x_start:x_end] = tile[:h_target, :w_target]
                
            x_values.add(x)
            y_values.add(y)

        result["tiles"] = (len(x_values), len(y_values))
        t1 = _time.perf_counter()
        result["timings"]["  stitch"] = t1 - t0

        # ── Extract directional views ──
        if config.get("create_directional_views"):
            from .directional_views import DirectionalViewExtractor, DirectionalViewConfig
            from .augmentations import apply_pixel_augmentations

            view_extractor = DirectionalViewExtractor()

            # Configure view extraction
            view_config = DirectionalViewConfig(
                output_resolution=config.get("view_resolution", 512),
                fov_degrees=config.get("view_fov", 90.0),
                num_views=config.get("num_views", 6),
                global_view=config.get("global_view", False),
                augment=config.get("augment", False),
                target_yaw=heading_deg,
                antialias_strength=0.0 if config.get("no_antialias") else config.get("aa_strength", 0.8),
                interpolation=config.get("interpolation", "lanczos"),
                yaw_offset=config.get("view_offset", 0.0)
            )

            t2 = _time.perf_counter()
            view_result = view_extractor.extract_views(full_img_cv, view_config)
            t3 = _time.perf_counter()
            result["timings"]["  projection"] = t3 - t2

            aug_time = 0.0
            encode_time = 0.0

            if view_result.success:
                # Process resulting views
                for i, (view, meta) in enumerate(zip(view_result.views, view_result.metadata)):

                    # Apply Pixel Augmentations if enabled (Blur, Noise, Color)
                    if config.get("augment"):
                        ta = _time.perf_counter()
                        view = apply_pixel_augmentations(view)
                        aug_time += _time.perf_counter() - ta

                    # Determine Filename
                    yaw = meta['yaw']
                    if config.get("global_view") or heading_deg is not None:
                         # Single Random View filename convention
                         fname = f"{panoid}_rnd_Y{int(yaw)}.jpg"
                         if config.get("augment"):
                             fname = f"{panoid}_aug_Y{int(yaw)}_P{int(meta['pitch'])}.jpg"
                    else:
                        # Standard filename convention
                        fname = f"{panoid}_zoom{zoom_level}_view{i:02d}_{yaw:.0f}deg.jpg"

                    # Encode to JPEG
                    te = _time.perf_counter()
                    jpeg_q = config.get("jpeg_quality", 95)
                    _, buffer = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
                    encode_time += _time.perf_counter() - te

                    result["views"].append(buffer.tobytes())
                    result["view_filenames"].append(fname)

            result["timings"]["  augmentation"] = aug_time
            result["timings"]["  jpeg_encode"] = encode_time

        # Encode panorama ONLY if --keep-pano was specified
        if config.get("keep_panorama"):
            jpeg_q = config.get("jpeg_quality", 95)
            _, buffer = cv2.imencode('.jpg', full_img_cv, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            result["panorama_bytes"] = buffer.tobytes()

        del full_img_cv
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


async def fetch_tile(
    session: aiohttp.ClientSession,
    panoid: str,
    x: int,
    y: int,
    zoom_level: int,
    retries: int = 2,
    backoff: float = 0.15
) -> Union[None, Tuple]:
    """Fetch a single panorama tile with retry support."""
    # Round-robin across 4 CDN hosts for higher aggregate connection throughput
    host = (x + y) % 4
    url = f"https://cbk{host}.google.com/cbk?output=tile&panoid={panoid}&zoom={zoom_level}&x={x}&y={y}"

    for attempt in range(1, retries + 1):
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None

                BLACK_TILE_BYTE_SIZE = 1184
                BLACK_TILE_SIZE = int(response.headers.get("Content-Length", 0))

                if BLACK_TILE_SIZE == BLACK_TILE_BYTE_SIZE:
                    return None

                data = await response.read()
                return (x, y, data)

        except Exception:
            if attempt < retries:
                await asyncio.sleep(backoff * (2 ** (attempt - 1)))

    return None


async def determine_dimensions(
    executor: ThreadPoolExecutor,
    tiles: list,
    zoom_level: int,
    x_tiles_count: int,
    y_tiles_count: int
) -> Tuple[int, int]:
    """Determine panorama dimensions using subprocess for image analysis."""
    if zoom_level == 0:
        black_perc = await asyncio.get_running_loop().run_in_executor(
            executor, _check_black_percentage, tiles[0][2]
        )
        return OLD_ZOOM_SIZES[zoom_level] if black_perc > 55 else ZOOM_SIZES[zoom_level]

    elif 0 < zoom_level <= 2:
        black = await asyncio.get_running_loop().run_in_executor(
            executor, _check_black_bottom, tiles[1][2]
        )
        return OLD_ZOOM_SIZES[zoom_level] if black else ZOOM_SIZES[zoom_level]

    return TILE_COUNT_TO_SIZE.get((x_tiles_count, y_tiles_count))


async def process_panoid(
    session: aiohttp.ClientSession,
    panoid: Union[str, dict],
    sem_pano: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    upload_executor: ThreadPoolExecutor,
    config: dict,
    gcs_uploader: Optional[GCSUploader] = None,
    zip_batcher: Optional[ZipBatcher] = None
) -> dict:
    """Download, reconstruct, and save a single panorama (OPTIMIZED with ZIP batching)."""
    
    # Extract panoid and optional heading
    if isinstance(panoid, dict):
        heading_deg = panoid.get('heading_deg')
        panoid_str = panoid['panoid']
    else:
        heading_deg = None
        panoid_str = panoid

    result = {
        "pano_id": panoid_str,
        "zoom": config.get("zoom_level", 2),
        "size": (0, 0),
        "tiles": (0, 0),
        "file_size": 0,
        "success": False,
        "error": "",
        "views_created": 0,
        "view_files": [],
        "gcs_uris": [],
        "uploaded_to_gcs": False
    }
    


    try:
        async with sem_pano:
            import time as _time
            t_pano_start = _time.perf_counter()

            zoom_level = config.get("zoom_level", 2)
            tiles_x, tiles_y = TILES_AXIS_COUNT[zoom_level]

            # ── Fetch tiles (async I/O) ──
            t0 = _time.perf_counter()
            required_y = config.get("_required_tile_rows")
            tasks = [
                fetch_tile(session, panoid_str, x, y, zoom_level)
                for x in range(tiles_x + 1)
                for y in range(tiles_y + 1)
                if required_y is None or y in required_y
            ]
            tiles = [tile for tile in await asyncio.gather(*tasks) if tile is not None]
            t1 = _time.perf_counter()

            if not tiles:
                result["error"] = "No tiles fetched"
                return result



            # ── Determine dimensions ──
            t2 = _time.perf_counter()
            x_tiles_count = len({x for x, _, _ in tiles})

            if required_y is not None and zoom_level >= 3:
                dims = X_COUNT_TO_SIZE.get(x_tiles_count)
                if dims:
                    w, h = dims
                else:
                    y_tiles_count = len({y for _, y, _ in tiles})
                    w, h = await determine_dimensions(executor, tiles, zoom_level, x_tiles_count, y_tiles_count)
            else:
                y_tiles_count = len({y for _, y, _ in tiles})
                w, h = await determine_dimensions(executor, tiles, zoom_level, x_tiles_count, y_tiles_count)
            t3 = _time.perf_counter()



            # ── OFFLOAD CPU-INTENSIVE WORK TO THREAD ──
            t4 = _time.perf_counter()
            process_result = await asyncio.get_running_loop().run_in_executor(
                executor,
                _stitch_and_process_tiles,
                tiles,
                w, h,
                config,
                panoid_str,
                zoom_level,
                heading_deg
            )
            t5 = _time.perf_counter()



            if not process_result["success"]:
                result["error"] = process_result["error"]
                return result

            result["size"] = process_result["size"]
            result["tiles"] = process_result["tiles"]
            result["views_created"] = len(process_result["views"])

            # ── Save files to disk ──
            t6 = _time.perf_counter()

            keep_pano = config.get("keep_panorama", False)
            gcs_also_local = config.get("gcs_also_local", False)

            if keep_pano and process_result["panorama_bytes"]:
                output_dir = config.get("output_dir", ".")
                zoom_output_folder = os.path.join(output_dir, f"panos_z{zoom_level}")
                os.makedirs(zoom_output_folder, exist_ok=True)
                out_path = os.path.join(zoom_output_folder, f"{panoid_str}.jpg")

                with open(out_path, 'wb') as f:
                    f.write(process_result["panorama_bytes"])

                result["file_size"] = len(process_result["panorama_bytes"])

            if result["views_created"] > 0 and gcs_also_local:
                output_dir = config.get("output_dir", ".")
                views_dir = os.path.join(output_dir, f"views_z{zoom_level}")
                os.makedirs(views_dir, exist_ok=True)

                for view_bytes, filename in zip(process_result["views"], process_result["view_filenames"]):
                    filepath = os.path.join(views_dir, filename)

                    with open(filepath, 'wb') as f:
                        f.write(view_bytes)

                    result["view_files"].append(filename)

            t7 = _time.perf_counter()



            # Add to ZIP batcher if GCS upload enabled
            if zip_batcher:
                if config.get("keep_panorama") and process_result["panorama_bytes"]:
                    pano_filename = f"{panoid_str}.jpg"
                    await zip_batcher.add_image(pano_filename, process_result["panorama_bytes"], "pano")
                    result["uploaded_to_gcs"] = True

                for view_bytes, filename in zip(process_result["views"], process_result["view_filenames"]):
                    await zip_batcher.add_image(filename, view_bytes, "view")
                    result["uploaded_to_gcs"] = True

            result["success"] = True



    except Exception as error:
        result["error"] = str(error)

    return result


async def fetch_panos(
    sem_pano: asyncio.Semaphore, 
    connector: aiohttp.TCPConnector, 
    max_workers: int, 
    config: dict,
    panoids: list[dict], 
    output_dir: Union[str, None] = None
) -> tuple[int, int, str]:
    """
    Download and process multiple panoramas concurrently (OPTIMIZED).
    """
    # ── Pipeline profiler ──


    print("[green]| Running OPTIMIZED Scraper with Directional Views..[/]\n")
    
    # Print configuration
    if config.get("create_directional_views"):
        msg = f"[green]| Directional Views: Enabled"
        if config.get("global_view"):
            msg += " [cyan](GLOBAL SINGLE RANDOM VIEW)[/]"
        else:
            msg += f" ({config.get('num_views', 6)} views)"
            
        if config.get("augment"):
            msg += " [magenta]+ AUGMENTED[/]"
        
        msg += f" {config.get('view_resolution', 512)}x{config.get('view_resolution', 512)}[/]"
        print(msg)
    
    if not config.get("keep_panorama"):
        print("[yellow]| Panorama Storage: In-memory only (not saving to disk)[/]")
    
    jpeg_q = config.get("jpeg_quality", 95)
    if jpeg_q != 95:
        print(f"[cyan]| JPEG Quality: {jpeg_q} (default 95)[/]")
    if config.get("no_antialias"):
        print("[cyan]| Antialiasing: DISABLED (--no-antialias)[/]")
    
    # GCS configuration
    gcs_config = config.get("gcs_config")
    if gcs_config and gcs_config.enabled:
        print(f"\n[green]Google Cloud Storage Configuration:[/]")
        print(f"  Bucket: {gcs_config.bucket_name}")
        if gcs_config.base_path:
            print(f"  Base Path: {gcs_config.base_path}")
        if gcs_config.credentials_file:
            print(f"  Credentials: {gcs_config.credentials_file}")
        else:
            print(f"  Credentials: Using default")
        if gcs_config.also_save_local:
            print(f"  Also saving locally: Yes")
    
    print()
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    config["output_dir"] = output_dir

    # ── Tile row skip optimization ──
    # When only extracting views (not keeping pano), skip tile rows outside the FOV's vertical reach.
    required_tile_rows = None
    if config.get("create_directional_views") and not config.get("keep_panorama"):
        zoom = config.get("zoom_level", 2)
        fov = config.get("view_fov", 90.0)
        aug = config.get("augment", False)
        required_tile_rows = compute_required_tile_rows(zoom, fov, aug)

        if required_tile_rows is not None:
            tiles_x, tiles_y = TILES_AXIS_COUNT[zoom]
            total_tiles = (tiles_x + 1) * (tiles_y + 1)
            fetched_tiles = len(required_tile_rows) * (tiles_x + 1)
            skipped = total_tiles - fetched_tiles
            print(f"[cyan]| Tile skip: fetching rows {sorted(required_tile_rows)} of {tiles_y + 1} "
                  f"({skipped}/{total_tiles} tiles skipped, {skipped/total_tiles*100:.0f}% saved)[/]")

    config["_required_tile_rows"] = required_tile_rows

    # Initialize GCS uploader if enabled (but NOT if --gcs-also-local is set, as that means "local only")
    gcs_uploader = None
    zip_batcher = None
    
    # If --gcs-also-local is set, skip GCP upload entirely (local-only mode)
    if config.get("gcs_also_local"):
        print("[yellow]Local-only mode enabled (--gcs-also-local). Skipping GCP upload.[/]\n")
    elif gcs_config and gcs_config.enabled and gcs_config.is_valid():
        gcs_uploader = GCSUploader()
        if not gcs_uploader.initialize(gcs_config):
            print("[yellow][WARNING] Failed to initialize GCS uploader. Will use local storage only.[/]")
            gcs_uploader = None
        else:
            # Initialize ZIP batcher for intelligent 5GB batching
            zip_batcher = ZipBatcher(
                output_dir=output_dir,
                gcs_uploader=gcs_uploader,
                gcs_config=gcs_config,
                threshold_gb=5.0,
                max_upload_workers=2  # Parallel uploads (1 active + 1 preparing)
            )
    
    # Create progress bar
    progress_bar = ProgressBar(len(panoids))
    success_count = 0
    fail_count = 0
    
    # ThreadPoolExecutor for CPU work (numpy/cv2 release the GIL — no IPC overhead)
    num_cpu_workers = max_workers
    num_upload_workers = 20

    print(f"[cyan]Optimization: Using {num_cpu_workers} thread workers for image processing (zero IPC overhead)[/]")
    print(f"[cyan]Optimization: Using {num_upload_workers} I/O workers for uploads[/]")
    
    # OPTIMIZATION: Streaming pipeline — no batch boundaries.
    # The semaphore controls concurrency; as each pano finishes and releases
    # its slot, the next one immediately starts tile I/O. This eliminates the
    # spike/stall pattern caused by discrete batch boundaries.
    print(f"[cyan]Optimization: Streaming pipeline (max {config.get('max_threads', 50)} concurrent)[/]\n")

    session_timeout = ClientTimeout(total=15, connect=8)
    async with aiohttp.ClientSession(connector=connector, timeout=session_timeout) as session:
        with ThreadPoolExecutor(max_workers=num_cpu_workers) as executor:
            with ThreadPoolExecutor(max_workers=num_upload_workers) as upload_executor:
                
                # OPTIMIZATION: Process in chunks to prevent OOM
                # Creating 800k+ tasks at once consumes GBs of RAM just for the Future objects.
                # We process in large batches (e.g. 10k) to keep the pipeline full but memory bounded.
                CHUNK_SIZE = 10000
                total_panoids = len(panoids)
                
                for i in range(0, total_panoids, CHUNK_SIZE):
                    chunk = panoids[i : i + CHUNK_SIZE]
                    # print(f"[grey50]Processing chunk {i//CHUNK_SIZE + 1}/{(total_panoids + CHUNK_SIZE - 1)//CHUNK_SIZE} ({len(chunk)} panos)...[/]")
                    
                    tasks = [
                        process_panoid(
                            session, panoid, sem_pano, executor, upload_executor,
                            config, gcs_uploader, zip_batcher
                        )
                        for panoid in chunk
                    ]

                    for coro in asyncio.as_completed(tasks):
                        try:
                            result = await coro
                            
                            if result["success"]:
                                success_count += 1
                                progress_bar.log_success(result, config)
                            else:
                                fail_count += 1
                                progress_bar.log_failure(result)

                            progress_bar.update(success_count, fail_count)
                        except Exception as e:
                            # Catch-all for any unhandled crashes in the task wrapper
                            print(f"[red]CRITICAL ERROR in worker: {e}[/]")
                            fail_count += 1
                            progress_bar.update(success_count, fail_count)
                    
                    # Optional: Force GC after each huge chunk if memory is tight
                    # import gc; gc.collect()
                
                # Finalize ZIP batcher (zip any remaining images and wait for uploads)
                if zip_batcher:
                    await zip_batcher.finalize()
    
    progress_bar.finish()



    return len(panoids), success_count, output_dir