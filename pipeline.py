#!/usr/bin/env python3
"""
VPS Pipeline: Google Street View Downloader → MegaLoc Feature Extraction → R2 Upload

Distributed worker version — no FAISS, hardcoded view settings.
Downloads its assigned CSV segment from R2, processes panos,
extracts MegaLoc features, uploads results to R2, and self-destructs.

Progress is logged to stdout in structured format for vastai logs polling:
    PROGRESS|{worker_index}|{processed}|{total}|{eta_seconds}|{speed}|{status}
"""

import asyncio
import csv
import gc
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ── Uvloop ────────────────────────────────────────────────────────────────────
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("[INFO] Using uvloop")
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration (Hardcoded per spec)
# ═══════════════════════════════════════════════════════════════════════════════

HARDCODED_CONFIG = {
    'zoom_level': 2,
    'max_threads': 150,
    'workers': 8,
    'create_directional_views': True,
    'keep_panorama': False,
    'view_resolution': 322,
    'view_fov': 60.0,
    'num_views': 8,
    'global_view': False,
    'augment': False,
    'no_antialias': True,
    'interpolation': 'cubic',
    'jpeg_quality': 85,
    'output_dir': None,
    'batch_size': 64,
    'queue_size': 2000,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Environment Variables
# ═══════════════════════════════════════════════════════════════════════════════

WORKER_INDEX = int(os.environ.get('WORKER_INDEX', '1'))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', '1'))
CSV_BUCKET_PREFIX = os.environ.get('CSV_BUCKET_PREFIX', 'CSV')
FEATURES_BUCKET_PREFIX = os.environ.get('FEATURES_BUCKET_PREFIX', 'Features')
CITY_NAME = os.environ.get('CITY_NAME', 'Unknown')
INSTANCE_ID = os.environ.get('INSTANCE_ID', '')
VAST_API_KEY = os.environ.get('VAST_API_KEY', '')

MAX_DISK_GB = 100
MIN_FREE_GB = 5

# ═══════════════════════════════════════════════════════════════════════════════
# Imports: Street View Downloader
# ═══════════════════════════════════════════════════════════════════════════════

import aiohttp
from gsvpd.core_optimized import (
    fetch_tile,
    determine_dimensions,
    _stitch_and_process_tiles,
    compute_required_tile_rows,
)
from gsvpd.constants import TILES_AXIS_COUNT, TILE_COUNT_TO_SIZE, X_COUNT_TO_SIZE
from concurrent.futures import ThreadPoolExecutor

# ═══════════════════════════════════════════════════════════════════════════════
# Imports: Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

import torch
from torchvision import transforms
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# R2 Storage
# ═══════════════════════════════════════════════════════════════════════════════

from r2_storage import R2Client

# ═══════════════════════════════════════════════════════════════════════════════
# Progress Logging
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressReporter:
    """Prints structured progress lines to stdout for vastai logs polling."""

    def __init__(self, worker_index: int, total: int, interval: float = 30.0):
        self.worker_index = worker_index
        self.total = total
        self.interval = interval
        self.processed = 0
        self.start_time = time.time()
        self._last_report = 0.0

    def update(self, processed: int, status: str = "EXTRACTING"):
        self.processed = processed
        now = time.time()
        if now - self._last_report >= self.interval:
            self._report(status)
            self._last_report = now

    def _report(self, status: str):
        elapsed = time.time() - self.start_time
        speed = self.processed / elapsed if elapsed > 0 else 0
        remaining = self.total - self.processed
        eta = remaining / speed if speed > 0 else 0
        line = f"PROGRESS|{self.worker_index}|{self.processed}|{self.total}|{eta:.0f}|{speed:.2f}|{status}"
        print(line, flush=True)

    def report_final(self, status: str):
        """Force a final progress report."""
        self._report(status)


# ═══════════════════════════════════════════════════════════════════════════════
# View Item & Shared State
# ═══════════════════════════════════════════════════════════════════════════════

_SENTINEL = None

class ViewItem:
    __slots__ = ('panoid', 'jpeg_bytes', 'lat', 'lng')
    def __init__(self, panoid: str, jpeg_bytes: bytes, lat: float, lng: float):
        self.panoid = panoid
        self.jpeg_bytes = jpeg_bytes
        self.lat = lat
        self.lng = lng

class SharedState:
    """Thread-safe writing to memmap + metadata + failures."""
    def __init__(self, features_memmap, metadata_file_path, failed_file_path, start_idx=0):
        self.memmap = features_memmap
        self.write_idx = start_idx
        self.lock = threading.Lock()
        self.metadata_handle = open(metadata_file_path, 'a', encoding='utf-8')
        self.failed_handle = open(failed_file_path, 'a', encoding='utf-8')
        self.metadata_entries = []

    def write_batch(self, features_batch: np.ndarray, metadata_batch: List[dict]):
        n = len(features_batch)
        if n == 0:
            return
        with self.lock:
            start = self.write_idx
            end = start + n
            self.memmap[start:end] = features_batch
            for meta in metadata_batch:
                meta['feature_index'] = start + len(self.metadata_entries) % n
                self.metadata_entries.append(meta)
                self.metadata_handle.write(json.dumps(meta) + '\n')
            self.metadata_handle.flush()
            self.write_idx = end

    def log_failure(self, panoid: str, reason: str):
        with self.lock:
            entry = {'panoid': panoid, 'reason': str(reason), 'timestamp': time.time()}
            self.failed_handle.write(json.dumps(entry) + '\n')
            self.failed_handle.flush()

    def close(self):
        with self.lock:
            if self.metadata_handle:
                self.metadata_handle.close()
                self.metadata_handle = None
            if self.failed_handle:
                self.failed_handle.close()
                self.failed_handle = None


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Feature Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class GpuExtractor:
    def __init__(self):
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[INFO] Loading MegaLoc model on {self.device}...")
        try:
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        except Exception:
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)

        model = model.to(self.device).eval()

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"[INFO] DataParallel: {gpu_count} GPUs")
            model = torch.nn.DataParallel(model)

        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except Exception:
                pass

        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.executor = ThreadPoolExecutor(max_workers=16)
        print("[INFO] GpuExtractor initialized")

    def extract_batch(self, items: List[ViewItem]):
        def _decode(item):
            try:
                img = Image.open(io.BytesIO(item.jpeg_bytes)).convert('RGB')
                return transforms.functional.to_tensor(img)
            except Exception:
                return None

        tensors_or_none = list(self.executor.map(_decode, items))
        valid_indices = []
        valid_tensors = []
        for i, t in enumerate(tensors_or_none):
            if t is not None:
                valid_tensors.append(t)
                valid_indices.append(i)

        if not valid_tensors:
            return None, [], []

        images = torch.stack(valid_tensors).to(self.device, non_blocking=True)
        images = torch.nn.functional.interpolate(images, size=(322, 322), mode='bilinear', align_corners=False)
        images = (images - self.mean) / self.std

        with torch.no_grad():
            feats = self.model(images)

        feats_np = feats.cpu().numpy()
        metadata_batch = [
            {'panoid': items[i].panoid, 'lat': items[i].lat, 'lng': items[i].lng}
            for i in valid_indices
        ]
        return feats_np, metadata_batch, valid_indices


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: str) -> Tuple[List[dict], Dict[str, Dict]]:
    records = []
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = ',' if sample.count(',') >= sample.count(';') else ';'
        reader = csv.DictReader(f, delimiter=delimiter)

        col_map = {}
        if reader.fieldnames:
            for field in reader.fieldnames:
                clean = field.lower().strip().replace('_', '').replace('-', '')
                if clean == 'panoid':
                    col_map['panoid'] = field
                elif clean in ('lat', 'latitude'):
                    col_map['lat'] = field
                elif clean in ('lon', 'lng', 'longitude'):
                    col_map['lon'] = field
                elif clean in ('headingdeg', 'heading', 'yaw'):
                    col_map['heading'] = field

        if 'panoid' not in col_map:
            print(f"[ERROR] No panoid column in CSV. Columns: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            panoid = row.get(col_map['panoid'], '').strip()
            if not panoid:
                continue
            record = {'panoid': panoid}
            if 'heading' in col_map and row.get(col_map['heading']):
                try:
                    record['heading_deg'] = float(row[col_map['heading']])
                except ValueError:
                    pass
            records.append(record)
            if 'lat' in col_map and 'lon' in col_map:
                try:
                    lat = float(row.get(col_map['lat'], '').strip())
                    lon = float(row.get(col_map['lon'], '').strip())
                    metadata[panoid] = {'lat': round(lat, 5), 'lng': round(lon, 5)}
                except (ValueError, AttributeError):
                    pass
    return records, metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Async Downloader
# ═══════════════════════════════════════════════════════════════════════════════

async def _download_single_pano(session, record, sem, executor, config, item_queue, metadata, stats, shared_state):
    panoid_str = record['panoid']
    heading_deg = record.get('heading_deg')
    zoom_level = config['zoom_level']

    retries = 3
    for attempt in range(1, retries + 1):
        try:
            async with sem:
                tiles_x, tiles_y = TILES_AXIS_COUNT[zoom_level]
                required_y = config.get('_required_tile_rows')
                tasks = [
                    fetch_tile(session, panoid_str, x, y, zoom_level)
                    for x in range(tiles_x + 1)
                    for y in range(tiles_y + 1)
                    if required_y is None or y in required_y
                ]
                tiles = [t for t in await asyncio.gather(*tasks) if t is not None]

                if not tiles:
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    stats['dl_fail'] += 1
                    shared_state.log_failure(panoid_str, "no_tiles")
                    return

                x_tc = len({x for x, _, _ in tiles})
                y_tc = len({y for _, y, _ in tiles})
                w, h = await determine_dimensions(executor, tiles, zoom_level, x_tc, y_tc)

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    executor, _stitch_and_process_tiles,
                    tiles, w, h, config, panoid_str, zoom_level, heading_deg
                )
                del tiles

                if not result['success'] or not result['views']:
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    stats['dl_fail'] += 1
                    shared_state.log_failure(panoid_str, "stitch_failed")
                    return

                meta = metadata.get(panoid_str, {'lat': 0.0, 'lng': 0.0})
                for view_bytes, _ in zip(result['views'], result['view_filenames']):
                    item = ViewItem(panoid_str, view_bytes, meta['lat'], meta['lng'])
                    while True:
                        try:
                            item_queue.put(item, timeout=1.0)
                            break
                        except queue.Full:
                            continue

                stats['dl_ok'] += 1
                stats['views_produced'] += len(result['views'])
                del result
                return

        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
            else:
                stats['dl_fail'] += 1
                shared_state.log_failure(panoid_str, f"exception: {e}")

async def _run_downloader(records, config, item_queue, metadata, stats, shared_state):
    from aiohttp import ClientTimeout
    sem = asyncio.Semaphore(config['max_threads'])
    connector = aiohttp.TCPConnector(limit=600, limit_per_host=200, ttl_dns_cache=300)
    timeout = ClientTimeout(total=15, connect=8)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with ThreadPoolExecutor(max_workers=config['workers']) as executor:
            CHUNK = 5000
            for i in range(0, len(records), CHUNK):
                chunk = records[i:i + CHUNK]
                tasks = [
                    _download_single_pano(session, rec, sem, executor, config,
                                          item_queue, metadata, stats, shared_state)
                    for rec in chunk
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

    item_queue.put(_SENTINEL)
    stats['dl_done'] = True

def downloader_thread(records, config, item_queue, metadata, stats, shared_state):
    asyncio.run(_run_downloader(records, config, item_queue, metadata, stats, shared_state))


# ═══════════════════════════════════════════════════════════════════════════════
# Disk Space Management
# ═══════════════════════════════════════════════════════════════════════════════

def get_free_gb(path: str = '/') -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)

def wait_for_disk_space(path: str = '/', min_gb: float = MIN_FREE_GB):
    while get_free_gb(path) < min_gb:
        print(f"[WARN] Only {get_free_gb(path):.1f}GB free, waiting for space (need {min_gb}GB)...")
        time.sleep(60)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Destruct
# ═══════════════════════════════════════════════════════════════════════════════

def self_destruct():
    """Destroy this Vast.ai instance."""
    if not INSTANCE_ID or not VAST_API_KEY:
        print("[WARN] Cannot self-destruct: INSTANCE_ID or VAST_API_KEY not set")
        return
    try:
        cmd = ["vastai", "--api-key", VAST_API_KEY, "destroy", "instance", str(INSTANCE_ID)]
        print(f"[INFO] Self-destructing instance {INSTANCE_ID}...")
        subprocess.run(cmd, timeout=30)
    except Exception as e:
        print(f"[ERROR] Self-destruct failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    work_dir = Path('/app/work')
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Worker {WORKER_INDEX}/{NUM_WORKERS} starting")
    print(f"[INFO] City: {CITY_NAME}")
    print(f"[INFO] CSV prefix: {CSV_BUCKET_PREFIX}")
    print(f"[INFO] Features prefix: {FEATURES_BUCKET_PREFIX}")

    # ── Step 1: Download CSV segment from R2 ──
    reporter = ProgressReporter(WORKER_INDEX, 0)
    reporter.report_final("DOWNLOADING_CSV")

    r2 = R2Client()
    csv_filename = f"{CITY_NAME}_{WORKER_INDEX}.{NUM_WORKERS}.csv"
    csv_key = f"{CSV_BUCKET_PREFIX}/{csv_filename}"
    local_csv = str(work_dir / csv_filename)

    print(f"[INFO] Downloading {csv_key} from R2...")
    if not r2.download_file(csv_key, local_csv, max_retries=5):
        print(f"[ERROR] Failed to download CSV: {csv_key}")
        reporter.report_final("FAILED:csv_download")
        sys.exit(1)

    # ── Step 2: Load CSV ──
    records, metadata_map = load_csv(local_csv)
    total_records = len(records)
    views_per_pano = HARDCODED_CONFIG['num_views']
    total_views_est = total_records * views_per_pano
    feature_dim = 8448

    print(f"[INFO] {total_records} panoids, ~{total_views_est} views expected")
    reporter = ProgressReporter(WORKER_INDEX, total_views_est)
    reporter.report_final("INITIALIZING")

    # ── Step 3: Setup output files ──
    features_file = str(work_dir / 'features.npy')
    metadata_file = str(work_dir / 'metadata.jsonl')
    failed_file = str(work_dir / 'failed.jsonl')

    # Resume check
    done_panoids: Set[str] = set()
    rows_done = 0
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    done_panoids.add(data['panoid'])
                    rows_done += 1
                except Exception:
                    pass
        print(f"[INFO] Resume: {rows_done} views already extracted")

    if os.path.exists(failed_file):
        with open(failed_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'panoid' in data:
                        done_panoids.add(data['panoid'])
                except Exception:
                    pass

    to_process = [r for r in records if r['panoid'] not in done_panoids]
    print(f"[INFO] Processing {len(to_process)}/{total_records} panoids")

    if not to_process:
        print("[INFO] All panoids already processed, skipping to upload")
    else:
        # Create memmap
        if os.path.exists(features_file):
            features_memmap = np.lib.format.open_memmap(features_file, mode='r+')
        else:
            features_memmap = np.lib.format.open_memmap(
                features_file, mode='w+', dtype='float32',
                shape=(total_views_est, feature_dim)
            )

        # Build config
        dl_config = dict(HARDCODED_CONFIG)
        dl_config['_required_tile_rows'] = compute_required_tile_rows(
            dl_config['zoom_level'], dl_config['view_fov'], dl_config['augment']
        )

        # Shared state
        shared_state = SharedState(features_memmap, metadata_file, failed_file, start_idx=rows_done)

        # Queue & stats
        item_queue = queue.Queue(maxsize=dl_config['queue_size'])
        stats = {'dl_ok': 0, 'dl_fail': 0, 'ext_ok': rows_done, 'views_produced': 0, 'dl_done': False}

        # Init GPU extractor
        print("[INFO] Initializing GPU extractor...")
        extractor = GpuExtractor()

        # Start downloader thread
        dl_thread = threading.Thread(
            target=downloader_thread,
            args=(to_process, dl_config, item_queue, metadata_map, stats, shared_state)
        )
        dl_thread.start()

        # ── Extraction loop ──
        batch_size = dl_config['batch_size']
        current_batch = []

        try:
            while True:
                # Check disk space
                wait_for_disk_space(str(work_dir), MIN_FREE_GB)

                # Fill batch
                while len(current_batch) < batch_size:
                    try:
                        item = item_queue.get(timeout=0.01)
                        if item is _SENTINEL:
                            continue
                        current_batch.append(item)
                    except queue.Empty:
                        if not dl_thread.is_alive():
                            break
                        else:
                            break

                if not current_batch:
                    if not dl_thread.is_alive():
                        break
                    continue

                # Extract features
                try:
                    feats_np, meta_batch, valid_indices = extractor.extract_batch(current_batch)
                    if feats_np is not None and len(meta_batch) > 0:
                        shared_state.write_batch(feats_np, meta_batch)
                        stats['ext_ok'] += len(meta_batch)
                        reporter.update(stats['ext_ok'], "EXTRACTING")
                    current_batch = []
                except Exception as e:
                    print(f"[ERROR] Batch extraction failed: {e}")
                    current_batch = []

        except KeyboardInterrupt:
            print("[WARN] Interrupted")

        dl_thread.join()
        final_count = shared_state.write_idx
        shared_state.close()

        # Truncate memmap to actual size
        del features_memmap
        gc.collect()

        if final_count > 0 and final_count < total_views_est:
            print(f"[INFO] Truncating features: {total_views_est} → {final_count}")
            mm = np.lib.format.open_memmap(features_file, mode='r+')
            truncated = mm[:final_count].copy()
            del mm
            np.save(features_file, truncated)
            del truncated

        print(f"[INFO] Extraction complete: {final_count} features extracted")

    # ── Step 4: Upload to R2 ──
    reporter.report_final("UPLOADING")
    print("[INFO] Uploading features to R2...")

    npy_key = f"{FEATURES_BUCKET_PREFIX}/{CITY_NAME}_{WORKER_INDEX}.{NUM_WORKERS}.npy"
    meta_key = f"{FEATURES_BUCKET_PREFIX}/Metadata_{CITY_NAME}_{WORKER_INDEX}.{NUM_WORKERS}.jsonl"

    # Retry uploads up to 5 times, then indefinitely every 60s
    def upload_with_retry(local_path, bucket_key, max_attempts=5):
        for attempt in range(1, max_attempts + 1):
            if r2.upload_file(local_path, bucket_key, max_retries=3):
                return True
            print(f"[WARN] Upload attempt {attempt}/{max_attempts} failed for {bucket_key}")
            time.sleep(2 ** attempt)
        # Indefinite retry
        while True:
            print(f"[WARN] Retrying {bucket_key} indefinitely (every 60s)...")
            if r2.upload_file(local_path, bucket_key, max_retries=3):
                return True
            time.sleep(60)

    success = True
    if os.path.exists(features_file) and os.path.getsize(features_file) > 0:
        if not upload_with_retry(features_file, npy_key):
            success = False
    if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
        if not upload_with_retry(metadata_file, meta_key):
            success = False

    if success:
        reporter.report_final("COMPLETED")
        print("[INFO] Upload complete! Self-destructing...")
        # Cleanup local files
        for f in [features_file, metadata_file, failed_file, local_csv]:
            try:
                os.remove(f)
            except Exception:
                pass
        self_destruct()
    else:
        reporter.report_final("FAILED:upload")
        print("[ERROR] Upload failed. Instance kept alive for debugging.")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[CRITICAL] {e}")
        import traceback
        traceback.print_exc()
        # Report failure
        reporter = ProgressReporter(WORKER_INDEX, 0)
        reporter.report_final(f"FAILED:{e}")
        sys.exit(1)
