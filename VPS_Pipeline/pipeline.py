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
import math
from typing import Dict, List, Set, Tuple

import numpy as np

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ── Uvloop Disabled (Stability Issues) ──────────────────────────────────────────
# try:
#     import uvloop
#     asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
#     print("[INFO] Using uvloop")
# except ImportError:
#     pass


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
    'jpeg_quality': 95,
    'output_dir': None,
    'batch_size': 32,
    'queue_size': 512,
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
# R2 Status Reporter
# ═══════════════════════════════════════════════════════════════════════════════

class R2StatusReporter:
    """Uploads structured status JSON to R2 for UI polling."""

    def __init__(self, r2_client, worker_index: int, total: int,
                 status_prefix: str, instance_id: str = '',
                 interval: float = 15.0):
        self.r2 = r2_client
        self.worker_index = worker_index
        self.total = total
        self.status_prefix = status_prefix
        self.instance_id = instance_id
        self.interval = interval
        self.processed = 0
        self.start_time = time.time()
        self._last_report = 0.0
        self._status_key = f"Status/{status_prefix}/worker_{worker_index}.json"

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
        data = {
            'worker': self.worker_index,
            'status': status,
            'processed': self.processed,
            'total': self.total,
            'speed': round(speed, 2),
            'eta': round(eta),
            'instance_id': self.instance_id,
            'timestamp': time.time(),
        }
        # Print to stdout as well for debugging
        print(f"PROGRESS|{self.worker_index}|{self.processed}|{self.total}|{eta:.0f}|{speed:.2f}|{status}", flush=True)
        try:
            self.r2.upload_json(self._status_key, data)
        except Exception as e:
            print(f"[WARN] Status upload failed: {e}")

    def report_final(self, status: str):
        """Force a final status report."""
        self._report(status)

    def report_upload(self, bytes_done: int, bytes_total: int,
                      speed_mb: float, eta: float, label: str = ""):
        """Report upload byte-progress to R2 and stdout.

        Args:
            bytes_done: Bytes transferred so far.
            bytes_total: Total file size in bytes.
            speed_mb: Current transfer speed in MB/s.
            eta: Estimated seconds remaining.
            label: File label for stdout (e.g. 'NPY', 'META').
        """
        status = f"UPLOADING_{label}" if label else "UPLOADING"
        data = {
            'worker': self.worker_index,
            'status': 'UPLOADING',
            'processed': bytes_done,
            'total': bytes_total,
            'speed': round(speed_mb, 2),
            'eta': round(eta),
            'instance_id': self.instance_id,
            'timestamp': time.time(),
        }
        pct = int(bytes_done / bytes_total * 100) if bytes_total > 0 else 0
        mb_done = bytes_done / (1024 * 1024)
        mb_total = bytes_total / (1024 * 1024)
        print(
            f"PROGRESS|{self.worker_index}|{bytes_done}|{bytes_total}"
            f"|{eta:.0f}|{speed_mb:.2f}|{status}"
            f"  [{pct}%  {mb_done:.0f}/{mb_total:.0f} MB  {speed_mb:.1f} MB/s]",
            flush=True,
        )
        try:
            self.r2.upload_json(self._status_key, data)
        except Exception as e:
            print(f"[WARN] Upload status report failed: {e}")


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
        self._batch_count = 0

    def write_batch(self, features_batch: np.ndarray, metadata_batch: List[dict]):
        n = len(features_batch)
        if n == 0:
            return
        with self.lock:
            start = self.write_idx
            end = start + n
            self.memmap[start:end] = features_batch
            for i, meta in enumerate(metadata_batch):
                meta['feature_index'] = start + i
                self.metadata_handle.write(json.dumps(meta) + '\n')
            self.metadata_handle.flush()
            self.write_idx = end
            self._batch_count += 1
            # Flush memmap to disk periodically to keep RSS low
            if self._batch_count % 100 == 0:
                self.memmap.flush()

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

class _InitWatchdog:
    """Watchdog that kills the process if init hangs too long."""
    def __init__(self, timeout_sec: int, stage: str = "unknown"):
        self.timeout = timeout_sec
        self.stage = stage
        self._timer = None

    def start(self, stage: str = None):
        if stage:
            self.stage = stage
        self.cancel()
        self._timer = threading.Timer(self.timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timeout(self):
        msg = (f"[FATAL] Watchdog timeout after {self.timeout}s during: {self.stage}. "
               f"Process is stuck — forcing exit.")
        print(msg, flush=True)
        # Report failure to R2 before dying
        try:
            r2 = R2Client()
            status_key = f"Status/{FEATURES_BUCKET_PREFIX}/worker_{WORKER_INDEX}.json"
            r2.upload_json(status_key, {
                'worker': WORKER_INDEX,
                'status': f'FAILED:watchdog_timeout_{self.stage}',
                'detail': msg,
                'timestamp': time.time(),
                'instance_id': INSTANCE_ID,
            })
        except Exception:
            pass
        try:
            upload_logs_to_r2()
        except Exception:
            pass
        os._exit(1)


def _run_with_timeout(fn, timeout_sec: int, stage: str):
    """Run fn() in a thread with a timeout. Raises TimeoutError if it hangs."""
    result_container = [None]
    error_container = [None]

    def _target():
        try:
            result_container[0] = fn()
        except Exception as e:
            error_container[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise TimeoutError(
            f"{stage} timed out after {timeout_sec}s — process is hung. "
            f"This usually means a network stall (model download) or CUDA driver issue."
        )

    if error_container[0] is not None:
        raise error_container[0]

    return result_container[0]


GPU_INIT_TIMEOUT = int(os.environ.get('GPU_INIT_TIMEOUT', '300'))  # 5 min default


class GpuExtractor:
    def __init__(self):
        t0 = time.time()
        watchdog = _InitWatchdog(GPU_INIT_TIMEOUT + 30, "gpu_init_overall")
        watchdog.start()

        try:
            self._init_gpu(t0)
        finally:
            watchdog.cancel()

    @staticmethod
    def _download_model_with_fallback(timeout_sec: int):
        """Download MegaLoc model with multiple fallback sources.

        Strategy:
          1. torch.hub (GitHub + HuggingFace) — two attempts with backoff
          2. R2 fallback — download safetensors from R2 bucket if configured
          3. Direct HuggingFace URL — bypasses torch.hub plumbing

        Returns the loaded model (on CPU).
        """
        errors = []

        # Set HF token for authenticated downloads (higher rate limits)
        # Token is injected via HF_TOKEN env var in the worker deployment config
        _hf_token = os.environ.get("HF_TOKEN", "")
        if _hf_token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

        # ── Source 1: torch.hub (standard path) ──
        def _try_torch_hub():
            for attempt in range(1, 4):
                try:
                    print(f"[INIT]   torch.hub attempt {attempt}/3...", flush=True)
                    return torch.hub.load(
                        "gmberton/MegaLoc", "get_trained_model", trust_repo=True
                    )
                except Exception as e:
                    print(f"[INIT]   torch.hub attempt {attempt} failed: "
                          f"{type(e).__name__}: {e}", flush=True)
                    errors.append(f"torch.hub#{attempt}: {e}")
                    if attempt < 3:
                        time.sleep(2 ** attempt)
            return None

        try:
            model = _run_with_timeout(_try_torch_hub, timeout_sec, "model_download_hub")
            if model is not None:
                print("[INIT]   Model loaded via torch.hub", flush=True)
                return model
        except TimeoutError:
            errors.append(f"torch.hub timed out after {timeout_sec}s")
            print(f"[WARN] torch.hub timed out after {timeout_sec}s", flush=True)
        except Exception as e:
            errors.append(f"torch.hub: {e}")

        # ── Source 2: R2 fallback ──
        r2_model_key = os.environ.get(
            'R2_MODEL_KEY', 'Models/MegaLoc/model.safetensors'
        )
        local_model_path = '/tmp/megaloc_model.pt'

        try:
            print(f"[INIT]   Trying R2 fallback: {r2_model_key}...", flush=True)
            r2 = R2Client()
            if r2.download_file(r2_model_key, local_model_path, max_retries=3):
                size_mb = os.path.getsize(local_model_path) / 1e6
                print(f"[INIT]   Downloaded model from R2 ({size_mb:.1f}MB)", flush=True)

                # Try loading as a full torch checkpoint first (model.torch),
                # then as safetensors state_dict
                model = None
                try:
                    loaded = torch.load(local_model_path, map_location='cpu', weights_only=False)
                    if isinstance(loaded, torch.nn.Module):
                        model = loaded
                        print("[INIT]   Loaded complete model from R2 (torch checkpoint)", flush=True)
                    elif isinstance(loaded, dict):
                        # It's a state_dict — need architecture
                        state_dict = loaded
                    else:
                        state_dict = loaded
                except Exception:
                    # Try safetensors format
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(local_model_path)
                    except Exception:
                        state_dict = None

                if model is not None:
                    return model

                # Have state_dict but need architecture — try cached hub repo
                if state_dict is not None:
                    hub_dir = Path(torch.hub.get_dir()) / 'gmberton_MegaLoc_main'
                    if not hub_dir.exists():
                        try:
                            print("[INIT]   Downloading model architecture from GitHub...", flush=True)
                            torch.hub._get_cache_or_reload(
                                "gmberton/MegaLoc", force_reload=False,
                                trust_repo=True, calling_fn="load"
                            )
                        except Exception as e2:
                            errors.append(f"R2 architecture download: {e2}")
                            print(f"[WARN] Could not download architecture: {e2}", flush=True)

                    if hub_dir.exists():
                        sys.path.insert(0, str(hub_dir))
                        try:
                            import importlib
                            hubconf = importlib.import_module('hubconf')
                            model = hubconf.get_trained_model()
                            model.load_state_dict(state_dict, strict=False)
                            print("[INIT]   Model loaded via R2 state_dict + hub architecture", flush=True)
                            return model
                        except Exception as e2:
                            errors.append(f"R2 architecture load: {e2}")
                            print(f"[WARN] Hub architecture load failed: {e2}", flush=True)
                        finally:
                            sys.path.pop(0)
            else:
                errors.append("R2 model download returned False (file may not exist)")
                print("[WARN] R2 model file not found or download failed", flush=True)
        except Exception as e:
            errors.append(f"R2 fallback: {e}")
            print(f"[WARN] R2 fallback failed: {e}", flush=True)

        # ── Source 3: Direct HuggingFace download via urllib (bypass httpx/torch.hub) ──
        try:
            import urllib.request
            hf_url = "https://huggingface.co/gberton/MegaLoc/resolve/main/model.safetensors"
            print(f"[INIT]   Trying direct HuggingFace download...", flush=True)

            def _direct_download():
                req = urllib.request.Request(
                    hf_url,
                    headers={"Authorization": f"Bearer {_hf_token}"}
                )
                with urllib.request.urlopen(req) as resp, open(local_model_path, 'wb') as f:
                    f.write(resp.read())
                return True

            _run_with_timeout(_direct_download, timeout_sec, "model_download_direct_hf")
            if os.path.exists(local_model_path) and os.path.getsize(local_model_path) > 1e6:
                print(f"[INIT]   Direct HF download OK ({os.path.getsize(local_model_path) / 1e6:.1f}MB)", flush=True)
                # Now load via torch.hub using cached repo + downloaded weights
                model = torch.hub.load(
                    "gmberton/MegaLoc", "get_trained_model", trust_repo=True
                )
                print("[INIT]   Model loaded via direct HuggingFace download", flush=True)
                return model
        except Exception as e:
            errors.append(f"Direct HF: {e}")
            print(f"[WARN] Direct HuggingFace download failed: {e}", flush=True)

        # All sources exhausted
        raise RuntimeError(
            f"All model download sources failed after {len(errors)} attempts:\n"
            + "\n".join(f"  - {err}" for err in errors)
            + "\n\nTo fix: upload MegaLoc model.safetensors to R2 at "
            + f"'{r2_model_key}', or ensure outbound network access to "
            + "github.com and huggingface.co."
        )

    def _init_gpu(self, t0: float):
        # ── Step 1: CUDA check ──
        print(f"[INIT] Step 1/5: Checking CUDA availability...", flush=True)
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise RuntimeError(
                "CUDA is not available! Check nvidia-smi, CUDA drivers, and container GPU passthrough. "
                "torch.cuda.is_available() returned False."
            )
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
        print(f"[INIT]   CUDA OK — GPU: {gpu_name}, VRAM: {gpu_mem:.1f}GB", flush=True)
        print(f"[INIT]   CUDA version: {torch.version.cuda}, PyTorch: {torch.__version__}", flush=True)

        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cuda')

        # ── Step 2: Download model ──
        print(f"[INIT] Step 2/5: Downloading MegaLoc model (timeout={GPU_INIT_TIMEOUT}s)...", flush=True)
        dl_start = time.time()
        model = self._download_model_with_fallback(GPU_INIT_TIMEOUT)
        print(f"[INIT]   Model ready in {time.time() - dl_start:.1f}s", flush=True)

        # ── Step 3: Move to GPU ──
        print(f"[INIT] Step 3/5: Moving model to {self.device}...", flush=True)
        move_start = time.time()
        try:
            model = _run_with_timeout(
                lambda: model.to(self.device).eval(),
                timeout_sec=120,
                stage="model_to_cuda"
            )
        except TimeoutError:
            raise RuntimeError(
                "model.to(cuda) timed out after 120s. CUDA driver may be unresponsive. "
                "Check nvidia-smi and dmesg for GPU errors."
            )
        print(f"[INIT]   Model on GPU in {time.time() - move_start:.1f}s", flush=True)

        # ── Step 4: DataParallel / compile ──
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"[INIT] Step 4/5: Wrapping with DataParallel ({gpu_count} GPUs)...", flush=True)
            model = torch.nn.DataParallel(model)

        if hasattr(torch, 'compile'):
            print(f"[INIT] Step 4/5: torch.compile()...", flush=True)
            compile_start = time.time()
            try:
                model = _run_with_timeout(
                    lambda: torch.compile(model),
                    timeout_sec=120,
                    stage="torch_compile"
                )
                print(f"[INIT]   torch.compile() done in {time.time() - compile_start:.1f}s", flush=True)
            except TimeoutError:
                print(f"[WARN] torch.compile() timed out after 120s — running without compilation (this is OK)", flush=True)
            except Exception as e:
                print(f"[WARN] torch.compile() failed: {type(e).__name__}: {e} — running without compilation", flush=True)
        else:
            print(f"[INIT] Step 4/5: torch.compile not available (PyTorch < 2.0), skipping", flush=True)

        # ── Step 5: Warmup inference ──
        print(f"[INIT] Step 5/5: Warmup inference...", flush=True)
        warmup_start = time.time()
        try:
            dummy = torch.randn(1, 3, 322, 322, device=self.device)
            dummy = (dummy - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                    torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            with torch.no_grad():
                _ = model(dummy)
            del dummy
            torch.cuda.synchronize()
            print(f"[INIT]   Warmup done in {time.time() - warmup_start:.1f}s", flush=True)
        except Exception as e:
            # torch.compile inductor can crash on low-VRAM GPUs (BrokenProcessPool).
            # Fall back to eager mode — this is safe, just slower compilation.
            print(f"[WARN] Compiled warmup failed: {type(e).__name__}: {e}", flush=True)
            print(f"[WARN] Falling back to eager mode (disabling torch.compile)...", flush=True)

            # Reset dynamo state and unwrap the compiled model
            try:
                torch._dynamo.reset()
            except Exception:
                pass

            # Get the original uncompiled model
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
                print(f"[WARN] Unwrapped compiled model to original module", flush=True)

            # Retry warmup in eager mode
            try:
                torch.cuda.empty_cache()
                dummy = torch.randn(1, 3, 322, 322, device=self.device)
                dummy = (dummy - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                        torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                with torch.no_grad():
                    _ = model(dummy)
                del dummy
                torch.cuda.synchronize()
                print(f"[INIT]   Eager warmup OK in {time.time() - warmup_start:.1f}s", flush=True)
            except Exception as e2:
                raise RuntimeError(
                    f"Warmup inference failed in both compiled AND eager mode: {type(e2).__name__}: {e2}. "
                    f"The model may be incompatible with this GPU or CUDA version."
                )

        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.executor = ThreadPoolExecutor(max_workers=16)

        # ── Step 6: Auto batch size probe ──
        print(f"[INIT] Step 6/6: Probing optimal batch size via VRAM measurement...", flush=True)
        self.batch_size = self._probe_max_batch_size()

        vram_used = torch.cuda.memory_allocated(0) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"[INIT] GpuExtractor ready — total init: {time.time() - t0:.1f}s, "
              f"VRAM: {vram_used:.2f}GB used / {vram_reserved:.2f}GB reserved, "
              f"batch_size={self.batch_size}", flush=True)

    def _probe_max_batch_size(self) -> int:
        """Run a single-image inference under autocast, measure peak VRAM delta,
        then compute the largest power-of-2 batch that fits in 75% of free VRAM."""
        try:
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated(0)
            torch.cuda.reset_peak_memory_stats(0)

            dummy = torch.randn(1, 3, 322, 322, device=self.device)
            dummy = (dummy - self.mean) / self.std
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    _ = self.model(dummy)
            torch.cuda.synchronize()

            peak = torch.cuda.max_memory_allocated(0)
            del dummy
            torch.cuda.empty_cache()

            per_image = max(peak - baseline, 1)
            free = torch.cuda.mem_get_info(0)[0]
            max_batch = max(8, int(free * 0.75 / per_image))
            max_batch = min(max_batch, 512)
            # Round down to nearest power of 2 for alignment
            max_batch = 2 ** int(math.log2(max_batch))

            print(f"[INIT]   Auto batch size: {max_batch} "
                  f"(per-image={per_image / 1e6:.1f} MB, free={free / 1e6:.0f} MB)", flush=True)
            return max_batch
        except Exception as e:
            print(f"[WARN] Batch size probe failed ({e}), defaulting to 32", flush=True)
            return 32

    @staticmethod
    def _decode_item(item: ViewItem):
        """Decode a single ViewItem JPEG → float32 CHW CPU tensor."""
        try:
            img = Image.open(io.BytesIO(item.jpeg_bytes)).convert('RGB')
            return transforms.functional.to_tensor(img)
        except Exception as e:
            print(f"[WARN] Decode failed panoid={item.panoid}: {type(e).__name__}: {e}", flush=True)
            return None

    def start_decode(self, items: List[ViewItem]) -> list:
        """Non-blocking: submit JPEG decode for all items; return list of futures.
        Call infer_prefetched() later to collect results and run GPU inference."""
        return [self.executor.submit(self._decode_item, item) for item in items]

    def _run_inference(self, items: List[ViewItem], valid_tensors: list, valid_indices: list):
        """GPU inference with pin_memory transfer, fp16 autocast, and OOM auto-retry.

        On OutOfMemoryError the batch is split in two and each half retried;
        self.batch_size is also shrunk so future batches stay safe.
        """
        try:
            # pin_memory() → page-locked host buffer → async DMA to GPU
            images = torch.stack(valid_tensors).pin_memory().to(self.device, non_blocking=True)
            images = torch.nn.functional.interpolate(
                images, size=(322, 322), mode='bilinear', align_corners=False
            )
            images = (images - self.mean) / self.std

            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    feats = self.model(images)
            del images

            # Convert fp16 → fp32 for storage (L2-normalised anyway, no accuracy loss)
            feats_np = feats.float().cpu().numpy()
            metadata_batch = [
                {'panoid': items[i].panoid, 'lat': items[i].lat, 'lng': items[i].lng}
                for i in valid_indices
            ]
            return feats_np, metadata_batch, valid_indices

        except torch.cuda.OutOfMemoryError:
            half = len(valid_tensors) // 2
            torch.cuda.empty_cache()
            if half == 0:
                print("[WARN] OOM on single image — skipping", flush=True)
                return None, [], []

            new_bs = max(8, half)
            print(f"[WARN] OOM on batch={len(valid_tensors)} → retrying as 2×{half}, "
                  f"shrinking batch_size {self.batch_size} → {new_bs}", flush=True)
            self.batch_size = new_bs

            f1, m1, vi1 = self._run_inference(items, valid_tensors[:half], valid_indices[:half])
            f2, m2, vi2 = self._run_inference(items, valid_tensors[half:], valid_indices[half:])

            if f1 is None and f2 is None:
                return None, [], []
            if f1 is None:
                return f2, m2, vi2
            if f2 is None:
                return f1, m1, vi1
            return np.concatenate([f1, f2], axis=0), m1 + m2, vi1 + vi2

    def infer_prefetched(self, items: List[ViewItem], futures: list):
        """Block on decode futures collected by start_decode(), then run GPU inference."""
        tensors_or_none = [f.result() for f in futures]
        valid_indices = [i for i, t in enumerate(tensors_or_none) if t is not None]
        valid_tensors = [tensors_or_none[i] for i in valid_indices]

        failures = len(items) - len(valid_tensors)
        if failures:
            print(f"[WARN] {failures}/{len(items)} images failed to decode in batch", flush=True)
        if not valid_tensors:
            return None, [], []
        return self._run_inference(items, valid_tensors, valid_indices)

    def extract_batch(self, items: List[ViewItem]):
        """Synchronous decode + GPU inference (no prefetch). Kept for compatibility."""
        tensors_or_none = list(self.executor.map(self._decode_item, items))
        valid_indices = [i for i, t in enumerate(tensors_or_none) if t is not None]
        valid_tensors = [tensors_or_none[i] for i in valid_indices]

        failures = len(items) - len(valid_tensors)
        if failures:
            print(f"[WARN] {failures}/{len(items)} images failed to decode in batch", flush=True)
        if not valid_tensors:
            return None, [], []
        return self._run_inference(items, valid_tensors, valid_indices)


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
# Log Capture & Upload
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FILE = f"/tmp/worker_{WORKER_INDEX}.log"


class TeeWriter:
    """Writes to both the original stream and a log file."""
    def __init__(self, original, log_file_handle):
        self.original = original
        self.log_file = log_file_handle

    def write(self, data):
        self.original.write(data)
        try:
            self.log_file.write(data)
        except Exception:
            pass

    def flush(self):
        self.original.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    # Delegate everything else to original stream
    def __getattr__(self, name):
        return getattr(self.original, name)


def _start_log_capture():
    """Tee stdout and stderr to a log file."""
    try:
        fh = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
        sys.stdout = TeeWriter(sys.__stdout__, fh)
        sys.stderr = TeeWriter(sys.__stderr__, fh)
        return fh
    except Exception as e:
        print(f"[WARN] Could not start log capture: {e}")
        return None


def upload_logs_to_r2():
    """Upload the captured log file to R2."""
    try:
        # Flush before uploading
        sys.stdout.flush()
        sys.stderr.flush()

        r2 = R2Client()
        log_key = f"Logs/{FEATURES_BUCKET_PREFIX}/worker_{WORKER_INDEX}.log"
        r2.upload_file(LOG_FILE, log_key)
        print(f"[INFO] Uploaded logs to R2: {log_key}")
    except Exception as e:
        print(f"[WARN] Failed to upload logs to R2: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Destruct
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_instance_id():
    """Detect this worker's instance ID from R2 (written by the UI at creation time).

    The old approach of `vastai show instances` + `instances[0]` was broken:
    with multiple workers, ALL of them would pick the same first instance,
    causing workers to destroy each other instead of themselves.
    """
    # Method 1: Read from R2 (reliable — UI writes worker-specific instance ID)
    try:
        r2 = R2Client()
        key = f"Status/{FEATURES_BUCKET_PREFIX}/worker_{WORKER_INDEX}_instance.json"
        data = r2.download_json(key)
        if data and data.get('instance_id'):
            detected = str(data['instance_id'])
            print(f"[INFO] Got INSTANCE_ID from R2: {detected}")
            return detected
    except Exception as e:
        print(f"[WARN] R2 instance ID lookup failed: {e}")

    # Method 2: Fallback — only safe if there's exactly one instance
    if not VAST_API_KEY:
        return None
    try:
        result = subprocess.run(
            ["vastai", "--api-key", VAST_API_KEY, "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            import json as _json
            instances = _json.loads(result.stdout.strip())
            if instances and len(instances) == 1:
                detected = str(instances[0].get('id', ''))
                if detected:
                    print(f"[INFO] Auto-detected INSTANCE_ID (sole instance): {detected}")
                    return detected
            if instances and len(instances) > 1:
                print(f"[WARN] {len(instances)} instances running — cannot auto-detect safely. "
                      "INSTANCE_ID should be set via R2 or env var.")
    except Exception as e:
        print(f"[WARN] Instance ID auto-detect failed: {e}")
    return None


def self_destruct():
    """Destroy this Vast.ai instance — retries forever until the instance is gone."""
    instance_id = _detect_instance_id() or INSTANCE_ID

    if not instance_id:
        print("[WARN] Cannot self-destruct: unable to determine INSTANCE_ID — sleeping forever to prevent restart loop")
        while True:
            time.sleep(3600)
    if not VAST_API_KEY:
        print("[WARN] Cannot self-destruct: VAST_API_KEY not set — sleeping forever to prevent restart loop")
        while True:
            time.sleep(3600)

    cmd = ["vastai", "--api-key", VAST_API_KEY, "destroy", "instance", str(instance_id)]
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[INFO] Self-destruct attempt {attempt} for instance {instance_id}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            print(f"[INFO] Self-destruct response: exit={result.returncode} stdout='{stdout}' stderr='{stderr}'")
            if result.returncode == 0:
                print(f"[INFO] Instance {instance_id} destroyed successfully.")
                return
            print(f"[WARN] Self-destruct attempt {attempt} failed (exit {result.returncode}) — retrying in 30s")
        except Exception as e:
            print(f"[WARN] Self-destruct attempt {attempt} exception: {e} — retrying in 30s")
        time.sleep(30)


# ═══════════════════════════════════════════════════════════════════════════════
# Resume Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_last_written_row(features_path: str) -> int:
    """
    Scan .npy backwards to find the last non-zero row.
    Returns the index of the NEXT writable position (last_nonzero + 1).
    If file is all zeros, returns 0.
    """
    print("[RESUME] Scanning features.npy for last written row...")
    features = np.load(features_path, mmap_mode='r')
    n_rows = features.shape[0]

    block_size = 1000
    last_nonzero = -1

    for block_start in range(max(0, n_rows - block_size), -1, -block_size):
        block_end = min(block_start + block_size, n_rows)
        block = np.array(features[block_start:block_end])
        row_sums = np.abs(block).sum(axis=1)
        nonzero_mask = row_sums > 0
        if np.any(nonzero_mask):
            last_in_block = np.where(nonzero_mask)[0][-1]
            last_nonzero = block_start + last_in_block
            break

    del features
    gc.collect()

    if last_nonzero == -1:
        print("[RESUME]   Features file is all zeros — no valid data.")
        return 0

    resume_from = last_nonzero + 1
    print(f"[RESUME]   Last non-zero row: {last_nonzero}")
    print(f"[RESUME]   Next writable index: {resume_from}")
    return resume_from


def truncate_metadata_file(metadata_path: str, keep_lines: int) -> Set[str]:
    """
    Truncate metadata JSONL to exactly `keep_lines` lines.
    Returns set of panoids from the kept lines.
    """
    kept_panoids: Set[str] = set()
    kept_lines_data = []

    if not os.path.exists(metadata_path):
        return kept_panoids

    with open(metadata_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= keep_lines:
                break
            kept_lines_data.append(line)
            try:
                data = json.loads(line)
                kept_panoids.add(data.get('panoid', ''))
            except Exception:
                pass

    # Rewrite file with only the kept lines
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for line in kept_lines_data:
            f.write(line)

    print(f"[RESUME] Truncated metadata to {len(kept_lines_data)} lines ({len(kept_panoids)} panoids)")
    return kept_panoids


def check_and_resume(r2, work_dir: Path):
    """
    Check R2 status and local files to determine resume strategy.

    Returns:
        (strategy, rows_done, done_panoids)
        strategy: "FRESH" | "RESUME" | "EXIT"
        rows_done: int — number of valid rows to start from
        done_panoids: Set[str] — panoids already processed (for filtering records)
    """
    status_key = f"Status/{FEATURES_BUCKET_PREFIX}/worker_{WORKER_INDEX}.json"
    features_file = str(work_dir / 'features.npy')
    metadata_file = str(work_dir / 'metadata.jsonl')
    failed_file = str(work_dir / 'failed.jsonl')

    print("[RESUME] ── Checking R2 status for prior run ──")
    status_data = r2.download_json(status_key)

    if not status_data:
        print("[RESUME] No prior status found. Starting fresh.")
        return "FRESH", 0, set()

    prev_status = status_data.get('status', 'UNKNOWN')
    prev_processed = status_data.get('processed', 0)
    prev_total = status_data.get('total', 0)
    print(f"[RESUME] Prior status: {prev_status}, processed: {prev_processed}/{prev_total}")

    # ── Case 1: Already completed → self-destruct ──
    if prev_status == "COMPLETED":
        print("[RESUME] Previous run COMPLETED. Self-destructing...")
        upload_logs_to_r2()
        self_destruct()
        sys.exit(0)

    # ── Case 2: Failed or unknown → just restart fresh ──
    if prev_status.startswith("FAILED") or prev_status == "UNKNOWN":
        print(f"[RESUME] Prior status is '{prev_status}'. Clearing status and starting fresh.")
        r2.delete_object(status_key)
        # Also clear any stale local files
        for f in [features_file, metadata_file, failed_file]:
            if os.path.exists(f):
                os.remove(f)
                print(f"[RESUME]   Removed stale {os.path.basename(f)}")
        return "FRESH", 0, set()

    # ── Case 3: Has progress ──
    if prev_processed > 0:
        has_features = os.path.exists(features_file) and os.path.getsize(features_file) > 0
        has_metadata = os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0

        if not has_features or not has_metadata:
            # Status says progress but no local files → wipe and restart
            print("[RESUME] Status shows progress but no local files found!")
            print("[RESUME] Clearing R2 status and starting fresh.")
            r2.delete_object(status_key)
            for f in [features_file, metadata_file, failed_file]:
                if os.path.exists(f):
                    os.remove(f)
            return "FRESH", 0, set()

        # Both files exist → compute safe resume point with rollback
        print("[RESUME] Local files found. Computing safe resume point...")

        # Count metadata lines
        meta_count = 0
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for _ in f:
                meta_count += 1
        print(f"[RESUME]   metadata.jsonl has {meta_count} lines")

        # Scan features for last written row
        feat_count = find_last_written_row(features_file)
        print(f"[RESUME]   features.npy has {feat_count} written rows")

        # Take the minimum to be safe (metadata and features must align)
        safe_count = min(meta_count, feat_count)

        # Roll back by one batch for safety (crash may have been mid-batch)
        batch_size = HARDCODED_CONFIG['batch_size']
        rollback = min(batch_size, safe_count)
        safe_count = max(0, safe_count - rollback)
        print(f"[RESUME]   Rolled back {rollback} rows → safe_count = {safe_count}")

        if safe_count == 0:
            # Rollback brought us to zero → just start fresh
            print("[RESUME] Rollback brought count to 0. Starting fresh.")
            r2.delete_object(status_key)
            for f in [features_file, metadata_file, failed_file]:
                if os.path.exists(f):
                    os.remove(f)
            return "FRESH", 0, set()

        # Truncate metadata to safe_count lines & collect panoids
        done_panoids = truncate_metadata_file(metadata_file, safe_count)

        # Also collect panoids from failed.jsonl (don't retry known failures)
        if os.path.exists(failed_file):
            with open(failed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'panoid' in data:
                            done_panoids.add(data['panoid'])
                    except Exception:
                        pass

        print(f"[RESUME] ✓ Will resume from row {safe_count} ({len(done_panoids)} panoids done)")
        return "RESUME", safe_count, done_panoids

    # ── Case 4: No progress yet (status like DOWNLOADING_CSV, INITIALIZING) ──
    print("[RESUME] Prior run had no progress. Starting fresh.")
    r2.delete_object(status_key)
    return "FRESH", 0, set()


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

    # ── Step 0: Check for prior run & decide resume strategy ──
    r2 = R2Client()
    status_prefix = FEATURES_BUCKET_PREFIX

    # Resolve correct instance ID early (R2 lookup, then env var fallback)
    global INSTANCE_ID
    resolved_id = _detect_instance_id() or INSTANCE_ID
    if resolved_id and resolved_id != INSTANCE_ID:
        print(f"[INFO] Overriding INSTANCE_ID: env={INSTANCE_ID!r} → R2={resolved_id!r}")
        INSTANCE_ID = resolved_id

    # ── Pre-check: are the output files already in R2? ──
    # This handles restart loops — if work is done, self-destruct immediately
    # regardless of local state or status JSON.
    npy_key  = f"{FEATURES_BUCKET_PREFIX}/{CITY_NAME}_{WORKER_INDEX}.{NUM_WORKERS}.npy"
    meta_key = f"{FEATURES_BUCKET_PREFIX}/Metadata_{CITY_NAME}_{WORKER_INDEX}.{NUM_WORKERS}.jsonl"
    print(f"[PRE-CHECK] Checking R2 for existing outputs...")
    print(f"[PRE-CHECK]   npy  key: {npy_key}")
    print(f"[PRE-CHECK]   meta key: {meta_key}")
    if r2.file_exists(npy_key) and r2.file_exists(meta_key):
        print(f"[PRE-CHECK] ✓ Both output files already in R2 — worker {WORKER_INDEX} is DONE.")
        reporter = R2StatusReporter(r2, WORKER_INDEX, 0, status_prefix, INSTANCE_ID)
        reporter.report_final("COMPLETED")
        upload_logs_to_r2()
        self_destruct()
        sys.exit(0)
    else:
        print(f"[PRE-CHECK] Output files not yet in R2 — proceeding with pipeline.")

    resume_strategy, rows_done, done_panoids = check_and_resume(r2, work_dir)
    print(f"[INFO] Resume strategy: {resume_strategy}, rows_done: {rows_done}")

    # ── Step 1: Download CSV segment from R2 ──
    reporter = R2StatusReporter(r2, WORKER_INDEX, 0, status_prefix, INSTANCE_ID)
    reporter.report_final("DOWNLOADING_CSV")

    csv_filename = f"{CITY_NAME}_{WORKER_INDEX}.{NUM_WORKERS}.csv"
    csv_key = f"{CSV_BUCKET_PREFIX}/{csv_filename}"
    local_csv = str(work_dir / csv_filename)

    # Only download CSV if we don't already have it locally
    if not os.path.exists(local_csv):
        print(f"[INFO] Downloading {csv_key} from R2...")
        if not r2.download_file(csv_key, local_csv, max_retries=5):
            print(f"[ERROR] Failed to download CSV: {csv_key}")
            reporter.report_final("FAILED:csv_download")
            sys.exit(1)
    else:
        print(f"[INFO] CSV already exists locally: {local_csv}")

    # ── Step 2: Load CSV ──
    records, metadata_map = load_csv(local_csv)
    total_records = len(records)
    views_per_pano = HARDCODED_CONFIG['num_views']
    total_views_est = total_records * views_per_pano
    feature_dim = 8448

    print(f"[INFO] {total_records} panoids, ~{total_views_est} views expected")
    reporter = R2StatusReporter(r2, WORKER_INDEX, total_views_est, status_prefix, INSTANCE_ID)
    reporter.report_final("INITIALIZING")

    # ── Step 3: Setup output files ──
    features_file = str(work_dir / 'features.npy')
    metadata_file = str(work_dir / 'metadata.jsonl')
    failed_file = str(work_dir / 'failed.jsonl')

    # If FRESH start, done_panoids is empty.
    # If RESUME, done_panoids and rows_done were set by check_and_resume.
    if resume_strategy == "FRESH":
        done_panoids = set()
        rows_done = 0

    # For fresh starts, also check local metadata for backward compat
    if resume_strategy == "FRESH" and os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    done_panoids.add(data['panoid'])
                    rows_done += 1
                except Exception:
                    pass
        if rows_done > 0:
            print(f"[INFO] Fresh start but found local metadata: {rows_done} views already done")

    if resume_strategy == "FRESH" and os.path.exists(failed_file):
        with open(failed_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'panoid' in data:
                        done_panoids.add(data['panoid'])
                except Exception:
                    pass

    to_process = [r for r in records if r['panoid'] not in done_panoids]
    print(f"[INFO] Processing {len(to_process)}/{total_records} panoids (rows_done={rows_done})")

    if not to_process:
        print("[INFO] All panoids already processed, skipping to upload")
    else:
        # Create or open memmap
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

        # Shared state — start writing from rows_done
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
        loop_start = time.time()
        last_progress_time = time.time()
        last_progress_count = rows_done
        last_log_time = time.time()
        batch_times = []
        STALL_TIMEOUT = int(os.environ.get('STALL_TIMEOUT', '600'))  # 10 min default
        LOG_INTERVAL = 30  # Log stats every 30s

        print(f"[INFO] Starting extraction loop (batch_size={extractor.batch_size}, stall_timeout={STALL_TIMEOUT}s)", flush=True)

        # Prefetch pipeline: decode batch N+1 in the thread pool while the GPU
        # runs inference on batch N, eliminating CPU-GPU idle time.
        pending_batch: List[ViewItem] = []
        pending_futures = None

        try:
            while True:
                # Check disk space
                wait_for_disk_space(str(work_dir), MIN_FREE_GB)

                # ── Stall detection ──
                now = time.time()
                since_progress = now - last_progress_time
                if since_progress > STALL_TIMEOUT and stats['ext_ok'] == last_progress_count:
                    msg = (f"[FATAL] Pipeline stalled — no progress for {since_progress:.0f}s "
                           f"(threshold: {STALL_TIMEOUT}s). ext_ok={stats['ext_ok']}, "
                           f"dl_ok={stats['dl_ok']}, dl_fail={stats['dl_fail']}, "
                           f"queue_size={item_queue.qsize()}, dl_alive={dl_thread.is_alive()}")
                    print(msg, flush=True)
                    reporter.report_final(f"FAILED:stall_timeout_{STALL_TIMEOUT}s")
                    upload_logs_to_r2()
                    raise RuntimeError(msg)

                # ── Periodic status log ──
                if now - last_log_time >= LOG_INTERVAL:
                    elapsed = now - loop_start
                    speed = (stats['ext_ok'] - rows_done) / elapsed if elapsed > 0 else 0
                    avg_batch = sum(batch_times[-50:]) / len(batch_times[-50:]) if batch_times else 0
                    remaining = total_views_est - stats['ext_ok']
                    eta = remaining / speed if speed > 0 else 0
                    vram_used = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0
                    panos_done = stats['ext_ok'] // HARDCODED_CONFIG['num_views']
                    pct = int(stats['ext_ok'] / total_views_est * 100) if total_views_est > 0 else 0
                    elapsed_min = elapsed / 60
                    print(f"[STATS] worker={WORKER_INDEX}/{NUM_WORKERS} | "
                          f"views={stats['ext_ok']:,}/{total_views_est:,} ({pct}%) | "
                          f"panos~{panos_done:,}/{total_records:,} | "
                          f"speed={speed:.1f} views/s | eta={eta/60:.1f}min | "
                          f"elapsed={elapsed_min:.1f}min | "
                          f"dl_ok={stats['dl_ok']} | dl_fail={stats['dl_fail']} | "
                          f"queue={item_queue.qsize()} | avg_batch={avg_batch:.2f}s | "
                          f"vram={vram_used:.2f}GB | dl_alive={dl_thread.is_alive()}", flush=True)
                    last_log_time = now

                # ── Fill next batch (extractor.batch_size may shrink after an OOM) ──
                current_batch: List[ViewItem] = []
                while len(current_batch) < extractor.batch_size:
                    try:
                        item = item_queue.get(timeout=0.01)
                        if item is _SENTINEL:
                            continue
                        current_batch.append(item)
                    except queue.Empty:
                        break

                # ── No new items: drain any pending batch then check for exit ──
                if not current_batch:
                    if pending_batch and pending_futures is not None:
                        batch_start = time.time()
                        try:
                            feats_np, meta_batch, _ = extractor.infer_prefetched(pending_batch, pending_futures)
                            if feats_np is not None and len(meta_batch) > 0:
                                shared_state.write_batch(feats_np, meta_batch)
                                stats['ext_ok'] += len(meta_batch)
                                reporter.update(stats['ext_ok'], "EXTRACTING")
                                last_progress_time = time.time()
                                last_progress_count = stats['ext_ok']
                                del feats_np, meta_batch
                            batch_times.append(time.time() - batch_start)
                            if stats['ext_ok'] % 5000 == 0:
                                gc.collect()
                        except Exception as e:
                            print(f"[ERROR] Batch extraction failed: {type(e).__name__}: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                        finally:
                            pending_batch = []
                            pending_futures = None
                    if not dl_thread.is_alive():
                        break
                    continue

                # ── Submit decode of current_batch immediately (non-blocking) ──
                # These JPEG decodes run in the thread pool while the GPU processes
                # the previous (pending) batch — that is the prefetch overlap.
                current_futures = extractor.start_decode(current_batch)

                # ── GPU inference on the previously decoded (pending) batch ──
                if pending_batch and pending_futures is not None:
                    batch_start = time.time()
                    try:
                        feats_np, meta_batch, _ = extractor.infer_prefetched(pending_batch, pending_futures)
                        if feats_np is not None and len(meta_batch) > 0:
                            shared_state.write_batch(feats_np, meta_batch)
                            stats['ext_ok'] += len(meta_batch)
                            reporter.update(stats['ext_ok'], "EXTRACTING")
                            last_progress_time = time.time()
                            last_progress_count = stats['ext_ok']
                            del feats_np, meta_batch
                        batch_times.append(time.time() - batch_start)
                        if stats['ext_ok'] % 5000 == 0:
                            gc.collect()
                    except Exception as e:
                        print(f"[ERROR] Batch extraction failed: {type(e).__name__}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()

                # ── Promote current batch to pending (its decode continues in bg) ──
                pending_batch = current_batch
                pending_futures = current_futures

        except KeyboardInterrupt:
            print("[WARN] Interrupted", flush=True)

        dl_thread.join()
        final_count = shared_state.write_idx
        shared_state.close()

        # Truncate memmap to actual size
        del features_memmap
        gc.collect()

        if final_count == 0 and total_records > 0:
            print(f"[ERROR] 0 features extracted from {total_records} records! Marking as FAILED.")
            reporter.report_final("FAILED:zero_features_extracted")
            # Clean up empty files
            try:
                os.remove(features_file)
                os.remove(metadata_file)
            except OSError:
                pass
            sys.exit(1)

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

    # Upload with progress reporting and indefinite retry for large files
    def upload_with_retry(local_path, bucket_key, label="FILE", max_attempts=5):
        file_size = os.path.getsize(local_path)
        _last_r2_report = [0.0]
        _upload_start = [time.time()]

        def _progress_cb(bytes_done, bytes_total):
            now = time.time()
            elapsed = max(now - _upload_start[0], 0.001)
            mb_done = bytes_done / (1024 * 1024)
            speed_mb = mb_done / elapsed
            remaining_mb = (bytes_total - bytes_done) / (1024 * 1024)
            eta = remaining_mb / speed_mb if speed_mb > 0 else 0
            # Throttle R2 status reports to every 30s
            if now - _last_r2_report[0] >= 30.0:
                reporter.report_upload(bytes_done, bytes_total, speed_mb, eta, label)
                _last_r2_report[0] = now

        for attempt in range(1, max_attempts + 1):
            print(f"[INFO] Uploading {label} ({file_size / (1024**3):.2f} GB), attempt {attempt}/{max_attempts}...")
            _upload_start[0] = time.time()
            _last_r2_report[0] = 0.0
            if r2.upload_file(local_path, bucket_key, max_retries=1,
                              progress_callback=_progress_cb):
                reporter.report_upload(file_size, file_size, 0, 0, label)
                return True
            print(f"[WARN] Upload attempt {attempt}/{max_attempts} failed for {bucket_key}")
            reporter.report_upload(0, file_size, 0, 0, f"{label}_RETRY")
            if attempt < max_attempts:
                wait = min(2 ** attempt, 120)
                print(f"[INFO] Retrying in {wait}s...")
                time.sleep(wait)

        # Indefinite retry every 60s — keep reporting so UI doesn't mark stale
        retry_count = 0
        while True:
            retry_count += 1
            print(f"[WARN] Indefinite retry #{retry_count} for {bucket_key} (every 60s)...")
            reporter.report_upload(0, file_size, 0, 0, f"{label}_RETRY")
            _upload_start[0] = time.time()
            _last_r2_report[0] = 0.0
            if r2.upload_file(local_path, bucket_key, max_retries=1,
                              progress_callback=_progress_cb):
                reporter.report_upload(file_size, file_size, 0, 0, label)
                return True
            time.sleep(60)

    success = True
    if os.path.exists(features_file) and os.path.getsize(features_file) > 0:
        if not upload_with_retry(features_file, npy_key, label="NPY"):
            success = False
    if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
        if not upload_with_retry(metadata_file, meta_key, label="META"):
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
        upload_logs_to_r2()
        self_destruct()
    else:
        reporter.report_final("FAILED:upload")
        print("[ERROR] Upload failed. Instance kept alive for debugging.")
        upload_logs_to_r2()
        sys.exit(1)


if __name__ == '__main__':
    _log_fh = _start_log_capture()
    try:
        main()
    except Exception as e:
        error_type = type(e).__name__
        print(f"[CRITICAL] {error_type}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Truncate error message for status key (R2 keys have limits)
        short_err = f"{error_type}:{str(e)[:100]}"
        upload_logs_to_r2()
        # Report failure to R2
        try:
            r2 = R2Client()
            reporter = R2StatusReporter(r2, WORKER_INDEX, 0, FEATURES_BUCKET_PREFIX, INSTANCE_ID)
            reporter.report_final(f"FAILED:{short_err}")
        except Exception as r2e:
            print(f"[CRITICAL] Could not report failure to R2: {r2e}", flush=True)
        sys.exit(1)
