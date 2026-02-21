"""
Intelligent ZIP batching system for images.

This module handles batching images into 5GB zip files with intelligent
memory management:
1. Accumulates images in memory until reaching 5GB threshold
2. Immediately zips and saves to disk
3. Clears batch from RAM
4. Starts uploading zip in background while next batch processes
"""
import os
import zipfile
import asyncio
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from rich import print


@dataclass
class ImageData:
    """Holds image data for batching."""
    filename: str
    data: bytes
    size: int


@dataclass
class ZipBatch:
    """Represents a batch of images to be zipped."""
    batch_id: int
    images: List[ImageData] = field(default_factory=list)
    total_size: int = 0
    zip_path: str = ""
    
    def add_image(self, filename: str, data: bytes):
        """Add an image to the batch."""
        size = len(data)
        self.images.append(ImageData(filename, data, size))
        self.total_size += size
    
    def should_zip(self, threshold_bytes: int = 5 * 1024 * 1024 * 1024) -> bool:
        """Check if batch has reached the size threshold."""
        return self.total_size >= threshold_bytes
    
    def clear(self):
        """Clear images from memory."""
        self.images.clear()
        self.total_size = 0


class ZipBatcher:
    """
    Intelligent ZIP batcher that manages 5GB batches.
    
    Features:
    - Accumulates images until 5GB threshold
    - Zips and saves to disk immediately when threshold reached
    - Clears RAM after zipping
    - Uploads zip files in background
    - Thread-safe operations
    """
    
    def __init__(
        self, 
        output_dir: str,
        gcs_uploader,
        gcs_config,
        threshold_gb: float = 5.0,
        max_upload_workers: int = 2
    ):
        self.output_dir = output_dir
        self.gcs_uploader = gcs_uploader
        self.gcs_config = gcs_config
        self.threshold_bytes = int(threshold_gb * 1024 * 1024 * 1024)
        
        # Create zips directory
        self.zips_dir = os.path.join(output_dir, "zip_batches")
        os.makedirs(self.zips_dir, exist_ok=True)
        
        # Current batch
        self.current_batch = ZipBatch(batch_id=1)
        self.batch_lock = threading.Lock()
        
        # Upload tracking
        self.upload_executor = ThreadPoolExecutor(max_workers=max_upload_workers)
        self.pending_uploads = []
        self.completed_uploads = []
        self.failed_uploads = []
        
        # Statistics
        self.total_images_batched = 0
        self.total_zips_created = 0
        self.total_bytes_processed = 0
        
        print(f"[cyan]ZIP Batcher initialized:[/]")
        print(f"  Output directory: {self.zips_dir}")
        print(f"  Threshold: {threshold_gb}GB per zip")
        print(f"  Max parallel uploads: {max_upload_workers}\n")
    
    async def add_image(self, filename: str, data: bytes, image_type: str = "pano"):
        """
        Add an image to the current batch.
        
        Args:
            filename: Name of the image file
            data: Image bytes
            image_type: Type of image (pano/view) for organizing in zip
        """
        with self.batch_lock:
            # Add to current batch
            zip_filename = f"{image_type}/{filename}"
            self.current_batch.add_image(zip_filename, data)
            self.total_images_batched += 1
            self.total_bytes_processed += len(data)
            
            # Check if we should zip and upload
            if self.current_batch.should_zip(self.threshold_bytes):
                await self._finalize_and_upload_batch()
    
    async def _finalize_and_upload_batch(self):
        """Zip the current batch, save to disk, clear RAM, and start upload."""
        if not self.current_batch.images:
            return
        
        batch_to_zip = self.current_batch
        
        # Create new batch for next images (so processing can continue)
        self.current_batch = ZipBatch(batch_id=batch_to_zip.batch_id + 1)
        
        # Zip in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        zip_path = await loop.run_in_executor(
            None,
            self._create_zip_file,
            batch_to_zip
        )
        
        if zip_path:
            batch_to_zip.zip_path = zip_path
            self.total_zips_created += 1
            
            # Clear images from RAM
            batch_to_zip.clear()
            
            # Start upload in background
            if self.gcs_uploader and self.gcs_uploader.is_configured():
                upload_future = self.upload_executor.submit(
                    self._upload_zip_file,
                    zip_path,
                    batch_to_zip.batch_id
                )
                self.pending_uploads.append((batch_to_zip.batch_id, upload_future))
                
                print(f"[green]✓ Batch {batch_to_zip.batch_id} zipped ({batch_to_zip.total_size / (1024**3):.2f}GB) "
                      f"and queued for upload[/]")
            else:
                print(f"[green]✓ Batch {batch_to_zip.batch_id} zipped ({batch_to_zip.total_size / (1024**3):.2f}GB), "
                      f"saved locally (no GCS configured)[/]")
    
    def _create_zip_file(self, batch: ZipBatch) -> Optional[str]:
        """Create zip file from batch (runs in thread pool)."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"batch_{batch.batch_id:04d}_{timestamp}.zip"
            zip_path = os.path.join(self.zips_dir, zip_filename)
            
            # Create zip file
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for img in batch.images:
                    zf.writestr(img.filename, img.data)
            
            return zip_path
            
        except Exception as e:
            print(f"[red]✗ Failed to create zip for batch {batch.batch_id}: {e}[/]")
            return None
    
    def _upload_zip_file(self, zip_path: str, batch_id: int) -> bool:
        """Upload zip file to GCS (runs in thread pool)."""
        try:
            zip_filename = os.path.basename(zip_path)
            gcs_path = f"zip_batches/{zip_filename}"
            
            result = self.gcs_uploader.upload_file(zip_path, gcs_path)
            
            if result.success:
                self.completed_uploads.append(batch_id)
                size_gb = result.bytes_uploaded / (1024**3)
                print(f"[green]✓ Batch {batch_id} uploaded to GCS ({size_gb:.2f}GB) - {result.gcs_uri}[/]")
                return True
            else:
                self.failed_uploads.append(batch_id)
                print(f"[red]✗ Failed to upload batch {batch_id}: {result.error}[/]")
                return False
                
        except Exception as e:
            self.failed_uploads.append(batch_id)
            print(f"[red]✗ Exception uploading batch {batch_id}: {e}[/]")
            return False
    
    async def finalize(self):
        """Finalize any remaining batch and wait for all uploads to complete."""
        with self.batch_lock:
            # Zip any remaining images
            if self.current_batch.images:
                print(f"\n[cyan]Finalizing remaining batch ({len(self.current_batch.images)} images, "
                      f"{self.current_batch.total_size / (1024**3):.2f}GB)...[/]")
                await self._finalize_and_upload_batch()
        
        # Wait for all pending uploads
        if self.pending_uploads:
            print(f"\n[cyan]Waiting for {len(self.pending_uploads)} upload(s) to complete...[/]")
            
            for batch_id, future in self.pending_uploads:
                try:
                    future.result(timeout=3600)  # 1 hour timeout per upload
                except Exception as e:
                    print(f"[red]✗ Upload of batch {batch_id} failed: {e}[/]")
        
        # Shutdown upload executor
        self.upload_executor.shutdown(wait=True)
        
        # Print summary
        print(f"\n[cyan]{'=' * 60}[/]")
        print(f"[green]ZIP Batching Summary:[/]")
        print(f"  Total images batched: {self.total_images_batched}")
        print(f"  Total data processed: {self.total_bytes_processed / (1024**3):.2f}GB")
        print(f"  Total zip files created: {self.total_zips_created}")
        print(f"  Successful uploads: {len(self.completed_uploads)}")
        if self.failed_uploads:
            print(f"  [red]Failed uploads: {len(self.failed_uploads)}[/]")
        print(f"[cyan]{'=' * 60}[/]\n")
