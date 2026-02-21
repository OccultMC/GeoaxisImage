"""
Progress bar with colored console output for panorama processing.

Displays real-time progress with success/failure counts, processing speed,
and estimated time to completion.
"""
import sys
import time
from typing import Optional
from rich import print as rprint
from rich.console import Console


class ProgressBar:
    """Progress bar with colored output and statistics."""
    
    def __init__(self, total_count: int):
        """
        Initialize progress bar.
        
        Args:
            total_count: Total number of items to process
        """
        self.total = total_count
        self.success_count = 0
        self.fail_count = 0
        self.start_time = time.time()
        self.console = Console()
        
        # Print initial message
        print(f"\nProcessing {total_count} panoramas...\n")
    
    def update(self, success_count: int, fail_count: int):
        """
        Update progress bar with current counts.
        
        Args:
            success_count: Number of successful operations
            fail_count: Number of failed operations
        """
        self.success_count = success_count
        self.fail_count = fail_count
        self._draw()
    
    def log_success(self, result: dict, config: dict):
        """
        Log a successful operation.
        
        Args:
            result: Result dictionary with panoid, zoom, views_created, etc.
            config: Configuration dictionary
        """
        # Clear the progress bar line and move to start of line
        print('\r' + ' ' * 120, end='\r')
        
        msg_parts = [f"✓ PanoID: {result['pano_id']}"]
        
        if config.get('create_directional_views'):
            msg_parts.append(f"Views: {result.get('views_created', 0)}")
        
        gcs_uploaded = result.get('uploaded_to_gcs', False)
        if gcs_uploaded:
            msg_parts.append("Uploaded to GCS")
        
        print(" | ".join(msg_parts))
        self._draw()
    
    def log_failure(self, result: dict):
        """
        Log a failed operation.
        
        Args:
            result: Result dictionary with panoid and error
        """
        # Clear the progress bar line and move to start
        print('\r' + ' ' * 120, end='\r')
        
        print(f"✗ PanoID: {result['pano_id']} | Error: {result.get('error', 'Unknown error')}")
        self._draw()
    
    def finish(self):
        """Finish progress bar and move to new line."""
        print()
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time duration as HH:MM:SS.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds > 360000 or seconds < 0:  # Cap at 100 hours
            return "--:--:--"
        
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def _draw(self):
        """Draw the progress bar."""
        processed = self.success_count + self.fail_count
        if self.total == 0:
            return
        
        elapsed_seconds = time.time() - self.start_time
        panos_per_sec = processed / elapsed_seconds if elapsed_seconds > 0.1 else 0.0
        eta_seconds = (self.total - processed) / panos_per_sec if panos_per_sec > 0 else float('inf')
        
        # Calculate bar widths
        bar_width = 40
        success_pos = int((self.success_count / self.total) * bar_width)
        fail_pos = int((self.fail_count / self.total) * bar_width)
        
        # Ensure we don't exceed bar_width
        if success_pos + fail_pos > bar_width:
            success_pos = bar_width - fail_pos
        
        # Build simple progress bar
        percentage = int(processed * 100 / self.total)
        bar_filled = "█" * success_pos
        bar_failed = "■" * fail_pos
        bar_empty = "░" * (bar_width - success_pos - fail_pos)
        
        progress_str = f"\rProgress: {percentage}% [{bar_filled}{bar_failed}{bar_empty}] {processed}/{self.total} | {panos_per_sec:.2f} p/s | ETA: {self._format_time(eta_seconds)}"
        
        # Print without newline
        print(progress_str, end='', flush=True)
