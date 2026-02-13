"""
Utility module for Google Street View panorama processing.

This module provides helper functions and classes for:

- Timing code execution (`timer` context manager).
- Analyzing panorama tiles for black pixels (`has_black_bottom`, `black_percentage`).
- Loading datasets (`open_dataset`).
- Parsing command-line arguments for the panorama downloader (`parse_args`).
- Saving panorama images and formatting file sizes (`save_img`, `format_size`).

Dependencies:
- numpy for image pixel analysis
- PIL/Pillow for image handling
- rich for colored terminal output
- argparse for CLI argument parsing
- json and os for dataset management and file handling
"""
import numpy as np
import time
import json
import csv
from rich import print
import argparse
import os
from PIL import Image

class timer:
    """
    Context manager to measure and print elapsed execution time.

    Usage:
        with timer():
            # your code here
    -----
    >>> with timer() as t:
    ...     # some code to measure
    ...     time.sleep(2)
    >>> print(t.time_elapsed)
    '0h 0m 2.00s'
    """

    def __enter__(self):
        self.start = time.time()
        self.time_elapsed = None
        return self


    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        hrs, rem = divmod(self.interval, 3600)
        mins, secs = divmod(rem, 60)
        self.time_elapsed = f"{int(hrs)}h {int(mins)}m {secs:.2f}s"
        return False 

def has_black_bottom(tile, black_threshold: int = 10, check_rows: int = 5) -> bool:
    """
    Check if the bottom rows of a tile image are completely black.

    Args:
        tile (PIL.Image.Image): The image tile to check.
        black_threshold (int, optional): Max RGB value considered 'black'. Defaults to 10.
        check_rows (int, optional): Number of bottom rows to inspect. Defaults to 5.

    Returns:
        bool: True if all bottom rows are black, False otherwise.
    """
    arr = np.array(tile)
    bottom = arr[-check_rows:]
    return np.all(bottom <= black_threshold)


def black_percentage(tile, threshold: int = 10) -> float:
    """
    Calculate the percentage of black pixels in a tile.

    Args:
        tile (PIL.Image.Image): Tile image to analyze.
        threshold (int, optional): Max RGB value to consider a pixel 'black'. Defaults to 10.

    Returns:
        float: Percentage of black pixels (0–100).
    """
    img_np = np.array(tile)

    # Pixel is black if all channels <= threshold
    black_pixels = np.all(img_np <= threshold, axis=2)

    # Compute percentage
    percent_black = np.sum(black_pixels) / black_pixels.size * 100
    return percent_black


def open_dataset(dataset_location: str) -> list[dict]:
    """
    Load dataset from JSON or CSV file.
    
    For JSON: Expects an array of panorama IDs (strings).
    For CSV: Auto-detects column 'panoid' and optional 'heading_deg'.

    Args:
        dataset_location (str): Path to dataset file (.json or .csv).

    Returns:
        list[dict]: List of dicts, e.g. [{'panoid': '...', 'heading_deg': 123.4}, ...]
    """
    file_ext = os.path.splitext(dataset_location)[1].lower()
    dataset_records = []
    
    if file_ext == '.csv':
        # utf-8-sig handles standard UTF-8 and Excel's BOM signatures
        with open(dataset_location, 'r', encoding='utf-8-sig') as csvfile:
            # 1. Sniff the delimiter (Comma or Semicolon)
            sample = csvfile.read(4096)
            csvfile.seek(0)
            
            delimiter = ',' # Default
            if ';' in sample and sample.count(';') > sample.count(','):
                delimiter = ';'
                
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            # 2. Find the correct column names (Case-Insensitive search)
            panoid_col = None
            heading_col = None
            
            country_code_col = None
            address_label_col = None

            if reader.fieldnames:
                for field in reader.fieldnames:
                    clean_field = field.lower().strip().replace('_', '').replace('-', '')
                    if clean_field == 'panoid':
                        panoid_col = field
                    elif clean_field in ['headingdeg', 'heading', 'yaw']:
                        heading_col = field
                    elif clean_field == 'countrycode':
                        country_code_col = field
                    elif clean_field == 'addresslabel':
                        address_label_col = field
            
            if not panoid_col:
                print(f"[red]Error: Could not find a 'panoid' column in CSV. Found columns: {reader.fieldnames}[/]")
                return []

            # 3. Extract IDs and optional metadata
            for row in reader:
                if panoid_col in row and row[panoid_col]:
                    val = row[panoid_col].strip()
                    if val:
                        record = {'panoid': val}

                        # Try to parse heading if available
                        if heading_col and heading_col in row and row[heading_col]:
                            try:
                                record['heading_deg'] = float(row[heading_col])
                            except ValueError:
                                pass # Ignore invalid headings

                        # Extract location metadata
                        if country_code_col and country_code_col in row and row[country_code_col]:
                            record['country_code'] = row[country_code_col].strip()
                        if address_label_col and address_label_col in row and row[address_label_col]:
                            record['address_label'] = row[address_label_col].strip()

                        dataset_records.append(record)
        
        return dataset_records
    else:
        # Load JSON file (default)
        with open(dataset_location, 'r', encoding='utf-8') as dataset:
            data = json.load(dataset)
            # Normalize JSON list of strings to list of dicts
            if isinstance(data, list):
                if data and isinstance(data[0], str):
                    return [{'panoid': pid} for pid in data]
                return data # Assume already dicts
            return []


def parse_args():
    """
    Parse command-line arguments for the panorama downloader.

    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="Google Street View Panorama Downloader with Directional View Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Basic options
    parser.add_argument("--zoom", type=int, default=2, help="Zoom level (0-5)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file (JSON or CSV)")
    parser.add_argument("--max-threads", type=int, default=20, help="Max concurrent downloads (default 20)")
    parser.add_argument("--workers", type=int, default=4, help="Max process pool workers (default 4)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of panoramas to process")
    parser.add_argument("--output", type=str, default=r"D:\GeoAxis\Hypervision\Output\Images", help="Output directory")
    
    # Directional views options
    parser.add_argument("--directional-views", action="store_true", help="Enable directional view extraction")
    parser.add_argument("--keep-pano", action="store_true", help="Save full panorama to disk")
    parser.add_argument("--view-resolution", type=int, default=512, help="Output resolution for views")
    parser.add_argument("--view-fov", type=float, default=90.0, help="Field of view in degrees")
    parser.add_argument("--num-views", type=int, default=6, help="Number of directional views")
    parser.add_argument("--view-offset", type=float, default=0.0, help="Yaw offset for directional views in degrees")
    parser.add_argument("--aa-strength", type=float, default=0.8, help="Antialiasing strength (0.0=Off, 1.0=Full)")
    parser.add_argument("--interpolation", type=str, default="lanczos", choices=["cubic", "lanczos"], help="Interpolation method")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG encode quality 1-100 (lower=faster+smaller, default 95)")
    parser.add_argument("--no-antialias", action="store_true", help="Skip mip/blur preprocessing for speed (may alias on thin features)")
    
    # New Augmentation Args
    parser.add_argument("--global", dest="global_view", action="store_true", help="Extract only ONE random view per panorama")
    parser.add_argument("--augment", action="store_true", help="Apply random geometric and pixel augmentations")
    
    # Google Cloud Storage options
    parser.add_argument("--gcs-bucket", type=str, default="", help="GCS bucket name (enables GCS upload)")
    parser.add_argument("--gcs-path", type=str, default="", help="Base path in bucket")
    parser.add_argument("--gcs-credentials", type=str, default="", help="Path to service account JSON")
    parser.add_argument("--gcs-also-local", action="store_true", help="Save to both GCS and local storage")

    # Profiling


    return parser.parse_args()


def format_size(num_bytes: int) -> str:
    """
    Convert a file size in bytes into a human-readable string.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"

def save_img(full_img: Image.Image, output_dir: str, panoid: str, zoom_level: int) -> str:
    """
    Save a PIL image to disk in a structured directory layout.
    """
    zoom_output_folder = os.path.join(output_dir, f"panos_z{zoom_level}")
    os.makedirs(zoom_output_folder, exist_ok=True) 
    out_path = os.path.join(zoom_output_folder, f"{panoid}.jpg")

    full_img.save(out_path)
    file_size_bytes = os.path.getsize(out_path)
    file_size_fmt = format_size(file_size_bytes)

    return file_size_fmt