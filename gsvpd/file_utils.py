"""
File utility functions for panorama processing.

Provides functions to scan directories for existing panoramas
and extract panorama IDs from filenames.
"""
import os
from typing import Set


def extract_panoid_from_filename(filename: str) -> str:
    """
    Extract panorama ID from a filename.

    Handles all view filename conventions:
        Standard:   "{panoid}_zoom2_view00_0deg.jpg"
        Global:     "{panoid}_rnd_Y180.jpg"
        Augmented:  "{panoid}_aug_Y180_P5.jpg"
        Panorama:   "{panoid}.jpg"

    Args:
        filename: Filename to parse

    Returns:
        Extracted panorama ID, or empty string if not found
    """
    name = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Standard: panoid_zoom2_view00_0deg
    if "_zoom" in name:
        return name.split("_zoom")[0]

    # Global random view: panoid_rnd_Y180
    if "_rnd_Y" in name:
        return name.split("_rnd_Y")[0]

    # Augmented: panoid_aug_Y180_P5
    if "_aug_Y" in name:
        return name.split("_aug_Y")[0]

    # Panorama file: panoid (no suffix)
    return name


def find_existing_panoids(directory: str) -> Set[str]:
    """
    Scan directory recursively for existing panoramas.
    
    Iterates through all files, extracts panoids from filenames,
    and returns a unique set of panorama IDs.
    
    Args:
        directory: Path to output directory to scan
        
    Returns:
        Set of unique panorama ID strings found in the directory
    """
    panoids = set()
    
    if not os.path.exists(directory):
        return panoids
    
    # Recursively scan all subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                panoid = extract_panoid_from_filename(filename)
                if panoid:
                    panoids.add(panoid)
    
    return panoids
