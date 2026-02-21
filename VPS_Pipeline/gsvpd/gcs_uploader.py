"""
Google Cloud Storage uploader for panoramas and directional views.

Provides functionality to upload images directly to GCS buckets,
with support for service account credentials and batch uploads.
"""
import os
from typing import Optional, List
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class GCSConfig:
    """Configuration for Google Cloud Storage uploads."""
    bucket_name: str = ""
    base_path: str = ""
    enabled: bool = False
    also_save_local: bool = False
    credentials_file: str = ""
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return self.enabled and bool(self.bucket_name)
    
    def get_full_path(self, relative_path: str) -> str:
        """Get full path for a file in the bucket."""
        if not self.base_path:
            return relative_path
        
        # Ensure no double slashes
        result = self.base_path.rstrip('/')
        if relative_path and not relative_path.startswith('/'):
            result += '/'
        result += relative_path
        return result


@dataclass
class GCSUploadResult:
    """Result of a GCS upload operation."""
    success: bool = False
    gcs_uri: str = ""
    public_url: str = ""
    error: str = ""
    bytes_uploaded: int = 0


class GCSUploader:
    """Upload images to Google Cloud Storage."""
    
    def __init__(self):
        self.config: Optional[GCSConfig] = None
        self.client = None
        self.bucket = None
        self.initialized = False
    
    def initialize(self, config: GCSConfig) -> bool:
        """
        Initialize GCS client with configuration.
        
        Args:
            config: GCS configuration
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            from google.cloud import storage
            
            self.config = config
            
            # Create client with credentials
            if config.credentials_file:
                self.client = storage.Client.from_service_account_json(
                    config.credentials_file
                )
            else:
                # Use default credentials (GOOGLE_APPLICATION_CREDENTIALS env var
                # or gcloud CLI credentials, or GCE/GKE metadata server)
                self.client = storage.Client()
            
            # Get bucket
            self.bucket = self.client.bucket(config.bucket_name)
            
            # Test connection
            if not self.bucket.exists():
                print(f"[GCS] Bucket '{config.bucket_name}' does not exist or is not accessible")
                return False
            
            print(f"[GCS] Successfully connected to bucket: {config.bucket_name}")
            if config.base_path:
                print(f"[GCS] Base path: {config.base_path}")
            
            self.initialized = True
            return True
            
        except ImportError:
            print("[GCS] Error: google-cloud-storage package not installed")
            print("[GCS] Install with: pip install google-cloud-storage")
            return False
        except Exception as e:
            print(f"[GCS] Initialization failed: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if GCS is properly configured."""
        return self.initialized and self.client is not None
    
    def upload_image(
        self,
        image: np.ndarray,
        destination_path: str,
        jpeg_quality: int = 95
    ) -> GCSUploadResult:
        """
        Upload an OpenCV image to GCS.
        
        Args:
            image: OpenCV image (numpy array)
            destination_path: Destination path in bucket
            jpeg_quality: JPEG compression quality (0-100)
            
        Returns:
            GCSUploadResult with upload status
        """
        result = GCSUploadResult()
        
        if not self.is_configured():
            result.error = "GCS not configured"
            return result
        
        try:
            # Encode image as JPEG
            success, buffer = cv2.imencode(
                '.jpg',
                image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            
            if not success:
                result.error = "Failed to encode image as JPEG"
                return result
            
            data = buffer.tobytes()
            
            # Upload
            return self.upload_data(data, destination_path, "image/jpeg")
            
        except Exception as e:
            result.error = f"Exception: {e}"
            return result
    
    def upload_data(
        self,
        data: bytes,
        destination_path: str,
        content_type: str = "image/jpeg"
    ) -> GCSUploadResult:
        """
        Upload raw data to GCS.
        
        Args:
            data: Raw bytes to upload
            destination_path: Destination path in bucket
            content_type: MIME type of content
            
        Returns:
            GCSUploadResult with upload status
        """
        result = GCSUploadResult()
        
        if not self.is_configured():
            result.error = "GCS not configured"
            return result
        
        try:
            full_path = self.config.get_full_path(destination_path)
            
            # Create blob and upload
            blob = self.bucket.blob(full_path)
            blob.upload_from_string(data, content_type=content_type)
            
            # Success
            result.success = True
            result.bytes_uploaded = len(data)
            result.gcs_uri = f"gs://{self.config.bucket_name}/{full_path}"
            result.public_url = f"https://storage.googleapis.com/{self.config.bucket_name}/{full_path}"
            
            return result
            
        except Exception as e:
            result.error = f"Exception: {e}"
            return result
    
    def upload_file(
        self,
        local_path: str,
        destination_path: str
    ) -> GCSUploadResult:
        """
        Upload a file from local disk to GCS.
        
        Args:
            local_path: Path to local file
            destination_path: Destination path in bucket
            
        Returns:
            GCSUploadResult with upload status
        """
        result = GCSUploadResult()
        
        if not self.is_configured():
            result.error = "GCS not configured"
            return result
        
        try:
            full_path = self.config.get_full_path(destination_path)
            
            # Determine content type
            content_type = "application/octet-stream"
            if local_path.endswith(('.jpg', '.jpeg')):
                content_type = "image/jpeg"
            elif local_path.endswith('.png'):
                content_type = "image/png"
            
            # Create blob and upload
            blob = self.bucket.blob(full_path)
            blob.upload_from_filename(local_path, content_type=content_type)
            
            # Success
            file_size = os.path.getsize(local_path)
            result.success = True
            result.bytes_uploaded = file_size
            result.gcs_uri = f"gs://{self.config.bucket_name}/{full_path}"
            result.public_url = f"https://storage.googleapis.com/{self.config.bucket_name}/{full_path}"
            
            return result
            
        except Exception as e:
            result.error = f"Exception: {e}"
            return result
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """
        List objects in bucket with given prefix.
        
        Args:
            prefix: Prefix to filter objects
            
        Returns:
            List of object names
        """
        if not self.is_configured():
            return []
        
        try:
            full_prefix = self.config.get_full_path(prefix)
            blobs = self.client.list_blobs(self.config.bucket_name, prefix=full_prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            print(f"[GCS] List objects failed: {e}")
            return []
    
    def object_exists(self, object_path: str) -> bool:
        """
        Check if an object exists in the bucket.
        
        Args:
            object_path: Path to object
            
        Returns:
            True if object exists, False otherwise
        """
        if not self.is_configured():
            return False
        
        try:
            full_path = self.config.get_full_path(object_path)
            blob = self.bucket.blob(full_path)
            return blob.exists()
        except Exception:
            return False
    
    def delete_object(self, object_path: str) -> bool:
        """
        Delete an object from the bucket.
        
        Args:
            object_path: Path to object
            
        Returns:
            True if deletion successful, False otherwise
        """
        if not self.is_configured():
            return False
        
        try:
            full_path = self.config.get_full_path(object_path)
            blob = self.bucket.blob(full_path)
            blob.delete()
            return True
        except Exception as e:
            print(f"[GCS] Delete failed: {e}")
            return False
