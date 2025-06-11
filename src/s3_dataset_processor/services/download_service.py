"""Service for file download operations."""

import os
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse

from config.settings import settings
from models.schemas import DownloadedFile


class DownloadService:
    """Service class for handling file download operations."""
    
    def __init__(self):
        """Initialize the download service."""
        self.data_dir = settings.DATA_DIR
        self.timeout = settings.DOWNLOAD_TIMEOUT
        self.chunk_size = settings.CHUNK_SIZE
    
    def download_file(self, url: str, local_path: str) -> bool:
        """
        Download a file from URL to local path.
        
        Args:
            url: The presigned URL to download from
            local_path: Local file path to save the file
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
            
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False
    
    def process_downloads(self, decoded_data: List[Dict[str, Any]]) -> tuple:
        """
        Process multiple downloads from decoded data.
        
        Args:
            decoded_data: List of decoded URL data
            
        Returns:
            Tuple of (downloaded_files, successful_downloads, failed_downloads)
        """
        downloaded_files = []
        successful_downloads = 0
        failed_downloads = 0
        
        for entry in decoded_data:
            agency_id = entry.get('agencyId', 'unknown')
            signed_url = entry.get('signedS3Url', '')
            
            if not signed_url:
                failed_downloads += 1
                continue
                
            # Parse URL to get filename
            parsed_url = urlparse(signed_url)
            original_filename = parsed_url.path.split('/')[-1] if parsed_url.path else f"{agency_id}.zip"
            
            # Download file to data directory
            local_file_path = os.path.join(self.data_dir, original_filename)
            print(f"Downloading {original_filename} for agency {agency_id}")
            
            if self.download_file(signed_url, local_file_path):
                file_size = os.path.getsize(local_file_path)
                
                downloaded_file = DownloadedFile(
                    agency_id=agency_id,
                    original_filename=original_filename,
                    local_path=local_file_path,
                    file_size=file_size
                )
                
                downloaded_files.append(downloaded_file)
                successful_downloads += 1
                print(f"Successfully downloaded {original_filename}")
                
            else:
                failed_downloads += 1
                print(f"Failed to download {original_filename}")
        
        return downloaded_files, successful_downloads, failed_downloads