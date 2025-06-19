"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Dict, Optional


class DownloadRequest(BaseModel):
    """Request model for downloading files from signed URLs."""
    encoded_data: str
    extract_files: Optional[bool] = True


class DownloadedFile(BaseModel):
    """Model for downloaded file information."""
    agency_id: str
    original_filename: str
    local_path: str
    file_size: int
    extracted_path: Optional[str] = None
    extraction_success: bool = False


class DownloadResponse(BaseModel):
    """Response model for download operation."""
    success: bool
    message: str
    total_downloads: int
    successful_downloads: int
    failed_downloads: int
    downloaded_files: List[DownloadedFile]
    extracted_folders: List[Dict[str, str]]
    total_extracted_folders: int