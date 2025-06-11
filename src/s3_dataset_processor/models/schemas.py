"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


# class DecodeRequest(BaseModel):
#     """Request model for decoding signed URLs."""
#     encoded_data: str


class DownloadRequest(BaseModel):
    """Request model for downloading files from signed URLs."""
    encoded_data: str
    extract_files: Optional[bool] = True


# class AgencyData(BaseModel):
#     """Model for individual agency data."""
#     entry_number: int
#     agency_id: str
#     signed_url: str
#     host: Optional[str] = None
#     path: Optional[str] = None
#     filename: Optional[str] = None
#     expires_in_seconds: Optional[str] = None


class DownloadedFile(BaseModel):
    """Model for downloaded file information."""
    agency_id: str
    original_filename: str
    local_path: str
    file_size: int
    extracted_path: Optional[str] = None
    extraction_success: bool = False


# class DecodeResponse(BaseModel):
#     """Response model for decoded URLs."""
#     success: bool
#     message: str
#     total_entries: int
#     entries: List[AgencyData]
#     extracted_urls: List[str]
#     total_urls: int


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