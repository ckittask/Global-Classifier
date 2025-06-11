"""Handler for API response formatting."""

from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs

from models.schemas import DownloadResponse, DownloadedFile


class ResponseHandler:
    """Handler class for formatting API responses."""
    
    # @staticmethod
    # def format_decoded_data(data: List[Dict[str, Any]]) -> DecodeResponse:
    #     """
    #     Format decoded data for API response.
        
    #     Args:
    #         data: List of decoded data dictionaries
            
    #     Returns:
    #         Formatted decode response
    #     """
    #     if not data:
    #         return DecodeResponse(
    #             success=False,
    #             message="No data found or failed to decode",
    #             total_entries=0,
    #             entries=[],
    #             extracted_urls=[],
    #             total_urls=0
    #         )
        
    #     formatted_entries = []
    #     extracted_urls = []
        
    #     for i, entry in enumerate(data, 1):
    #         agency_id = entry.get('agencyid', 'N/A')
    #         signed_url = entry.get('signedS3url', '')
            
    #         formatted_entry = AgencyData(
    #             entry_number=i,
    #             agency_id=agency_id,
    #             signed_url=signed_url
    #         )
            
    #         if signed_url:
    #             # Parse URL to show components
    #             parsed_url = urlparse(signed_url)
    #             query_params = parse_qs(parsed_url.query)
                
    #             # Extract filename from path
    #             filename = parsed_url.path.split('/')[-1] if parsed_url.path else 'unknown'
                
    #             formatted_entry.host = parsed_url.netloc
    #             formatted_entry.path = parsed_url.path
    #             formatted_entry.filename = filename
    #             formatted_entry.expires_in_seconds = query_params.get('X-Amz-Expires', ['unknown'])[0]
                
    #             extracted_urls.append(signed_url)
            
    #         formatted_entries.append(formatted_entry)
        
    #     return DecodeResponse(
    #         success=True,
    #         message="Successfully decoded signed URLs",
    #         total_entries=len(data),
    #         entries=formatted_entries,
    #         extracted_urls=extracted_urls,
    #         total_urls=len(extracted_urls)
    #     )
    
    @staticmethod
    def format_download_response(
        decoded_data: List[Dict[str, Any]],
        downloaded_files: List[DownloadedFile],
        successful_downloads: int,
        failed_downloads: int,
        extracted_folders: List[Dict[str, str]]
    ) -> DownloadResponse:
        """
        Format download response.
        
        Args:
            decoded_data: Original decoded data
            downloaded_files: List of downloaded files
            successful_downloads: Number of successful downloads
            failed_downloads: Number of failed downloads
            extracted_folders: List of extracted folder information
            
        Returns:
            Formatted download response
        """
        return DownloadResponse(
            success=successful_downloads > 0,
            message=f"Downloaded {successful_downloads} files, {failed_downloads} failed",
            total_downloads=len(decoded_data),
            successful_downloads=successful_downloads,
            failed_downloads=failed_downloads,
            downloaded_files=downloaded_files,
            extracted_folders=extracted_folders,
            total_extracted_folders=len(extracted_folders)
        )