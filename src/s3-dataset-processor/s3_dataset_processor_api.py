"""FastAPI application for S3 dataset processor."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import urllib.parse
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs
import uvicorn

app = FastAPI(
    title="S3 Dataset Processor API",
    description="API for decoding and processing S3 presigned URLs",
    version="1.0.0"
)

class DecodeRequest(BaseModel):
    """Request model for decoding signed URLs."""
    encoded_data: str

class AgencyData(BaseModel):
    """Model for individual agency data."""
    entry_number: int
    agency_id: str
    signed_url: str
    host: Optional[str] = None
    path: Optional[str] = None
    filename: Optional[str] = None
    expires_in_seconds: Optional[str] = None

class DecodeResponse(BaseModel):
    """Response model for decoded URLs."""
    success: bool
    message: str
    total_entries: int
    entries: List[AgencyData]
    extracted_urls: List[str]
    total_urls: int

def decode_signed_urls(encoded_data: str) -> List[Dict[str, Any]]:
    """
    Decode URL-encoded signed URLs data.
    
    Args:
        encoded_data: URL-encoded JSON string containing signed URLs
        
    Returns:
        List of decoded URL data dictionaries
    """
    try:
        # URL decode the data
        decoded_data = urllib.parse.unquote(encoded_data)
        
        # Parse JSON
        parsed_data = json.loads(decoded_data)
        
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
    except Exception as e:
        raise ValueError(f"Failed to decode data: {e}")

def format_decoded_data(data: List[Dict[str, Any]]) -> DecodeResponse:
    """
    Format decoded data for API response.
    
    Args:
        data: List of decoded data dictionaries
        
    Returns:
        Formatted response
    """
    if not data:
        return DecodeResponse(
            success=False,
            message="No data found or failed to decode",
            total_entries=0,
            entries=[],
            extracted_urls=[],
            total_urls=0
        )
    
    formatted_entries = []
    extracted_urls = []
    
    for i, entry in enumerate(data, 1):
        agency_id = entry.get('agencyid', 'N/A')
        signed_url = entry.get('signedS3url', '')
        
        formatted_entry = AgencyData(
            entry_number=i,
            agency_id=agency_id,
            signed_url=signed_url
        )
        
        if signed_url:
            # Parse URL to show components
            parsed_url = urlparse(signed_url)
            query_params = parse_qs(parsed_url.query)
            
            # Extract filename from path
            filename = parsed_url.path.split('/')[-1] if parsed_url.path else 'unknown'
            
            formatted_entry.host = parsed_url.netloc
            formatted_entry.path = parsed_url.path
            formatted_entry.filename = filename
            formatted_entry.expires_in_seconds = query_params.get('X-Amz-Expires', ['unknown'])[0]
            
            extracted_urls.append(signed_url)
        
        formatted_entries.append(formatted_entry)
    
    return DecodeResponse(
        success=True,
        message="Successfully decoded signed URLs",
        total_entries=len(data),
        entries=formatted_entries,
        extracted_urls=extracted_urls,
        total_urls=len(extracted_urls)
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "S3 Dataset Processor API",
        "version": "1.0.0",
        "description": "API for decoding and processing S3 presigned URLs",
        "endpoints": {
            "health": "/health",
            "decode": "/decode-urls (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "s3-dataset-processor",
        "version": "1.0.0"
    }

@app.post("/decode-urls", response_model=DecodeResponse)
async def decode_urls(request: DecodeRequest):
    """
    Decode signed URLs from encoded data.
    
    Args:
        request: Request containing encoded data
        
    Returns:
        Decoded URL information
    """
    try:
        if not request.encoded_data or not isinstance(request.encoded_data, str):
            raise HTTPException(
                status_code=400,
                detail="'encoded_data' must be a non-empty string"
            )
        
        # Decode the data
        decoded_data = decode_signed_urls(request.encoded_data)
        
        # Format response
        response = format_decoded_data(decoded_data)
        
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Decoding error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/decode-urls-get")
async def decode_urls_get(encoded_data: str):
    """
    Decode signed URLs via GET request (for testing).
    
    Args:
        encoded_data: URL-encoded JSON string as query parameter
        
    Returns:
        Decoded URL information
    """
    try:
        if not encoded_data:
            raise HTTPException(
                status_code=400,
                detail="'encoded_data' parameter is required"
            )
        
        # Decode the data
        decoded_data = decode_signed_urls(encoded_data)
        
        # Format response
        response = format_decoded_data(decoded_data)
        
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Decoding error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)