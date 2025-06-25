"""FastAPI application with refactored structure."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn
import re
import json
import requests
from loguru import logger
import sys

from config.settings import settings
from models.schemas import DownloadRequest, DownloadResponse
from services.url_decoder_service import URLDecoderService
from services.download_service import DownloadService
from services.extraction_service import ExtractionService
from handlers.response_handler import ResponseHandler

logger.remove()
# Add stdout handler with your preferred format
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

# Initialize services
url_decoder_service = URLDecoderService()
download_service = DownloadService()
extraction_service = ExtractionService()
response_handler = ResponseHandler()


def process_callback_background(file_path: str, encoded_results: str):
    """
    Background function to process the dataset generation callback.
    This runs asynchronously after returning 200 to the caller.
    """
    try:
        print(f"[CALLBACK] Starting background processing for: {file_path}")

        # Extract dataset ID from file path (e.g., output_datasets/single_question/12.json -> 12)
        dataset_id_match = re.search(r"/([^/]+)\.json$", file_path)
        dataset_id = dataset_id_match.group(1) if dataset_id_match else "unknown"

        logger.info(f"[CALLBACK] Extracted dataset ID: {dataset_id}")

        # Decode the results using the existing service
        decoded_results = url_decoder_service.decode_signed_urls(encoded_results)

        logger.info(f"[CALLBACK] Decoded {len(decoded_results)} results")

        # Process the decoded results to create the required payload
        agencies = []
        overall_success = True

        for i, result in enumerate(decoded_results):
            # Extract agency_id from dataset_metadata
            dataset_metadata = result.get("dataset_metadata", {})
            agency_id = dataset_metadata.get("agency_id", "unknown")
            success = result.get("success", False)

            sync_status = "Synced_with_CKB" if success else "Sync_with_CKB_Failed"

            agencies.append({"agencyId": agency_id, "syncStatus": sync_status})

            logger.info(
                f"[CALLBACK] Agency {i + 1}: ID={agency_id}, Success={success}, Status={sync_status}"
            )

            if not success:
                overall_success = False

        generation_status = (
            "Generation_Success" if overall_success else "Generation_Failed"
        )

        # Create the exact payload format requested
        callback_payload = {
            "agencies": agencies,
            "datasetId": dataset_id,
            "generationStatus": generation_status,
        }

        # Log the processed callback
        logger.info(f"[CALLBACK] {json.dumps(callback_payload, indent=2)}")
        logger.info(f"[CALLBACK] Dataset ID: {dataset_id}")
        logger.info(f"[CALLBACK] Generation Status: {generation_status}")
        logger.info(f"[CALLBACK] Total Agencies: {len(agencies)}")
        logger.info(
            f"[CALLBACK] Successful Agencies: {len([a for a in agencies if a['syncStatus'] == 'Synced_with_CKB'])}"
        )
        logger.info(
            f"[CALLBACK] Failed Agencies: {len([a for a in agencies if a['syncStatus'] == 'Sync_with_CKB_Failed'])}"
        )

        STATUS_UPDATE_URL = (
            "http://ruuter-public:8086/global-classifier/agencies/data/generation"
        )

        logger.info(f"[CALLBACK] Sending callback payload to: {STATUS_UPDATE_URL}")

        try:
            # Send POST request to the status update endpoint
            response = requests.post(
                STATUS_UPDATE_URL,
                json=callback_payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            logger.info(
                f"[CALLBACK] Status update response - HTTP Status: {response.status_code}"
            )
            logger.info(f"[CALLBACK] Status update response body: {response.text}")

            if response.status_code == 200:
                logger.info(
                    "[CALLBACK] ✅ Successfully sent callback payload to status update endpoint"
                )
            else:
                logger.warning(
                    f"[CALLBACK] ⚠️ Status update endpoint returned non-200 status: {response.status_code}"
                )
                logger.info(f"[CALLBACK] Response: {response.text}")

        except requests.exceptions.RequestException as webhook_error:
            logger.error(
                f"[CALLBACK] ❌ Error sending callback to status update endpoint: {str(webhook_error)}"
            )
            logger.debug(f"[CALLBACK] URL: {STATUS_UPDATE_URL}")
            logger.debug(
                f"[CALLBACK] Payload: {json.dumps(callback_payload, indent=2)}"
            )

        except Exception as unexpected_error:
            logger.error(
                f"[CALLBACK] ❌ Unexpected error during status update: {str(unexpected_error)}"
            )

    except Exception as e:
        logger.error(f"[CALLBACK] Error in background processing: {str(e)}")
        logger.debug(f"[CALLBACK] File path: {file_path}")
        logger.debug(
            f"[CALLBACK] Encoded results length: {len(encoded_results) if encoded_results else 0}"
        )
        # You might want to implement retry logic or error handling here


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "endpoints": {
            "health": "/health",
            "decode": "/decode-urls (POST)",
            "download": "/download-datasets (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "s3-dataset-processor",
        "version": settings.API_VERSION,
        "data_dir": settings.DATA_DIR,
    }


@app.post("/download-datasets", response_model=DownloadResponse)
async def download_datasets(request: DownloadRequest):
    """
    Download dataset files from signed URLs and extract if needed.

    Args:
        request: Request containing encoded data and download options

    Returns:
        Download results with file locations
    """
    try:
        if not request.encoded_data or not isinstance(request.encoded_data, str):
            raise HTTPException(
                status_code=400, detail="'encoded_data' must be a non-empty string"
            )

        # Decode the data using service
        decoded_data = url_decoder_service.decode_signed_urls(request.encoded_data)
        logger.info(f"Starting download for {len(decoded_data)} files")

        # Process downloads using service
        downloaded_files, successful_downloads, failed_downloads = (
            download_service.process_downloads(decoded_data)
        )

        # Process extractions using service
        extracted_folders = extraction_service.process_extractions(
            downloaded_files, request.extract_files
        )

        # Format response using handler
        response = response_handler.format_download_response(
            decoded_data,
            downloaded_files,
            successful_downloads,
            failed_downloads,
            extracted_folders,
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Decoding error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/process-generation-callback")
async def process_generation_callback(request: dict, background_tasks: BackgroundTasks):
    """
    Process dataset generation callback in the background.
    Returns 200 immediately and processes the callback asynchronously.

    Args:
        request: Dictionary containing 'file_path' and 'results' (encoded)
        background_tasks: FastAPI background tasks handler

    Returns:
        Immediate 200 response, actual processing happens in background
    """
    try:
        file_path = request.get("file_path", "")
        encoded_results = request.get("results", "")

        if not encoded_results or not isinstance(encoded_results, str):
            raise HTTPException(
                status_code=400, detail="'results' must be a non-empty string"
            )

        if not file_path:
            raise HTTPException(status_code=400, detail="'file_path' is required")

        # Add the background task
        background_tasks.add_task(
            process_callback_background, file_path, encoded_results
        )

        # Return immediate response
        return {
            "message": "Callback processing started",
            "status": "accepted",
            "file_path": file_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
