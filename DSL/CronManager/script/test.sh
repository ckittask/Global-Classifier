#!/bin/bash

echo "Started Shell Script for S3 DataSet Processing"

# Check if environment variable is set
if [ -z "$signedUrls" || [ -z "$datasetId" ]]; then
  echo "Please set the signedUrls environment variable."
  exit 1
fi

# Logging function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

data_generation_request="$signedUrls"

log "S3 data processing request received"
log "Encoded data length: ${#data_generation_request} characters"

# API endpoint for downloading datasets
API_URL="http://s3-dataset-processor:8001/download-datasets"
CURRENT_DATASET_ID="$datasetId"

log "🔍 Calling S3 Dataset Processor API to download files..."

# Call the API to download datasets with safe response parsing
response=$(curl -s -o /tmp/response_body.txt -w "%{http_code}" -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{\"encoded_data\":\"$data_generation_request\", \"extract_files\": true}")

http_code="$response"
response_body=$(cat /tmp/response_body.txt)

log "🔍 HTTP Status Code: $http_code"
log "🔍 Raw Response Body: $response_body"

# Check if API call was successful
if [ "$http_code" = "200" ] && [ -n "$response_body" ]; then
    log "✅ API call successful"
    
    # Check if the response indicates success using grep instead of jq
    if printf "%s\n" "$response_body" | grep -q '"success":true'; then
        success_status="true"
    else
        success_status="false"
    fi
    
    log "🔍 Success status: $success_status"
    
    if [ "$success_status" = "true" ]; then
        log "✅ S3 download and extraction successful"
        
        # Get successful downloads count using grep and sed
        successful_downloads=$(printf "%s\n" "$response_body" | grep -o '"successful_downloads":[0-9]*' | grep -o '[0-9]*' || echo "0")
        log "Successfully downloaded and extracted $successful_downloads files"
        
        # Generate datasets for each extracted folder
        log "🔄 Starting dataset generation for extracted folders..."
        
        # Extract folder information using grep and sed (fallback if jq fails)
        if command -v jq >/dev/null 2>&1; then
            # Use jq if available
            printf "%s\n" "$response_body" | jq -r '.extracted_folders[] | "\(.agency_id):\(.folder_path)"' 2>/dev/null | while IFS=':' read -r agency_id folder_path; do
                if [ -n "$agency_id" ] && [ -n "$folder_path" ]; then
                    log "🔄 Generating dataset for agency: $agency_id at path: $folder_path"
                    
                    # Call the dataset generation service
                    dataset_response=$(curl -s -X POST "http://dataset-gen-service:8000/generate-bulk" \
                        -H "Content-Type: application/json" \
                        -d "{\"data_path\": \"$folder_path\", \"output_filename\": \"$CURRENT_DATASET_ID\"}")
                    
                    if [ $? -eq 0 ]; then
                        # Check if generation was successful
                        if printf "%s\n" "$dataset_response" | grep -q '"status":"success"'; then
                            log "✅ Dataset generation successful for agency: $agency_id"
                            log "Response: $dataset_response"
                        else
                            log "❌ Dataset generation failed for agency: $agency_id"
                            log "Error response: $dataset_response"
                        fi
                    else
                        log "❌ Failed to call dataset generation service for agency: $agency_id"
                    fi
                else
                    log "⚠️ Skipping invalid folder: agency_id=$agency_id, folder_path=$folder_path"
                fi
            done
        else
            # Fallback parsing without jq
            log "⚠️ jq not available, using grep/sed for parsing"
            
            # Extract agency_id and folder_path pairs using grep and sed
            printf "%s\n" "$response_body" | grep -o '"agency_id":"[^"]*","folder_path":"[^"]*"' | while read -r folder_info; do
                agency_id=$(echo "$folder_info" | sed 's/.*"agency_id":"\([^"]*\)".*/\1/')
                folder_path=$(echo "$folder_info" | sed 's/.*"folder_path":"\([^"]*\)".*/\1/')
                
                if [ -n "$agency_id" ] && [ -n "$folder_path" ]; then
                    log "🔄 Generating dataset for agency: $agency_id at path: $folder_path"
                    
                    # Call the dataset generation service
                    dataset_response=$(curl -s -X POST "http://dataset-gen-service:8000/generate-bulk" \
                        -H "Content-Type: application/json" \
                        -d "{\"data_path\": \"$folder_path\"}")
                    
                    if [ $? -eq 0 ]; then
                        # Check if generation was successful
                        if printf "%s\n" "$dataset_response" | grep -q '"status":"success"'; then
                            log "✅ Dataset generation successful for agency: $agency_id"
                            log "Response: $dataset_response"
                        else
                            log "❌ Dataset generation failed for agency: $agency_id"
                            log "Error response: $dataset_response"
                        fi
                    else
                        log "❌ Failed to call dataset generation service for agency: $agency_id"
                    fi
                else
                    log "⚠️ Skipping invalid folder: agency_id=$agency_id, folder_path=$folder_path"
                fi
            done
        fi
        
        log "✅ Dataset generation process completed"
        
    else
        log "❌ S3 download failed - success status: $success_status"
        log "Response: $response_body"
        exit 1
    fi
    
elif [ "$http_code" != "200" ]; then
    log "❌ API call failed with HTTP status: $http_code"
    log "Response: $response_body"
    exit 1
else
    log "❌ API call failed - no response received"
    exit 1
fi

# Cleanup temp file
rm -f /tmp/response_body.txt

log "🎉 S3 Dataset Processing and Generation completed successfully"
DATASET_FILE_NAME="$CURRENT_DATASET_ID.json"

# Read structure_name from config - it's nested under dataset_generation
CONFIG_FILE="/app/config/config.yaml"
STRUCTURE_NAME=$(grep -A 10 "dataset_generation:" "$CONFIG_FILE" | grep "structure_name:" | sed 's/.*structure_name:[[:space:]]*"\([^"]*\)".*/\1/' | tr -d ' ')

# If config reading fails, exit with error
if [ -z "$STRUCTURE_NAME" ]; then
    log "❌ Could not read structure_name from config file: $CONFIG_FILE"
    log "Debug: Attempting to read dataset_generation section..."
    grep -A 10 "dataset_generation:" "$CONFIG_FILE" || log "Could not find dataset_generation section"
    exit 1
fi

log "📋 Using structure_name from config: $STRUCTURE_NAME"

# Construct the generated dataset file path using output_datasets, structure_name, and dataset ID
GENERATED_DATASET_PATH="output_datasets/${STRUCTURE_NAME}/${CURRENT_DATASET_ID}.json"

# Store the generated dataset in S3
log "📤 Transferring generated dataset to S3..."
log "Dataset path: $GENERATED_DATASET_PATH"
log "Dataset filename: $DATASET_FILE_NAME"

# Call the store_in_s3 endpoint
s3_response=$(curl -s -X POST "http://ruuter-public:8086/global-classifier/data/store_in_s3" \
  -H "Content-Type: application/json" \
  -d "{\"filePath\": \"$GENERATED_DATASET_PATH\", \"fileName\": \"$DATASET_FILE_NAME\"}")

log "S3 transfer response: $s3_response"

# Check if S3 transfer was successful
if echo "$s3_response" | grep -q '"message":"Dataset successfully transferred to S3"'; then
  log "✅ Dataset successfully transferred to S3"
  log "S3 location: $(echo "$s3_response" | grep -o '"location":"[^"]*"' | cut -d'"' -f4)"
else
  log "❌ Failed to transfer dataset to S3"
  log "Error: $s3_response"
  exit 1
fi

exit 0