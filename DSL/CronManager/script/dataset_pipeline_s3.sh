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
CURRENT_DATASET_ID=$(echo "$CURRENT_DATASET_ID" | tr -d '"')

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
        
        # Prepare dataset generation payload as a list
        log "🔄 Preparing dataset generation payload..."
        
        # Create temporary file for building the JSON payload
        temp_payload="/tmp/dataset_payload.json"
        echo '{"datasets": [' > "$temp_payload"
        
        first_entry=true
        
        # Extract folder information and build the payload list
        if command -v jq >/dev/null 2>&1; then
            # Use jq if available
            printf "%s\n" "$response_body" | jq -r '.extracted_folders[] | "\(.agency_id):\(.folder_path)"' 2>/dev/null | while IFS=':' read -r agency_id folder_path; do
                if [ -n "$agency_id" ] && [ -n "$folder_path" ]; then
                    if [ "$first_entry" = false ]; then
                        echo ',' >> "$temp_payload"
                    fi
                    echo "    {" >> "$temp_payload"
                    echo "      \"agency_id\": \"$agency_id\"," >> "$temp_payload"
                    echo "      \"data_path\": \"$folder_path\"," >> "$temp_payload"
                    echo "      \"output_filename\": \"$CURRENT_DATASET_ID\"" >> "$temp_payload"
                    echo "    }" >> "$temp_payload"
                    first_entry=false
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
                    if [ "$first_entry" = false ]; then
                        echo ',' >> "$temp_payload"
                    fi
                    echo "    {" >> "$temp_payload"
                    echo "      \"agency_id\": \"$agency_id\"," >> "$temp_payload"
                    echo "      \"data_path\": \"$folder_path\"," >> "$temp_payload"
                    echo "      \"output_filename\": \"$CURRENT_DATASET_ID\"" >> "$temp_payload"
                    echo "    }" >> "$temp_payload"
                    first_entry=false
                fi
            done
        fi
        
        # Close the JSON array and object
        echo '' >> "$temp_payload"
        echo ']}' >> "$temp_payload"
        
        # Read the complete payload
        payload_content=$(cat "$temp_payload")
        log "🔍 Dataset generation payload: $payload_content"
        
        # Call the dataset generation service with the list payload
        log "🔄 Calling dataset generation service for bulk processing..."
        
        dataset_response=$(curl -s -o /tmp/dataset_response_body.txt -w "%{http_code}" -X POST "http://dataset-gen-service:8000/generate-bulk" \
            -H "Content-Type: application/json" \
            -d "$payload_content")
        
        dataset_http_code="$dataset_response"
        dataset_response_body=$(cat /tmp/dataset_response_body.txt)

        log "🔍 Dataset Generation HTTP Status Code: $dataset_http_code"
        
        log "🔍 Dataset Generation HTTP Status Code: $dataset_http_code"
        log "🔍 Dataset Generation Response: $dataset_response_body"
        
        if [ "$dataset_http_code" = "200" ]; then
            log "✅ Dataset generation request submitted successfully"
            log "✅ Background task initiated for dataset processing"
            log "Response: $dataset_response_body"
        else
            log "❌ Failed to submit dataset generation request"
            log "HTTP Status: $dataset_http_code"
            log "Error response: $dataset_response_body"
            # Cleanup temp files
            rm -f /tmp/dataset_payload.json /tmp/dataset_response_body.txt /tmp/response_body.txt
            exit 1
        fi
        
        # Cleanup temp files
        rm -f /tmp/dataset_payload.json /tmp/dataset_response_body.txt
        
        log "✅ S3 Dataset Processing completed successfully"
        log "✅ Dataset generation is running in background"
        
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

log "🎉 S3 Dataset Processing script completed successfully"
log "📋 Note: Dataset generation is running as a background task"

exit 0