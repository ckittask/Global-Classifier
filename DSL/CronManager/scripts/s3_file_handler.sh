#!/bin/bash

echo "Started Shell Script for S3 DataSet Processing"

# Check if environment variable is set
if [ -z "$signedUrls" ]; then
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

# API endpoint
API_URL="http://s3-dataset-processor:8001/decode-urls"

log "🔍 Calling S3 Dataset Processor API..."

# Call the API to decode URLs
response=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{\"encoded_data\":\"$data_generation_request\"}")

# Check if API call was successful
if [ $? -eq 0 ]; then
    log "✅ API call successful"
    echo "$response"
else
    log "❌ API call failed"
    exit 1
fi

log "🎉 Process completed"