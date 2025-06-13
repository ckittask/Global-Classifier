#!/bin/bash

echo "Started Shell Script to DataSet Generation" 
# Ensure required environment variables are set
if [ -z "$datasetStructureName" ] || [ -z "$promptTemplateName" ] || [ -z "$traversalType" ] || [ -z "$bucketName" ] || [ -z "$prefix" ] || [ -z "$noOfSamples" ] || [ -z "$generatedDatasetFolder" ]; then
  echo "One or more environment variables are missing."
  echo "Please set datasetStructureName, promptTemplateName, traversalType, bucketName, prefix, noOfSamples, generatedDatasetFolder."
  exit 1
fi

# Configuration variables
DATA_DIR="/app/data"
API_URL="http://dataset-gen-service:8000"
RUUTER_URL="http://ruuter-public:8086/global-classifier"

# Construct the payload using here document
data_generation_payload=$(cat <<EOF
{
  "dataset_structure_name": "$datasetStructureName",
  "prompt_template_name": "$promptTemplateName",
  "data_path": "$DATA_DIR",
  "traversal_strategy": "$traversalType",
  "no_of_samples": $noOfSamples 
}
EOF
)

DATASET_TYPE=$datasetStructureName
DATASET_FOLDER_NAME=$generatedDatasetFolder

# Call the bulk generation API
log "Calling the dataset generation service..."
response=$(curl -s -X POST "$API_URL/generate-bulk" \
  -H "Content-Type: application/json" \
  -d "$data_generation_payload")

log "Response from dataset generation service: $response"

# Track generation status
generation_success=false
if echo "$response" | grep -q '"status":"success"'; then
  log "✅ Dataset generated successfully"
  generation_success=true
else
  log "❌ Dataset generation failed: $(echo "$response" | jq -r '.message // "Unknown error"')"
fi

ZIP_FILE="${DATASET_FOLDER_NAME}/${DATASET_TYPE}.zip"
S3_KEY="datasets/${DATASET_TYPE}.zip"
S3_BUCKET="$bucketName"

# Call the Ruuter endpoint to transfer the zip file to S3
log "Transferring dataset zip to S3..."
s3_response=$(curl -s -X POST "${RUUTER_URL}/data/store_in_s3" \
  -H "Content-Type: application/json" \
  -d @- << EOF
{
  "filePath": "${ZIP_FILE}",
  "bucketName": "${S3_BUCKET}",
  "s3Key": "${S3_KEY}"
}
EOF
)

if echo "$s3_response" | grep -q '"operationSuccessful":true'; then
  log "✅ Dataset zip successfully transferred to S3"
  location=$(echo "$s3_response" | grep -o '"location":"[^"]*' | cut -d'"' -f4 || echo "${S3_PREFIX}${DATASET_ID}.zip")
  log "S3 Location: $location"
  
  # Clean up the zip file to save space
  rm -f "$ZIP_FILE"
  log "Removed local zip file to save space"
  
  # Success exit code
  exit 0
else
  log "❌ Failed to transfer dataset zip to S3"
  log "Error: $s3_response"
  exit 1
fi

log "Dataset generation and upload process completed"