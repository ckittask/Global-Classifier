#!/bin/bash

echo "Started Shell Script for DataSet Generation"

# Check if environment variable is set
if [ -z "$signedUrls" ]; then
  echo "Please set the signedUrls environment variable."
  exit 1
fi

# Install unzip if not available
if ! command -v unzip &> /dev/null; then
    log "📦 Installing unzip utility..."
    
    # Detect the package manager and install unzip
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        apt-get update && apt-get install -y unzip
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        yum install -y unzip
    elif command -v apk &> /dev/null; then
        # Alpine
        apk add --no-cache unzip
    else
        log "❌ Could not detect package manager to install unzip"
        exit 1
    fi
    
    log "✅ unzip installed successfully"
fi

# Logging function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Paths
LOG_FILE="/app/logs/generation_$(date +%Y%m%d).log"
DATA_DIR="/app/data"  # This should be mapped to cron_data volume
TMP_DIR="/tmp"
mkdir -p "$DATA_DIR"

# HARDCODED URLs for debugging
URL_ARRAY=(
  "http://minio:9000/ckb/agencies/Politsei-_ja_Piirivalveamet/Politsei-_ja_Piirivalveamet.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250530%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250530T041900Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=14aaba4daf7676424652d1274bbdc1880bf51b2d56312d8eb95a08346d3b8df6"
  "http://minio:9000/ckb/agencies/ID.ee/ID.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250530%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250530T041900Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c6a271b8c83a91ac8aafdf423cd2bb5d6b5375e5c9a02a9a3f6596c39f323637"
)

log "Using hardcoded URLs for debugging:"
for url in "${URL_ARRAY[@]}"; do
  echo "$url"
done

# Download files using curl directly to /tmp
log "Starting download of hardcoded URLs..."
for url in "${URL_ARRAY[@]}"; do
  [ -z "$url" ] && continue

  # Extract filename from URL (remove query parameters)
  filename=$(basename "${url%%\?*}")
  tmp_path="$TMP_DIR/$filename"

  log "Downloading: $filename from $url"
  log "Temp path: $tmp_path"
  
  # Download directly to tmp without quotes around the path variable
  curl -o $tmp_path "$url"

  if [ $? -eq 0 ]; then
    log "✅ Downloaded to /tmp: $tmp_path"
    
    # Verify the file was actually downloaded and has content
    if [ -f $tmp_path ] && [ -s $tmp_path ]; then
      file_size=$(ls -lh $tmp_path | awk '{print $5}')
      log "File size: $file_size"
    else
      log "❌ Downloaded file is empty or doesn't exist: $tmp_path"
    fi
  else
    log "❌ Failed to download: $url"
    log "Failed curl command: curl -fSL -o $tmp_path \"$url\""
  fi
done

log "✅ All downloads completed to $TMP_DIR"

# Copy files from /tmp to cron_data volume (/app/data)
log "🔁 Copying downloaded files from $TMP_DIR to $DATA_DIR..."

for file in "$TMP_DIR"/*.zip "$TMP_DIR"/*.tar.gz "$TMP_DIR"/*.tgz; do
  # Skip if the pattern didn't match any files
  if [[ "$file" == *"*"* ]]; then
    continue
  fi
  
  if [ ! -f "$file" ]; then
    continue
  fi
  
  filename=$(basename "$file")
  dest="$DATA_DIR/$filename"
  
  log "Copying: $file -> $dest"
  
  if cp "$file" "$dest" 2>/dev/null; then
    log "✅ Copied to cron_data volume: $dest"
    rm -f "$file"  # Clean up temp file after successful copy
    log "🗑️ Removed temp file: $file"
  else
    log "❌ Failed to copy file: $file"
    log "Debug info:"
    log "  Source file exists: $([ -f "$file" ] && echo "yes" || echo "no")"
    log "  Source file size: $(stat -c%s "$file" 2>/dev/null || echo "unknown")"
    log "  Destination directory writable: $([ -w "$DATA_DIR" ] && echo "yes" || echo "no")"
    log "  Available disk space: $(df -h "$DATA_DIR" | tail -1 | awk '{print $4}')"
  fi
done

# Extract all zip files in the cron_data volume
log "📦 Starting extraction of zip files in $DATA_DIR..."

extracted_count=0
for zip_file in "$DATA_DIR"/*.zip; do
  [ -f "$zip_file" ] || continue  # skip if no zip files found
  
  zip_filename=$(basename "$zip_file")
  extract_dir="$DATA_DIR/${zip_filename%.*}"  # Remove .zip extension for directory name
  
  log "Processing zip file: $zip_filename"
  log "Extract directory: $extract_dir"
  
  # Extract zip file directly to DATA_DIR first
  if unzip -q "$zip_file" -d "$DATA_DIR"; then
    log "✅ Successfully extracted: $zip_filename"
    
    # Check if extraction created a directory with the same name as the zip
    zip_name_dir="$DATA_DIR/${zip_filename%.*}"
    
    if [ -d "$zip_name_dir" ] && [ "$zip_name_dir" != "$extract_dir" ]; then
      # If the extracted directory has the same name, it's already in the right place
      log "📁 Zip extracted to correct location: $zip_name_dir"
    else
      # Look for any directory that was just created
      newest_dir=$(find "$DATA_DIR" -maxdepth 1 -type d -name "*${zip_filename%.*}*" | head -1)
      if [ -n "$newest_dir" ] && [ "$newest_dir" != "$extract_dir" ]; then
        mv "$newest_dir" "$extract_dir"
        log "📁 Moved extracted directory to: $extract_dir"
      fi
    fi
    
    # Count extracted files
    extracted_files=$(find "$extract_dir" -type f 2>/dev/null | wc -l)
    log "📁 Extracted $extracted_files files"
    extracted_count=$((extracted_count + 1))
    
    # List some extracted contents (first few files)
    log "Sample extracted contents:"
    find "$extract_dir" -type f 2>/dev/null | head -5 | while read extracted_file; do
      file_size=$(ls -lh "$extracted_file" | awk '{print $5}')
      relative_path=$(echo "$extracted_file" | sed "s|$extract_dir/||")
      log "  $relative_path ($file_size)"
    done
    
    # Remove the zip file after successful extraction
    rm -f "$zip_file"
    log "🗑️ Removed zip file: $zip_filename"
  else
    log "❌ Failed to extract: $zip_filename"
    log "Keeping zip file for debugging: $zip_file"
  fi
done

# Clean up any remaining empty directories
log "🧹 Cleaning up empty directories..."
find "$DATA_DIR" -type d -empty -delete 2>/dev/null || true

log "✅ All downloads, copying, and extraction completed"

# List final files and directories in cron_data volume
log "Final contents in $DATA_DIR (cron_data volume):"
ls -la "$DATA_DIR"

# Summary
valid_extractions=$(find "$DATA_DIR" -type d -mindepth 1 | wc -l)
total_files=$(find "$DATA_DIR" -type f | wc -l)
log "📊 Summary:"
log "  - Successful extractions: $extracted_count"
log "  - Total directories: $valid_extractions"
log "  - Total files: $total_files"

# Optional: Show disk usage
disk_usage=$(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}')
log "  - Total disk usage: $disk_usage"

# Configuration variables for dataset generation
API_URL="http://dataset-gen-service:8000"
RUUTER_URL="http://ruuter-public:8086/global-classifier"
S3_BUCKET="$bucketName"

# Array to track generated datasets
declare -a GENERATED_DATASETS=()

# Now generate datasets for each extracted directory
log "🔄 Starting dataset generation for all extracted directories..."

for extract_dir in "$DATA_DIR"/*/; do
  [ -d "$extract_dir" ] || continue  # skip if no directories found
  
  # Use the directory name as the dataset identifier
  dataset_name=$(basename "$extract_dir")
  
  log "🔄 Starting dataset generation for directory: $dataset_name"
  
  # Construct the payload for this specific directory
  data_generation_payload=$(cat <<EOF
{
  "dataset_structure_name": "single_question",
  "prompt_template_name": "institute_topic_question",
  "data_path": "$DATA_DIR",
  "traversal_strategy": "institutional",
  "no_of_samples": 10 
}
EOF
)

  datasetStructureName ="single_question"
  generatedDatasetFolder = "output_datasets"

  # Call the bulk generation API for this directory
  log "Calling dataset generation service for: $dataset_name"
  response=$(curl -s -X POST "$API_URL/generate-bulk" \
    -H "Content-Type: application/json" \
    -d "$data_generation_payload")

  log "Response from dataset generation service: $response"

  # Track generation status
  generation_success=false
  if echo "$response" | grep -q '"status":"success"'; then
    log "✅ Dataset generated successfully for: $dataset_name"
    generation_success=true
    
    # Prepare for S3 transfer
    ZIP_FILE="${generatedDatasetFolder}/${dataset_name}_${datasetStructureName}.zip"
    S3_KEY="datasets/${dataset_name}_${datasetStructureName}.zip"
    
    # Call the Ruuter endpoint to transfer the zip file to S3
    log "Transferring dataset zip to S3 for: $dataset_name"
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
      log "✅ Dataset zip successfully transferred to S3 for: $dataset_name"
      location=$(echo "$s3_response" | grep -o '"location":"[^"]*' | cut -d'"' -f4 || echo "datasets/${dataset_name}_${datasetStructureName}.zip")
      log "S3 Location: $location"
      
      # Clean up the zip file to save space
      rm -f "$ZIP_FILE"
      log "Removed local zip file to save space: $ZIP_FILE"
      
      # Add to successful datasets list
      GENERATED_DATASETS+=("$dataset_name:$location")
    else
      log "❌ Failed to transfer dataset zip to S3 for: $dataset_name"
      log "Error: $s3_response"
    fi
  else
    log "❌ Dataset generation failed for: $dataset_name"
    log "Error: $(echo "$response" | jq -r '.message // "Unknown error"')"
  fi
done

# # Final summary
# log "📊 Final Summary:"
# log "  - Successful extractions: $extracted_count"
# log "  - Generated datasets: ${#GENERATED_DATASETS[@]}"

# # List all generated datasets
# if [ ${#GENERATED_DATASETS[@]} -gt 0 ]; then
#   log "Generated datasets:"
#   for dataset in "${GENERATED_DATASETS[@]}"; do
#     log "  - $dataset"
#   done
#   log "Dataset generation and upload process completed successfully"
#   exit 0
# else
#   log "❌ No datasets were successfully generated"
#   exit 1
# fi