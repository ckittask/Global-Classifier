#!/bin/bash

cd "$(dirname "$0")"

# Source the config file if needed
if [ -f "../config/config.ini" ]; then
  source ../config/config.ini
fi

script_name=$(basename $0)

echo $(date -u +"%Y-%m-%d %H:%M:%S.%3NZ") - $script_name started

# Get the last synced timestamp from database
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-global_classifier}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-dbadmin}

# Get last synced timestamp from database
last_synced=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -d $DB_NAME -U $DB_USER -t -c "SELECT last_synced_timestamp FROM public.sync_timestamps WHERE sync_type = 'agency_data' ORDER BY updated_at DESC LIMIT 1;")

# If no timestamp found, use 24 hours ago
if [ -z "$last_synced" ]; then
  last_synced=$(date -u -d "24 hours ago" +"%Y-%m-%dT%H:%M:%S.000Z")
else
  # Trim whitespace
  last_synced=$(echo $last_synced | xargs)
fi

echo "Syncing agencies with timestamp: $last_synced"

current_time=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")

# Call the agencies/data/sync endpoint
response=$(curl -s -w "\nHTTP_STATUS_CODE:%{http_code}" -X POST \
  "http://ruuter-public:8086/global-classifier/agencies/data/sync" \
  -H "Cookie: customJwtCookie=$cookie" "Content-Type: application/json" \
  -d "{\"lastSyncedTimestamp\": \"$last_synced\"}")

# Extract the HTTP status code from the response
http_status=$(echo "$response" | grep "HTTP_STATUS_CODE" | awk -F: '{print $2}' | tr -d '[:space:]')

# Extract the body from the response
http_body=$(echo "$response" | grep -v "HTTP_STATUS_CODE")

# Check if the request was successful
if [ "$http_status" -ge 200 ] && [ "$http_status" -lt 300 ]; then
  echo "Agency data sync completed successfully."
  echo "Response: $http_body"
  
  # Update the last synced timestamp in the database
  PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -d $DB_NAME -U $DB_USER -c "
    UPDATE public.sync_timestamps 
    SET last_synced_timestamp = '$current_time', updated_at = CURRENT_TIMESTAMP 
    WHERE sync_type = 'agency_data';"
  
  echo "Updated last synced timestamp in database to: $current_time"
else
  echo "Agency data sync failed with status code $http_status."
  echo "Response: $http_body"
fi

echo $(date -u +"%Y-%m-%d %H:%M:%S.%3NZ") - $script_name finished