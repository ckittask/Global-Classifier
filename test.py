import requests
import json

url = 'http://localhost:8086/global-classifier/data/generate'

payload = json.dumps({
  'signedUrls': [
    {
      'agencyid': '1234',
      'signedS3url': 'http://minio:9000/ckb/agencies/ID.ee/ID.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250530%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250530T041900Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c6a271b8c83a91ac8aafdf423cd2bb5d6b5375e5c9a02a9a3f6596c39f323637'
    },
    {
      'agencyid': '5234',
      'signedS3url': 'http://minio:9000/ckb/agencies/Politsei-_ja_Piirivalveamet/Politsei-_ja_Piirivalveamet.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250530%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250530T041900Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=14aaba4daf7676424652d1274bbdc1880bf51b2d56312d8eb95a08346d3b8df6'
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request('POST', url, headers=headers, data=payload)

print(response.text)
