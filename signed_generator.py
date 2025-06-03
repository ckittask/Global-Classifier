import boto3

def generate_presigned_url(bucket_name, object_key, expiration=14400):  # 4 hours
    s3_client = boto3.client('s3')
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': object_key},
        ExpiresIn=expiration
    )
    return url

# Example
bucket = 'converseai'
key = 'ID.ee/ID.zip'

url = generate_presigned_url(bucket, key)

# Save to file
output_file = 'signed_urls.txt'
with open(output_file, 'w') as f:
    f.write(url + '\n')

print(f"Presigned URL saved to {output_file}")
