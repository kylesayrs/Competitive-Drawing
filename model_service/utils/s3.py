import boto3
from io import BytesIO

def get_object_file_stream(bucket: str, key: str):
    obj = boto3.resource("s3").Object(bucket, key)
    tmp = obj.get()['Body'].read()
    return BytesIO(tmp)
