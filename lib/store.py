import os
import boto3
import botocore
import json
from boto3.s3.transfer import S3Transfer

def s3_client():
    s3 = boto3.client('s3' , 
        aws_access_key_id = 'AKIAJTQ76T2PBBLZT34A' , 
        aws_secret_access_key = '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')
    
    return s3

def s3_down_file(client , bucket_name , key , f):
    try:
        client.download_file(bucket_name , key , f)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print('The {} does not exist.'.format(key))
        else:
            raise

def s3_upload_file(client , bucket_name , key , f):
    #transfer = S3Transfer(client)
    try:
        client.upload_file(f , bucket_name , key,
                        ExtraArgs= {'ACL':'public-read'})
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print('The {} upload false.'.format(key))
        else:
            raise    

def s3_delete_file(client , bucket_name , key):
    try:
        client.delete_object(Bucket = bucket_name , Key = key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print('The {} upload false.'.format(key))
        else:
            raise 

def get_bucket(s3_client , bucket_name):
    for bucket in s3.buckets.all():
        if bucket.name == bucket_name:
            return bucket
    return None

def get_listfiles(s3_client , bucket_name):
    res = []
    response = s3_client.list_objects_v2(Bucket = bucket_name)
    content = response['Contents']
    for obj in content:
        res.append((obj['Key'] , obj['Size']))
    return res

def get_url(bucket_name , key_name):
    url = 'https://s3.us-east-2.amazonaws.com/{}/{}'.format(bucket_name , key_name)
    return url

if __name__ == '__main__':
    s3 = boto3.resource('s3' ,aws_access_key_id = 'AKIAJTQ76T2PBBLZT34A' , aws_secret_access_key ='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')
    bucket = get_bucket(s3 , 'pasingweb-video')
    
    s3 = s3_client()
    # ------ s3 list flie
    print(get_listfiles(s3 , 'pasingweb-video'))

    # ------ s3 upload file
    #s3_upload_file(s3 , 'parsing-other' , 'score.json','./tmp/score.json')

    # ------ s3 down file
    # s3_down_file(s3 , 'pasingweb-video' , 'score.json' , 'static/score.json')
    # with open('static/score.json' , 'r') as f:
    #     model = json.load(f)
    #     print(model)

    # ------ s3 delete file
    s3_delete_file(s3 , 'pasingweb-video' , 'score.json')
    
