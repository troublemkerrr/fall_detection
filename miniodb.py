from minio import Minio
from minio.error import S3Error
import os
import uuid

class VideoStorage:
    def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self.ensure_bucket_exists()
    
    def ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            raise Exception(f"无法连接到MinIO: {str(e)}")
    
    def upload_video(self, file_path, object_name=None):
        if not object_name:
            object_name = f"events/{uuid.uuid4()}.mp4"
        
        try:
            self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path
            )
            return f"{self.bucket_name}/{object_name}"
        except S3Error as e:
            raise Exception(f"上传视频失败: {str(e)}")
    
    def download_video(self, object_path, dest_path):
        try:
            # 从完整路径中提取bucket和object
            if '/' in object_path:
                bucket, object_name = object_path.split('/', 1)
            else:
                bucket = self.bucket_name
                object_name = object_path
                
            self.client.fget_object(bucket, object_name, dest_path)
            return True
        except S3Error as e:
            raise Exception(f"下载视频失败: {str(e)}")
    
    def delete_video(self, object_path):
        try:
            if '/' in object_path:
                bucket, object_name = object_path.split('/', 1)
            else:
                bucket = self.bucket_name
                object_name = object_path
                
            self.client.remove_object(bucket, object_name)
            return True
        except S3Error as e:
            raise Exception(f"删除视频失败: {str(e)}")