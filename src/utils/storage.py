# src/utils/storage.py
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, Tuple
import io
from PIL import Image
import numpy as np
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

class StorageManager:
    """Manage storage operations with S3/MinIO"""
    
    def __init__(self, endpoint_url: Optional[str] = None, 
                 bucket_name: str = "document-processing",
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        
        self.bucket_name = bucket_name
        
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            
            # Try to create bucket if it doesn't exist
            self._ensure_bucket_exists()
            logger.info(f"Successfully connected to S3/MinIO storage, bucket: {bucket_name}")
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to initialize storage: {e}")
            # Don't raise error, continue without storage
            self.s3_client = None
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} already exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                logger.info(f"Creating bucket: {self.bucket_name}")
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                raise e
    
    def upload_image(self, image_array: np.ndarray, key: str, 
                    format: str = 'PNG') -> bool:
        """Upload image array to storage"""
        if not self.s3_client:
            logger.warning("S3 client not initialized, skipping upload")
            return False
            
        try:
            image = Image.fromarray(image_array)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format=format)
            img_buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=img_buffer,
                ContentType=f'image/{format.lower()}'
            )
            logger.info(f"Image uploaded to: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload image {key}: {e}")
            return False
    
    def download_image(self, key: str) -> Optional[np.ndarray]:
        """Download image from storage as numpy array"""
        if not self.s3_client:
            return None
            
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data))
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to download image {key}: {e}")
            return None
    
    def object_exists(self, key: str) -> bool:
        """Check if object exists in storage"""
        if not self.s3_client:
            return False
            
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a presigned URL to access an object.

        Returns None if the client is not initialized or on failure.
        """
        if not self.s3_client:
            return None

        try:
            # For MinIO/S3, use generate_presigned_url
            url = self.s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {key}: {e}")
            return None