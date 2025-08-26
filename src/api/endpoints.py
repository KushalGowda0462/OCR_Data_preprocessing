# src/api/endpoints.py
import numpy as np, cv2
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, APIRouter, Query
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import redis
import json
import time
import logging
from typing import Optional
import os
from PIL import Image
from ..segmentation import lines_mmocr, lines_kraken
from ..preprocessing.core import DocumentPreprocessor, preprocess_safe
from src.preprocessing.core import DocumentPreprocessor
from src.preprocessing.models import PreprocessingConfig, ProcessingResult, PreprocessingError
from src.utils.storage import StorageManager
from config.settings import settings

# Initialize application
app = FastAPI(title="Document Preprocessing Service", version="1.0.0")

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize Redis with error handling
try:
    redis_client = redis.Redis(
        host=settings.redis_host, 
        port=settings.redis_port, 
        db=settings.redis_db
    )
    redis_client.ping()  # Test connection
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
    # Simple in-memory fallback
    class MemoryStorage:
        def __init__(self):
            self.data = {}
        
        def setex(self, key, expire, value):
            self.data[key] = value
        
        def get(self, key):
            return self.data.get(key)
        
        def lpush(self, key, value):
            if key not in self.data:
                self.data[key] = []
            self.data[key].insert(0, value)
    
    redis_client = MemoryStorage()

# Initialize Preprocessor
preprocessing_config = PreprocessingConfig(
    deskew_enabled=settings.deskew_enabled,
    line_removal_enabled=settings.line_removal_enabled
)
preprocessor = DocumentPreprocessor(preprocessing_config)

# Initialize Storage Manager with error handling
try:
    storage_manager = StorageManager(
        endpoint_url=settings.s3_endpoint,
        bucket_name=settings.s3_bucket,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key
    )
    logger.info("Storage manager initialized successfully")
except Exception as e:
    logger.warning(f"Storage initialization failed: {e}. Using fallback storage.")
    # Fallback storage implementation
    class FallbackStorage:
        def __init__(self, base_path: str = "data/storage"):
            self.base_path = base_path
            os.makedirs(base_path, exist_ok=True)
            os.makedirs(os.path.join(base_path, "processed"), exist_ok=True)
            logger.info(f"Using fallback storage at: {os.path.abspath(base_path)}")
        
        def upload_image(self, image_array, key: str, format: str = 'PNG') -> bool:
            try:
                output_path = os.path.join(self.base_path, key)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                image = Image.fromarray(image_array)
                image.save(output_path, format=format)
                logger.info(f"Image saved to fallback storage: {output_path}")
                return True
            except Exception as e:
                logger.error(f"Fallback storage failed: {e}")
                return False
        
        def download_image(self, key: str):
            try:
                image_path = os.path.join(self.base_path, key)
                if not os.path.exists(image_path):
                    return None
                image = Image.open(image_path)
                return np.array(image)
            except Exception as e:
                logger.error(f"Failed to load from fallback: {e}")
                return None
        
        def object_exists(self, key: str) -> bool:
            return os.path.exists(os.path.join(self.base_path, key))
    
    storage_manager = FallbackStorage()

@app.get("/")
async def root():
    return {"message": "Document Preprocessing Service", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Simple health check endpoint without external dependencies"""
    try:
        # Test basic functionality
        test_image = np.ones((10, 10), dtype=np.uint8) * 255
        _ = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR) if len(test_image.shape) == 2 else test_image
        
        # Test Redis connection if available
        redis_status = "not_configured"
        if redis_client:
            try:
                if hasattr(redis_client, 'ping'):
                    redis_client.ping()
                    redis_status = "connected"
                else:
                    redis_status = "memory_cache"
            except:
                redis_status = "error"
        
        # Test storage
        storage_status = "working"
        try:
            test_storage_image = np.ones((5, 5), dtype=np.uint8) * 255
            if hasattr(storage_manager, 'upload_image'):
                storage_manager.upload_image(test_storage_image, "health_test.png")
                storage_status = "working"
            else:
                storage_status = "unknown"
        except:
            storage_status = "error"
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "redis": redis_status,
                "storage": storage_status,
                "opencv": "working",
                "numpy": "working"
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/preprocess/{document_id}")
async def preprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = None
):
    try:
        # Get image data
        if file:
            image_data = await file.read()
        elif image_url:
            import requests
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content
        else:
            raise HTTPException(status_code=400, detail="Either file or image_url must be provided")
        
        # Process in background
        background_tasks.add_task(process_image_task, document_id, image_data)
        
        return {"message": "Processing started", "document_id": document_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{document_id}")
async def get_processing_status(document_id: str):
    try:
        result_data = redis_client.get(f"preprocessing:{document_id}")
        if not result_data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        result = json.loads(result_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_image_task(document_id: str, image_data: bytes):
    """Background task to process an image"""
    start_time = time.time()
    
    try:
        import io
        import numpy as np
        
        # Load image
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        original_shape = image_array.shape
        
        # Process image
        processed_image = preprocessor.preprocess_image(image_array)
        ink_color = preprocessor.detect_ink_color(image_array)
        
        # Save to storage
        output_key = f"processed/{document_id}.png"
        success = storage_manager.upload_image(processed_image, output_key)
        
        if not success:
            logger.warning(f"Failed to upload image {output_key}, but continuing processing")
            # Don't raise error, just continue without storage
        
        # Store result
        result = ProcessingResult(
            document_id=document_id,
            processing_time=time.time() - start_time,
            ink_color=ink_color.value if ink_color else "unknown",
            output_path=output_key if success else "not_stored",
            original_shape=original_shape,
            processed_shape=processed_image.shape,
            status="success"
        )
        
        # Store in Redis or memory
        if hasattr(redis_client, 'setex'):
            redis_client.setex(
                f"preprocessing:{document_id}", 
                3600,  # 1 hour expiration
                json.dumps(result.__dict__)
            )
        else:
            redis_client.setex(f"preprocessing:{document_id}", 3600, json.dumps(result.__dict__))
        
    except Exception as e:
        logger.error(f"Processing failed for {document_id}: {e}")
        error_result = {
            "document_id": document_id,
            "status": "error",
            "error_message": str(e),
            "timestamp": time.time()
        }
        if hasattr(redis_client, 'setex'):
            redis_client.setex(f"preprocessing:{document_id}", 3600, json.dumps(error_result))
            redis_client.lpush("preprocessing:errors", json.dumps(error_result))
        else:
            redis_client.setex(f"preprocessing:{document_id}", 3600, json.dumps(error_result))
            redis_client.lpush("preprocessing:errors", json.dumps(error_result))

@router.post("/segment")
async def segment(
    file: UploadFile = File(...),
    engine: str = Query("mmocr", enum=["mmocr", "kraken"])
):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    binv = preprocess_safe(bgr)  # safe baseline
    if engine == "mmocr":
        lines = lines_mmocr.detect_lines(bgr)   # most detectors expect BGR/RGB; use original
    else:
        lines = lines_kraken.detect_lines(bgr)

    # return metadata only; client can crop locally if needed
    return {"ok": True, "n_lines": len(lines), "lines": lines}








# # src/api/endpoints.py
# import numpy as np
# import cv2
# from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse, Response
# from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
# try:
#     from healthcheck import HealthCheck
# except ImportError:
#     # Fallback for older version
#     from healthcheck import HealthCheck as HealthCheck
# import redis
# import json
# import time
# import logging
# from typing import Optional

# from src.preprocessing.core import DocumentPreprocessor
# from src.preprocessing.models import PreprocessingConfig, ProcessingResult, PreprocessingError
# from src.utils.storage import StorageManager
# from config.settings import settings

# # Initialize application
# app = FastAPI(title="Document Preprocessing Service", version="1.0.0")

# # Initialize components
# redis_client = redis.Redis(
#     host=settings.redis_host, 
#     port=settings.redis_port, 
#     db=settings.redis_db
# )

# preprocessing_config = PreprocessingConfig(
#     deskew_enabled=settings.deskew_enabled,
#     line_removal_enabled=settings.line_removal_enabled
# )

# preprocessor = DocumentPreprocessor(preprocessing_config)
# storage_manager = StorageManager(
#     endpoint_url=settings.s3_endpoint,
#     bucket_name=settings.s3_bucket,
#     aws_access_key_id=settings.s3_access_key,
#     aws_secret_access_key=settings.s3_secret_key
# )

# # Health check
# health = HealthCheck()

# @app.get("/")
# async def root():
#     return {"message": "Document Preprocessing Service", "status": "healthy"}
    
# @app.get("/health")
# async def health_check():
#     """Simple health check endpoint without external dependencies"""
#     try:
#         # Test basic functionality
#         test_image = np.ones((10, 10), dtype=np.uint8) * 255
#         _ = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR) if len(test_image.shape) == 2 else test_image
        
#         # Test Redis connection if available
#         if redis_client:
#             redis_client.ping()
            
#         return {
#             "status": "healthy",
#             "timestamp": time.time(),
#             "services": {
#                 "redis": "connected" if redis_client else "not_configured",
#                 "opencv": "working",
#                 "numpy": "working"
#             }
#         }
#     except Exception as e:
#         return {"status": "unhealthy", "error": str(e)}, 500
# # @app.get("/health")
# # async def health_check():
# #     return JSONResponse(content=health.run())

# @app.get("/metrics")
# async def metrics():
#     return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# @app.post("/preprocess/{document_id}")
# async def preprocess_document(
#     document_id: str,
#     background_tasks: BackgroundTasks,
#     file: Optional[UploadFile] = File(None),
#     image_url: Optional[str] = None
# ):
#     try:
#         # Get image data
#         if file:
#             image_data = await file.read()
#         elif image_url:
#             import requests
#             response = requests.get(image_url)
#             response.raise_for_status()
#             image_data = response.content
#         else:
#             raise HTTPException(status_code=400, detail="Either file or image_url must be provided")
        
#         # Process in background
#         background_tasks.add_task(process_image_task, document_id, image_data)
        
#         return {"message": "Processing started", "document_id": document_id}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/status/{document_id}")
# async def get_processing_status(document_id: str):
#     try:
#         result_data = redis_client.get(f"preprocessing:{document_id}")
#         if not result_data:
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         result = json.loads(result_data)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# async def process_image_task(document_id: str, image_data: bytes):
#     """Background task to process an image"""
#     start_time = time.time()
    
#     try:
#         from PIL import Image
#         import io
#         import numpy as np
        
#         # Load image
#         image = Image.open(io.BytesIO(image_data))
#         image_array = np.array(image)
#         original_shape = image_array.shape
        
#         # Process image
#         processed_image = preprocessor.preprocess_image(image_array)
#         ink_color = preprocessor.detect_ink_color(image_array)
        
#         # Save to storage
#         output_key = f"processed/{document_id}.png"
#         success = storage_manager.upload_image(processed_image, output_key)
        
#         if not success:
#             raise PreprocessingError("Failed to upload processed image")
        
#         # Store result
#         result = ProcessingResult(
#             document_id=document_id,
#             processing_time=time.time() - start_time,
#             ink_color=ink_color,
#             output_path=output_key,
#             original_shape=original_shape,
#             processed_shape=processed_image.shape
#         )
        
#         redis_client.setex(
#             f"preprocessing:{document_id}", 
#             3600,  # 1 hour expiration
#             json.dumps(result.__dict__)
#         )
        
#     except Exception as e:
#         error_result = {
#             "document_id": document_id,
#             "status": "error",
#             "error_message": str(e),
#             "timestamp": time.time()
#         }
#         redis_client.setex(f"preprocessing:{document_id}", 3600, json.dumps(error_result))
#         redis_client.lpush("preprocessing:errors", json.dumps(error_result))