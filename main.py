# main.py
import uvicorn
import logging
from config.settings import settings

if __name__ == "__main__":
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    uvicorn.run(
        "src.api.endpoints:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True  # Enable auto-reload in development
    )