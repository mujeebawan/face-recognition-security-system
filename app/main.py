"""
Face Recognition Security System - Main Application
FastAPI server for face detection and recognition.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Face Recognition Security System",
    description="AI-powered face recognition system using Hikvision IP camera on Jetson AGX Orin",
    version="0.1.0",
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("=" * 60)
    logger.info("Face Recognition Security System Starting...")
    logger.info("=" * 60)
    logger.info(f"Camera IP: {settings.camera_ip}")
    logger.info(f"Database: {settings.database_url}")
    logger.info(f"GPU Enabled: {settings.enable_gpu}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Face Recognition System...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition Security System API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "camera_configured": bool(settings.camera_ip),
        "database": settings.database_url.split("///")[-1] if "sqlite" in settings.database_url else "configured",
        "gpu_enabled": settings.enable_gpu
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
