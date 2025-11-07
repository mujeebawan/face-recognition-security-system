"""
Face Recognition Security System - Main Application
FastAPI server for face detection and recognition.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import logging
import os
from app.config import settings
from app.api.routes import detection, recognition, alerts, websocket, auth, analytics
from app.api.routes import settings as settings_router

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

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Global exception handler
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )

# Include routers
app.include_router(detection.router)
app.include_router(recognition.router)
app.include_router(alerts.router)
app.include_router(analytics.router)
app.include_router(settings_router.router)
app.include_router(websocket.router)
app.include_router(auth.router)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global startup_time
    from datetime import datetime
    startup_time = datetime.now()

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
    """Root endpoint - redirects to login page"""
    return RedirectResponse(url="/login", status_code=302)


@app.get("/login")
async def login_page():
    """Serve login page"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "login.html")
    return FileResponse(html_path)


@app.get("/live")
async def live_viewer():
    """Serve live stream viewer HTML page"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "live_stream.html")
    return FileResponse(html_path)


@app.get("/dashboard")
async def dashboard():
    """Serve real-time dashboard with WebSocket alerts"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "dashboard.html")
    return FileResponse(html_path)


@app.get("/admin")
async def admin_panel():
    """Serve LEA admin panel for wanted persons management"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "admin.html")
    return FileResponse(html_path)


@app.get("/alerts")
async def alerts_management():
    """Serve alert management page"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "alerts.html")
    return FileResponse(html_path)


@app.get("/reports")
async def reports_analytics():
    """Serve reports and analytics page"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "reports.html")
    return FileResponse(html_path)


@app.get("/settings")
async def system_settings():
    """Serve system settings page"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "settings.html")
    return FileResponse(html_path)


@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint for production monitoring.
    Returns system status, database connectivity, and resource usage.
    """
    import psutil
    from datetime import datetime
    from app.core.database import get_db
    from sqlalchemy import text

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int((datetime.now() - startup_time).total_seconds()) if 'startup_time' in globals() else 0
    }

    try:
        # Check database connectivity
        db = next(get_db())
        db.execute(text("SELECT 1"))
        health_status["database"] = {
            "status": "connected",
            "url": settings.database_url.split("///")[-1] if "sqlite" in settings.database_url else "configured"
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["database"] = {
            "status": "error",
            "error": str(e)
        }

    # System resources
    try:
        memory = psutil.virtual_memory()
        health_status["resources"] = {
            "memory_used_mb": int(memory.used / 1024 / 1024),
            "memory_percent": memory.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "disk_percent": psutil.disk_usage('/').percent
        }
    except Exception as e:
        logger.warning(f"Failed to get system resources: {e}")

    # Configuration status
    health_status["config"] = {
        "camera_configured": bool(settings.camera_ip),
        "gpu_enabled": settings.enable_gpu,
        "debug_mode": settings.debug
    }

    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
