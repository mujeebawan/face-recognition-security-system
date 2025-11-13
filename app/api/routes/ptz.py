"""
PTZ Control API Routes
API endpoints for camera zoom, pan, tilt control
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import logging

from app.core.ptz_control import get_ptz_controller

router = APIRouter(prefix="/api/ptz", tags=["ptz-control"])
logger = logging.getLogger(__name__)


class PTZResponse(BaseModel):
    """Response model for PTZ operations"""
    success: bool
    message: str


class PresetRequest(BaseModel):
    """Request model for preset operations"""
    preset_id: int = Field(..., ge=1, le=255, description="Preset ID (1-255)")
    name: Optional[str] = Field(None, description="Optional preset name")


@router.post("/zoom/in", response_model=PTZResponse)
async def zoom_in(
    speed: int = Query(50, ge=1, le=100, description="Zoom speed (1-100)")
):
    """
    Zoom in (continuous)
    Camera will zoom in until zoom/stop is called
    """
    try:
        logger.info(f"📹 PTZ Zoom IN requested, speed={speed}")
        ptz = get_ptz_controller()
        success, message = ptz.zoom_in(speed)

        if success:
            logger.info(f"✅ Zoom IN successful: {message}")
        else:
            logger.warning(f"⚠️ Zoom IN failed: {message}")

        return PTZResponse(success=success, message=message)

    except Exception as e:
        logger.error(f"❌ Zoom in error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/zoom/out", response_model=PTZResponse)
async def zoom_out(
    speed: int = Query(50, ge=1, le=100, description="Zoom speed (1-100)")
):
    """
    Zoom out (continuous)
    Camera will zoom out until zoom/stop is called
    """
    try:
        logger.info(f"📹 PTZ Zoom OUT requested, speed={speed}")
        ptz = get_ptz_controller()
        success, message = ptz.zoom_out(speed)

        if success:
            logger.info(f"✅ Zoom OUT successful: {message}")
        else:
            logger.warning(f"⚠️ Zoom OUT failed: {message}")

        return PTZResponse(success=success, message=message)

    except Exception as e:
        logger.error(f"❌ Zoom out error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/zoom/stop", response_model=PTZResponse)
async def zoom_stop():
    """
    Stop zoom movement
    """
    try:
        logger.info(f"📹 PTZ Zoom STOP requested")
        ptz = get_ptz_controller()
        success, message = ptz.zoom_stop()

        if success:
            logger.info(f"✅ Zoom STOP successful: {message}")
        else:
            logger.warning(f"⚠️ Zoom STOP failed: {message}")

        return PTZResponse(success=success, message=message)

    except Exception as e:
        logger.error(f"❌ Zoom stop error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pan/left", response_model=PTZResponse)
async def pan_left(
    speed: int = Query(50, ge=1, le=100, description="Pan speed (1-100)")
):
    """Pan camera left"""
    try:
        ptz = get_ptz_controller()
        success, message = ptz.pan_left(speed)
        return PTZResponse(success=success, message=message)
    except Exception as e:
        logger.error(f"Pan left error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pan/right", response_model=PTZResponse)
async def pan_right(
    speed: int = Query(50, ge=1, le=100, description="Pan speed (1-100)")
):
    """Pan camera right"""
    try:
        ptz = get_ptz_controller()
        success, message = ptz.pan_right(speed)
        return PTZResponse(success=success, message=message)
    except Exception as e:
        logger.error(f"Pan right error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tilt/up", response_model=PTZResponse)
async def tilt_up(
    speed: int = Query(50, ge=1, le=100, description="Tilt speed (1-100)")
):
    """Tilt camera up"""
    try:
        ptz = get_ptz_controller()
        success, message = ptz.tilt_up(speed)
        return PTZResponse(success=success, message=message)
    except Exception as e:
        logger.error(f"Tilt up error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tilt/down", response_model=PTZResponse)
async def tilt_down(
    speed: int = Query(50, ge=1, le=100, description="Tilt speed (1-100)")
):
    """Tilt camera down"""
    try:
        ptz = get_ptz_controller()
        success, message = ptz.tilt_down(speed)
        return PTZResponse(success=success, message=message)
    except Exception as e:
        logger.error(f"Tilt down error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=PTZResponse)
async def stop_all():
    """
    Stop all PTZ movements (pan, tilt, zoom)
    """
    try:
        ptz = get_ptz_controller()
        success, message = ptz.stop_all()

        return PTZResponse(success=success, message=message)

    except Exception as e:
        logger.error(f"PTZ stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preset/goto", response_model=PTZResponse)
async def goto_preset(request: PresetRequest):
    """
    Move camera to a saved preset position
    """
    try:
        ptz = get_ptz_controller()
        success, message = ptz.goto_preset(request.preset_id)

        return PTZResponse(success=success, message=message)

    except Exception as e:
        logger.error(f"Goto preset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preset/save", response_model=PTZResponse)
async def save_preset(request: PresetRequest):
    """
    Save current camera position as preset
    """
    try:
        ptz = get_ptz_controller()
        success, message = ptz.save_preset(request.preset_id, request.name or "")

        return PTZResponse(success=success, message=message)

    except Exception as e:
        logger.error(f"Save preset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_ptz_status():
    """
    Get PTZ status including current zoom level
    """
    try:
        ptz = get_ptz_controller()
        status = ptz.get_ptz_status()

        if status:
            return status
        else:
            raise HTTPException(status_code=503, detail="Failed to get PTZ status")

    except Exception as e:
        logger.error(f"PTZ status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/zoom/level")
async def get_zoom_level():
    """
    Get current zoom level percentage
    """
    try:
        ptz = get_ptz_controller()
        return {
            "success": True,
            "zoom_level": ptz.get_zoom_level()
        }
    except Exception as e:
        logger.error(f"Get zoom level error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/zoom/level")
async def set_zoom_level(level: int = Query(..., ge=0, le=100, description="Zoom level (0-100%)")):
    """
    Set/reset zoom level percentage manually
    """
    try:
        ptz = get_ptz_controller()
        ptz.set_zoom_level(level)
        return {
            "success": True,
            "message": f"Zoom level set to {level}%",
            "zoom_level": ptz.get_zoom_level()
        }
    except Exception as e:
        logger.error(f"Set zoom level error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
