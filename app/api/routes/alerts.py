"""
API routes for alert management.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.core.alerts import AlertManager
from app.models.schemas import (
    AlertResponse,
    AlertEvent,
    AlertAcknowledgeRequest,
    SystemStatus,
)
from app.models.database import Alert, Person

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

# Global alert manager instance
alert_manager = AlertManager()


@router.get("/active", response_model=AlertResponse)
async def get_active_alerts(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Get all unacknowledged alerts.

    Args:
        limit: Maximum number of alerts to return (default: 50)
        db: Database session

    Returns:
        AlertResponse with list of active alerts
    """
    try:
        alerts = alert_manager.get_active_alerts(db, limit=limit)

        alert_events = [
            AlertEvent(
                id=alert.id,
                timestamp=alert.timestamp,
                event_type=alert.event_type,
                person_id=alert.person_id,
                person_name=alert.person_name,
                confidence=alert.confidence,
                num_faces=alert.num_faces,
                snapshot_path=alert.snapshot_path,
                acknowledged=alert.acknowledged,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at,
            )
            for alert in alerts
        ]

        return AlertResponse(
            success=True,
            total=len(alert_events),
            alerts=alert_events
        )

    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent", response_model=AlertResponse)
async def get_recent_alerts(
    hours: int = Query(24, ge=1, le=168),  # Max 7 days
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Get alerts from the last N hours.

    Args:
        hours: Number of hours to look back (default: 24)
        limit: Maximum number of alerts to return (default: 100)
        db: Database session

    Returns:
        AlertResponse with list of recent alerts
    """
    try:
        alerts = alert_manager.get_recent_alerts(db, hours=hours, limit=limit)

        alert_events = [
            AlertEvent(
                id=alert.id,
                timestamp=alert.timestamp,
                event_type=alert.event_type,
                person_id=alert.person_id,
                person_name=alert.person_name,
                confidence=alert.confidence,
                num_faces=alert.num_faces,
                snapshot_path=alert.snapshot_path,
                acknowledged=alert.acknowledged,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at,
            )
            for alert in alerts
        ]

        return AlertResponse(
            success=True,
            total=len(alert_events),
            alerts=alert_events
        )

    except Exception as e:
        logger.error(f"Failed to get recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acknowledge")
async def acknowledge_alert(
    request: AlertAcknowledgeRequest,
    db: Session = Depends(get_db)
):
    """
    Acknowledge an alert.

    Args:
        request: Acknowledge request with alert_id and acknowledged_by
        db: Database session

    Returns:
        Success message with updated alert
    """
    try:
        alert = alert_manager.acknowledge_alert(
            db,
            alert_id=request.alert_id,
            acknowledged_by=request.acknowledged_by,
            notes=request.notes
        )

        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {request.alert_id} not found")

        return {
            "success": True,
            "message": f"Alert {request.alert_id} acknowledged",
            "alert": AlertEvent(
                id=alert.id,
                timestamp=alert.timestamp,
                event_type=alert.event_type,
                person_id=alert.person_id,
                person_name=alert.person_name,
                confidence=alert.confidence,
                num_faces=alert.num_faces,
                snapshot_path=alert.snapshot_path,
                acknowledged=alert.acknowledged,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at,
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_alert_statistics(
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    Get alert statistics for the last N hours.

    Args:
        hours: Number of hours to analyze (default: 24)
        db: Database session

    Returns:
        Alert statistics
    """
    try:
        stats = alert_manager.get_alert_statistics(db, hours=hours)

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        logger.error(f"Failed to get alert statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete an alert (admin only - for now no auth check).

    Args:
        alert_id: ID of alert to delete
        db: Database session

    Returns:
        Success message
    """
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()

        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        db.delete(alert)
        db.commit()

        logger.info(f"Alert {alert_id} deleted")

        return {
            "success": True,
            "message": f"Alert {alert_id} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete alert: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_alert_config():
    """
    Get current alert configuration.

    Returns:
        Alert configuration
    """
    return {
        "success": True,
        "config": alert_manager.config
    }


@router.put("/config")
async def update_alert_config(config: dict):
    """
    Update alert configuration (placeholder - would persist to database).

    Args:
        config: New configuration values

    Returns:
        Success message with updated config
    """
    try:
        # Update configuration
        for key, value in config.items():
            if key in alert_manager.config:
                alert_manager.config[key] = value
                logger.info(f"Alert config updated: {key} = {value}")

        return {
            "success": True,
            "message": "Alert configuration updated",
            "config": alert_manager.config
        }

    except Exception as e:
        logger.error(f"Failed to update alert config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
