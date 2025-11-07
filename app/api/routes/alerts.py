"""
API routes for alert management.
"""

import logging
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
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
from app.models.database import Alert, Person, User
from app.api.routes.auth import get_current_user

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

        alert_events = []
        for alert in alerts:
            # Get original person image if person_id exists
            original_image_url = None
            if alert.person_id:
                person = db.query(Person).filter(Person.id == alert.person_id).first()
                if person:
                    original_image_url = f"/api/alerts/original-image/{alert.id}"

            alert_events.append(
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
                    original_image_url=original_image_url,
                    # Guard verification fields
                    guard_verified=alert.guard_verified,
                    guard_action=alert.guard_action,
                    guard_verified_by=alert.guard_verified_by,
                    guard_verified_at=alert.guard_verified_at,
                    action_notes=alert.action_notes,
                    # Watchlist fields
                    threat_level=alert.threat_level,
                    watchlist_status=alert.watchlist_status,
                )
            )

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

        alert_events = []
        for alert in alerts:
            # Get original person image if person_id exists
            original_image_url = None
            if alert.person_id:
                person = db.query(Person).filter(Person.id == alert.person_id).first()
                if person:
                    original_image_url = f"/api/alerts/original-image/{alert.id}"

            alert_events.append(
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
                    original_image_url=original_image_url,
                    # Guard verification fields
                    guard_verified=alert.guard_verified,
                    guard_action=alert.guard_action,
                    guard_verified_by=alert.guard_verified_by,
                    guard_verified_at=alert.guard_verified_at,
                    action_notes=alert.action_notes,
                    # Watchlist fields
                    threat_level=alert.threat_level,
                    watchlist_status=alert.watchlist_status,
                )
            )

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
                # Guard verification fields
                guard_verified=alert.guard_verified,
                guard_action=alert.guard_action,
                guard_verified_by=alert.guard_verified_by,
                guard_verified_at=alert.guard_verified_at,
                action_notes=alert.action_notes,
                # Watchlist fields
                threat_level=alert.threat_level,
                watchlist_status=alert.watchlist_status,
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/verify")
async def verify_alert(
    alert_id: int,
    guard_action: str,
    guard_username: str,
    action_notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Guard verification for an alert - marks alert with guard's decision.

    Args:
        alert_id: ID of alert to verify
        guard_action: Action taken ('confirmed', 'false_alarm', 'investigating', 'apprehended', 'escalated')
        guard_username: Username of guard
        action_notes: Optional notes about the action
        db: Database session

    Returns:
        Success message with updated alert
    """
    try:
        from datetime import datetime

        # Validate guard action
        valid_actions = ['confirmed', 'false_alarm', 'investigating', 'apprehended', 'escalated']
        if guard_action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid guard action. Must be one of: {', '.join(valid_actions)}"
            )

        # Get alert
        alert = db.query(Alert).filter(Alert.id == alert_id).first()

        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        # Update guard verification fields
        alert.guard_verified = True
        alert.guard_action = guard_action
        alert.guard_verified_by = guard_username
        alert.guard_verified_at = datetime.now()
        alert.action_notes = action_notes

        db.commit()
        db.refresh(alert)

        logger.info(f"Alert {alert_id} verified by {guard_username} with action: {guard_action}")

        return {
            "success": True,
            "message": f"Alert {alert_id} verified with action: {guard_action}",
            "alert": {
                "id": alert.id,
                "guard_verified": alert.guard_verified,
                "guard_action": alert.guard_action,
                "guard_verified_by": alert.guard_verified_by,
                "guard_verified_at": alert.guard_verified_at.isoformat() if alert.guard_verified_at else None,
                "action_notes": alert.action_notes,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify alert: {e}")
        db.rollback()
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


@router.get("/snapshot/{alert_id}")
async def get_alert_snapshot(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """
    Serve alert snapshot image (captured image) for dashboard display.
    No authentication required for real-time alert viewing.

    Args:
        alert_id: The alert ID
        db: Database session

    Returns:
        FileResponse with the snapshot image
    """
    try:
        # Get alert from database
        alert = db.query(Alert).filter(Alert.id == alert_id).first()

        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        if not alert.snapshot_path:
            raise HTTPException(status_code=404, detail="No snapshot available for this alert")

        # Check if file exists
        if not os.path.exists(alert.snapshot_path):
            logger.error(f"Snapshot file not found: {alert.snapshot_path}")
            raise HTTPException(status_code=404, detail="Snapshot file not found")

        # Return the image file with no-cache to prevent showing old snapshots
        return FileResponse(
            alert.snapshot_path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Disposition": f'inline; filename="alert_{alert_id}_snapshot.jpg"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving snapshot for alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/original-image/{alert_id}")
async def get_alert_original_image(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """
    Serve original enrolled person image for alert comparison on dashboard.
    No authentication required for real-time alert viewing.

    Args:
        alert_id: The alert ID
        db: Database session

    Returns:
        FileResponse with the original enrolled person image
    """
    try:
        # Get alert from database
        alert = db.query(Alert).filter(Alert.id == alert_id).first()

        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        if not alert.person_id:
            raise HTTPException(status_code=404, detail="No person associated with this alert (unknown person)")

        # Get person from database
        person = db.query(Person).filter(Person.id == alert.person_id).first()

        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        if not person.reference_image_path:
            raise HTTPException(status_code=404, detail="No reference image available for this person")

        # Check if file exists
        if not os.path.exists(person.reference_image_path):
            logger.error(f"Original image file not found: {person.reference_image_path}")
            raise HTTPException(status_code=404, detail="Original image file not found")

        # Return the image file with no-cache
        return FileResponse(
            person.reference_image_path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Disposition": f'inline; filename="alert_{alert_id}_original.jpg"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving original image for alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
