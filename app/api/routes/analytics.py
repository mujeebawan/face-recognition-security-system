"""
API routes for analytics and reporting.
"""

import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Dict, Any

from app.core.database import get_db
from app.models.database import Alert, RecognitionLog, Person
from app.api.routes.auth import get_current_user, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary")
async def get_summary_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics for the dashboard.

    Args:
        days: Number of days to look back
        current_user: Current authenticated user
        db: Database session

    Returns:
        Summary statistics including alerts, recognition logs, persons
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Total alerts
        total_alerts = db.query(func.count(Alert.id)).scalar()
        recent_alerts = db.query(func.count(Alert.id)).filter(
            Alert.timestamp >= cutoff_date
        ).scalar()

        # Alerts by type
        known_alerts = db.query(func.count(Alert.id)).filter(
            and_(Alert.event_type == 'known_person', Alert.timestamp >= cutoff_date)
        ).scalar()

        unknown_alerts = db.query(func.count(Alert.id)).filter(
            and_(Alert.event_type == 'unknown_person', Alert.timestamp >= cutoff_date)
        ).scalar()

        # Recognition logs
        total_recognitions = db.query(func.count(RecognitionLog.id)).filter(
            RecognitionLog.timestamp >= cutoff_date
        ).scalar()

        successful_matches = db.query(func.count(RecognitionLog.id)).filter(
            and_(RecognitionLog.matched == 1, RecognitionLog.timestamp >= cutoff_date)
        ).scalar()

        # Success rate
        success_rate = (successful_matches / total_recognitions * 100) if total_recognitions > 0 else 0

        # Total enrolled persons
        total_persons = db.query(func.count(Person.id)).scalar()

        # Average confidence for successful matches
        avg_confidence = db.query(func.avg(RecognitionLog.confidence)).filter(
            and_(RecognitionLog.matched == 1, RecognitionLog.timestamp >= cutoff_date)
        ).scalar()

        return {
            "success": True,
            "period_days": days,
            "alerts": {
                "total": total_alerts,
                "recent": recent_alerts,
                "known_persons": known_alerts,
                "unknown_persons": unknown_alerts
            },
            "recognition": {
                "total_attempts": total_recognitions,
                "successful_matches": successful_matches,
                "success_rate": round(success_rate, 2),
                "average_confidence": round(avg_confidence, 4) if avg_confidence else 0
            },
            "persons": {
                "total_enrolled": total_persons
            }
        }

    except Exception as e:
        logger.error(f"Failed to get summary stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/timeseries")
async def get_alerts_timeseries(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    interval: str = Query("day", regex="^(hour|day|week)$", description="Time interval"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get time-series data for alerts.

    Args:
        days: Number of days to look back
        interval: Time interval (hour, day, week)
        current_user: Current authenticated user
        db: Database session

    Returns:
        Time-series data with counts by interval
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get all alerts in the time range
        alerts = db.query(Alert).filter(Alert.timestamp >= cutoff_date).all()

        # Group by interval
        time_series = {}

        for alert in alerts:
            if interval == "hour":
                key = alert.timestamp.strftime("%Y-%m-%d %H:00")
            elif interval == "day":
                key = alert.timestamp.strftime("%Y-%m-%d")
            else:  # week
                # Get week start (Monday)
                week_start = alert.timestamp - timedelta(days=alert.timestamp.weekday())
                key = week_start.strftime("%Y-%m-%d")

            if key not in time_series:
                time_series[key] = {
                    "timestamp": key,
                    "total": 0,
                    "known_person": 0,
                    "unknown_person": 0
                }

            time_series[key]["total"] += 1
            if alert.event_type == "known_person":
                time_series[key]["known_person"] += 1
            elif alert.event_type == "unknown_person":
                time_series[key]["unknown_person"] += 1

        # Convert to list and sort
        result = sorted(time_series.values(), key=lambda x: x["timestamp"])

        return {
            "success": True,
            "interval": interval,
            "period_days": days,
            "data": result
        }

    except Exception as e:
        logger.error(f"Failed to get alerts timeseries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recognition/timeseries")
async def get_recognition_timeseries(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    interval: str = Query("day", regex="^(hour|day|week)$", description="Time interval"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get time-series data for recognition attempts.

    Args:
        days: Number of days to look back
        interval: Time interval (hour, day, week)
        current_user: Current authenticated user
        db: Database session

    Returns:
        Time-series data with recognition success/failure counts
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get all recognition logs in the time range
        logs = db.query(RecognitionLog).filter(RecognitionLog.timestamp >= cutoff_date).all()

        # Group by interval
        time_series = {}

        for log in logs:
            if interval == "hour":
                key = log.timestamp.strftime("%Y-%m-%d %H:00")
            elif interval == "day":
                key = log.timestamp.strftime("%Y-%m-%d")
            else:  # week
                week_start = log.timestamp - timedelta(days=log.timestamp.weekday())
                key = week_start.strftime("%Y-%m-%d")

            if key not in time_series:
                time_series[key] = {
                    "timestamp": key,
                    "total": 0,
                    "matched": 0,
                    "unmatched": 0,
                    "avg_confidence": []
                }

            time_series[key]["total"] += 1
            if log.matched == 1:
                time_series[key]["matched"] += 1
                time_series[key]["avg_confidence"].append(log.confidence)
            else:
                time_series[key]["unmatched"] += 1

        # Calculate average confidence and success rate
        for data in time_series.values():
            if data["avg_confidence"]:
                data["avg_confidence"] = round(sum(data["avg_confidence"]) / len(data["avg_confidence"]), 4)
            else:
                data["avg_confidence"] = 0

            data["success_rate"] = round((data["matched"] / data["total"] * 100), 2) if data["total"] > 0 else 0

        # Convert to list and sort
        result = sorted(time_series.values(), key=lambda x: x["timestamp"])

        return {
            "success": True,
            "interval": interval,
            "period_days": days,
            "data": result
        }

    except Exception as e:
        logger.error(f"Failed to get recognition timeseries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persons/statistics")
async def get_person_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    limit: int = Query(10, ge=1, le=100, description="Top N persons"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get person-wise statistics.

    Args:
        days: Number of days to look back
        limit: Number of top persons to return
        current_user: Current authenticated user
        db: Database session

    Returns:
        Person-wise statistics (most detected, alert counts, etc.)
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get all persons
        persons = db.query(Person).all()

        person_stats = []

        for person in persons:
            # Recognition count
            recognition_count = db.query(func.count(RecognitionLog.id)).filter(
                and_(
                    RecognitionLog.person_id == person.id,
                    RecognitionLog.matched == 1,
                    RecognitionLog.timestamp >= cutoff_date
                )
            ).scalar()

            # Alert count
            alert_count = db.query(func.count(Alert.id)).filter(
                and_(
                    Alert.person_id == person.id,
                    Alert.timestamp >= cutoff_date
                )
            ).scalar()

            # Average confidence
            avg_confidence = db.query(func.avg(RecognitionLog.confidence)).filter(
                and_(
                    RecognitionLog.person_id == person.id,
                    RecognitionLog.matched == 1,
                    RecognitionLog.timestamp >= cutoff_date
                )
            ).scalar()

            # Last seen
            last_recognition = db.query(RecognitionLog).filter(
                and_(
                    RecognitionLog.person_id == person.id,
                    RecognitionLog.matched == 1
                )
            ).order_by(RecognitionLog.timestamp.desc()).first()

            person_stats.append({
                "person_id": person.id,
                "person_name": person.name,
                "person_cnic": person.cnic,
                "recognition_count": recognition_count,
                "alert_count": alert_count,
                "avg_confidence": round(avg_confidence, 4) if avg_confidence else 0,
                "last_seen": last_recognition.timestamp.isoformat() if last_recognition else None
            })

        # Sort by recognition count (most detected first)
        person_stats.sort(key=lambda x: x["recognition_count"], reverse=True)

        return {
            "success": True,
            "period_days": days,
            "total_persons": len(person_stats),
            "data": person_stats[:limit]
        }

    except Exception as e:
        logger.error(f"Failed to get person statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/distribution")
async def get_alert_distribution(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get alert distribution by type and hour of day.

    Args:
        days: Number of days to look back
        current_user: Current authenticated user
        db: Database session

    Returns:
        Alert distribution data
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        alerts = db.query(Alert).filter(Alert.timestamp >= cutoff_date).all()

        # Distribution by type
        type_distribution = {}

        # Distribution by hour of day (0-23)
        hour_distribution = {str(i): 0 for i in range(24)}

        for alert in alerts:
            # By type
            event_type = alert.event_type
            type_distribution[event_type] = type_distribution.get(event_type, 0) + 1

            # By hour
            hour = str(alert.timestamp.hour)
            hour_distribution[hour] += 1

        return {
            "success": True,
            "period_days": days,
            "by_type": type_distribution,
            "by_hour": hour_distribution
        }

    except Exception as e:
        logger.error(f"Failed to get alert distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))
