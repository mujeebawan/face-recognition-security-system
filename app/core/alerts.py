"""
Alert system for face recognition events.
Handles unknown person detection, alerts, and notifications.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import cv2
import requests
from sqlalchemy.orm import Session

from app.models.database import Alert, Person
from app.config import settings

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alerts for recognition events"""

    def __init__(self):
        """Initialize alert manager"""
        self.config = self._load_config()
        self.last_alert_times: Dict[str, datetime] = {}  # Cooldown tracking
        self.snapshots_dir = Path("data/alert_snapshots")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Alert Manager initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration from settings or database"""
        # Default configuration
        config = {
            "enabled": True,
            "alert_on_unknown": True,
            "alert_on_known": False,
            "min_confidence_unknown": 0.5,
            "cooldown_seconds": 60,
            "webhook_url": os.getenv("ALERT_WEBHOOK_URL"),
            "email_recipients": os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",") if os.getenv("ALERT_EMAIL_RECIPIENTS") else [],
            "save_snapshot": True,
        }
        return config

    def should_alert(self, event_type: str, person_id: Optional[int] = None) -> bool:
        """
        Check if an alert should be triggered based on configuration and cooldown.

        Args:
            event_type: Type of event ('unknown_person', 'known_person', 'multiple_unknown')
            person_id: ID of recognized person (None for unknown)

        Returns:
            bool: True if alert should be triggered
        """
        if not self.config["enabled"]:
            return False

        # Check if this type of event should trigger an alert
        if event_type == "unknown_person" and not self.config["alert_on_unknown"]:
            return False

        if event_type == "known_person" and not self.config["alert_on_known"]:
            return False

        # Check cooldown
        cooldown_key = f"{event_type}_{person_id}" if person_id else event_type
        last_alert_time = self.last_alert_times.get(cooldown_key)

        if last_alert_time:
            time_since_last = (datetime.now() - last_alert_time).total_seconds()
            if time_since_last < self.config["cooldown_seconds"]:
                logger.debug(f"Alert cooldown active for {cooldown_key}: {time_since_last:.1f}s < {self.config['cooldown_seconds']}s")
                return False

        # Update last alert time
        self.last_alert_times[cooldown_key] = datetime.now()
        return True

    def save_snapshot(self, frame, alert_id: int) -> Optional[str]:
        """
        Save frame snapshot for alert.

        Args:
            frame: OpenCV frame (numpy array)
            alert_id: Alert ID for filename

        Returns:
            str: Path to saved snapshot, or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alert_{alert_id}_{timestamp}.jpg"
            filepath = self.snapshots_dir / filename

            cv2.imwrite(str(filepath), frame)
            logger.info(f"Snapshot saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def create_alert(
        self,
        db: Session,
        event_type: str,
        person_id: Optional[int] = None,
        person_name: Optional[str] = None,
        confidence: Optional[float] = None,
        num_faces: int = 1,
        frame = None,
    ) -> Optional[Alert]:
        """
        Create and save alert to database.

        Args:
            db: Database session
            event_type: Type of event
            person_id: ID of recognized person
            person_name: Name of recognized person
            confidence: Recognition confidence
            num_faces: Number of faces detected
            frame: OpenCV frame for snapshot

        Returns:
            Alert: Created alert object, or None if alert shouldn't be triggered
        """
        # Check if alert should be triggered
        if not self.should_alert(event_type, person_id):
            return None

        try:
            # Create alert record
            alert = Alert(
                event_type=event_type,
                person_id=person_id,
                person_name=person_name,
                confidence=confidence,
                num_faces=num_faces,
                acknowledged=False,
            )

            # Add to database to get ID
            db.add(alert)
            db.flush()  # Get ID without committing

            # Save snapshot if configured and frame provided
            if self.config["save_snapshot"] and frame is not None:
                snapshot_path = self.save_snapshot(frame, alert.id)
                alert.snapshot_path = snapshot_path

            db.commit()
            db.refresh(alert)

            logger.info(f"Alert created: {alert}")

            # Send notifications
            self._send_notifications(alert)

            return alert

        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            db.rollback()
            return None

    def _send_notifications(self, alert: Alert):
        """
        Send alert notifications via configured channels.

        Args:
            alert: Alert object
        """
        # Webhook notification
        if self.config["webhook_url"]:
            self._send_webhook(alert)

        # Email notification (placeholder - would need email service)
        if self.config["email_recipients"]:
            self._send_email(alert)

    def _send_webhook(self, alert: Alert):
        """
        Send alert to webhook URL.

        Args:
            alert: Alert object
        """
        try:
            payload = {
                "alert_id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "event_type": alert.event_type,
                "person_id": alert.person_id,
                "person_name": alert.person_name,
                "confidence": alert.confidence,
                "num_faces": alert.num_faces,
                "snapshot_path": alert.snapshot_path,
            }

            response = requests.post(
                self.config["webhook_url"],
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"Webhook sent successfully for alert {alert.id}")
            else:
                logger.warning(f"Webhook failed with status {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    def _send_email(self, alert: Alert):
        """
        Send alert via email (placeholder).

        Args:
            alert: Alert object
        """
        # Placeholder for email functionality
        # Would integrate with SendGrid, AWS SES, or SMTP
        logger.info(f"Email notification for alert {alert.id} (not implemented)")

    def get_active_alerts(self, db: Session, limit: int = 50) -> List[Alert]:
        """
        Get unacknowledged alerts.

        Args:
            db: Database session
            limit: Maximum number of alerts to return

        Returns:
            List of unacknowledged alerts
        """
        alerts = (
            db.query(Alert)
            .filter(Alert.acknowledged == False)
            .order_by(Alert.timestamp.desc())
            .limit(limit)
            .all()
        )
        return alerts

    def get_recent_alerts(self, db: Session, hours: int = 24, limit: int = 100) -> List[Alert]:
        """
        Get alerts from the last N hours.

        Args:
            db: Database session
            hours: Number of hours to look back
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        alerts = (
            db.query(Alert)
            .filter(Alert.timestamp >= cutoff_time)
            .order_by(Alert.timestamp.desc())
            .limit(limit)
            .all()
        )
        return alerts

    def acknowledge_alert(
        self,
        db: Session,
        alert_id: int,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> Optional[Alert]:
        """
        Acknowledge an alert.

        Args:
            db: Database session
            alert_id: Alert ID to acknowledge
            acknowledged_by: Username who acknowledged
            notes: Optional notes

        Returns:
            Updated alert object, or None if not found
        """
        try:
            alert = db.query(Alert).filter(Alert.id == alert_id).first()

            if not alert:
                logger.warning(f"Alert {alert_id} not found")
                return None

            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            if notes:
                alert.notes = notes

            db.commit()
            db.refresh(alert)

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return alert

        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            db.rollback()
            return None

    def get_alert_statistics(self, db: Session, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert statistics for the last N hours.

        Args:
            db: Database session
            hours: Number of hours to analyze

        Returns:
            Dictionary with alert statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Total alerts
        total = db.query(Alert).filter(Alert.timestamp >= cutoff_time).count()

        # Unacknowledged alerts
        unacknowledged = db.query(Alert).filter(
            Alert.timestamp >= cutoff_time,
            Alert.acknowledged == False
        ).count()

        # Unknown person alerts
        unknown = db.query(Alert).filter(
            Alert.timestamp >= cutoff_time,
            Alert.event_type == 'unknown_person'
        ).count()

        # Known person alerts
        known = db.query(Alert).filter(
            Alert.timestamp >= cutoff_time,
            Alert.event_type == 'known_person'
        ).count()

        return {
            "period_hours": hours,
            "total_alerts": total,
            "unacknowledged": unacknowledged,
            "unknown_persons": unknown,
            "known_persons": known,
            "acknowledged": total - unacknowledged,
        }
