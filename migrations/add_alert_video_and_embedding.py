"""
Database migration: Add video_path and matched_embedding_id to alerts table

This migration adds support for:
1. Video recording of alerts (video_path column)
2. Tracking which embedding (original/augmented) matched best (matched_embedding_id column)

Run this migration once to update the database schema.
"""

import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "face_recognition.db"


def migrate():
    """Add new columns to alerts table"""
    if not os.path.exists(DB_PATH):
        logger.error(f"Database not found at {DB_PATH}")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check if columns already exist
        cursor.execute("PRAGMA table_info(alerts)")
        columns = [col[1] for col in cursor.fetchall()]

        # Add matched_embedding_id if it doesn't exist
        if 'matched_embedding_id' not in columns:
            logger.info("Adding matched_embedding_id column to alerts table...")
            cursor.execute("""
                ALTER TABLE alerts
                ADD COLUMN matched_embedding_id INTEGER
                REFERENCES face_embeddings(id)
            """)
            logger.info("✓ matched_embedding_id column added")
        else:
            logger.info("✓ matched_embedding_id column already exists")

        # Add video_path if it doesn't exist
        if 'video_path' not in columns:
            logger.info("Adding video_path column to alerts table...")
            cursor.execute("""
                ALTER TABLE alerts
                ADD COLUMN video_path VARCHAR(500)
            """)
            logger.info("✓ video_path column added")
        else:
            logger.info("✓ video_path column already exists")

        conn.commit()
        conn.close()

        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Alert Video & Embedding Migration")
    print("=" * 60)
    print()
    migrate()
