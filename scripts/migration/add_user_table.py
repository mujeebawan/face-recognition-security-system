"""
Database migration script to add users table.
Run this script to create the users table in the existing database.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, inspect
from app.models.database import Base, User
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_table_exists(engine, table_name):
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def migrate_database():
    """Add users table to existing database."""
    logger.info("=" * 60)
    logger.info("Database Migration: Adding Users Table")
    logger.info("=" * 60)

    # Connect to database
    engine = create_engine(settings.database_url)
    logger.info(f"Connected to database: {settings.database_url}")

    # Check if users table already exists
    if check_table_exists(engine, 'users'):
        logger.warning("⚠️  Users table already exists! Skipping migration.")
        logger.info("If you want to recreate it, manually drop the table first.")
        return False

    # Create users table only
    logger.info("Creating users table...")
    User.__table__.create(engine)

    # Verify table was created
    if check_table_exists(engine, 'users'):
        logger.info("✅ Users table created successfully!")

        # Show table structure
        inspector = inspect(engine)
        columns = inspector.get_columns('users')
        logger.info("\nTable structure:")
        for col in columns:
            logger.info(f"  - {col['name']}: {col['type']}")

        return True
    else:
        logger.error("❌ Failed to create users table!")
        return False


if __name__ == "__main__":
    try:
        success = migrate_database()

        if success:
            logger.info("\n" + "=" * 60)
            logger.info("✅ Migration completed successfully!")
            logger.info("=" * 60)
            logger.info("\nNext step: Run 'scripts/setup/create_default_admin.py'")
            logger.info("           to create the default admin user.\n")
            sys.exit(0)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("⚠️  Migration skipped (table already exists)")
            logger.info("=" * 60)
            sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
