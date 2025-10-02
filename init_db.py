"""
Database initialization script.
Creates all tables in the SQLite database.
"""

import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.database import init_db, check_db_connection
from app.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 60)
    print("Face Recognition System - Database Initialization")
    print("=" * 60 + "\n")

    print(f"Database: {settings.database_url}")
    print(f"Type: {'SQLite' if 'sqlite' in settings.database_url else 'PostgreSQL'}")
    print()

    # Initialize database
    logger.info("Creating database tables...")
    if init_db():
        logger.info("✓ All tables created successfully")

        # Test connection
        logger.info("Testing database connection...")
        if check_db_connection():
            logger.info("✓ Database connection successful")
            print("\n" + "=" * 60)
            print("✓ Database initialization COMPLETED!")
            print("=" * 60 + "\n")
            return 0
        else:
            logger.error("✗ Database connection test failed")
            return 1
    else:
        logger.error("✗ Database initialization failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
