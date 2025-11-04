"""
Create default admin user for the system.

Default credentials:
- Username: admin
- Password: admin123
- Role: admin

⚠️ IMPORTANT: Change the password immediately after first login!
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import User
from app.core.auth import hash_password
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_default_admin():
    """Create default admin user if it doesn't exist."""
    logger.info("=" * 60)
    logger.info("Creating Default Admin User")
    logger.info("=" * 60)

    # Connect to database
    engine = create_engine(settings.database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Check if admin user already exists
        existing_admin = session.query(User).filter(User.username == "admin").first()

        if existing_admin:
            logger.warning("⚠️  Admin user already exists!")
            logger.info(f"   Username: {existing_admin.username}")
            logger.info(f"   Role: {existing_admin.role}")
            logger.info(f"   Created: {existing_admin.created_at}")
            logger.info("\nSkipping user creation.")
            return False

        # Create default admin user
        logger.info("Creating admin user...")
        logger.info(f"   Username: admin")
        logger.info(f"   Password: admin123")
        logger.info(f"   Role: admin")

        admin_user = User(
            username="admin",
            email="admin@facerecognition.local",
            full_name="System Administrator",
            password_hash=hash_password("admin123"),
            role="admin",
            is_active=True
        )

        session.add(admin_user)
        session.commit()

        logger.info("\n✅ Default admin user created successfully!")
        logger.info("\n" + "=" * 60)
        logger.info("⚠️  SECURITY NOTICE")
        logger.info("=" * 60)
        logger.info("\nDefault credentials:")
        logger.info("  Username: admin")
        logger.info("  Password: admin123")
        logger.info("\n⚠️  CHANGE THIS PASSWORD IMMEDIATELY after first login!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        session.rollback()
        logger.error(f"❌ Failed to create admin user: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        session.close()


if __name__ == "__main__":
    try:
        success = create_default_admin()

        if success:
            logger.info("\n✅ Setup completed successfully!")
            logger.info("\nYou can now:")
            logger.info("  1. Start the server: ./start_server.sh")
            logger.info("  2. Visit: http://localhost:8000/docs")
            logger.info("  3. Test login at POST /api/auth/login")
            logger.info("     with username='admin' and password='admin123'")
            logger.info("\nNext: Implement frontend login page (Phase 3)\n")
            sys.exit(0)
        else:
            logger.info("\n⚠️  Admin user already exists. Nothing to do.\n")
            sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
