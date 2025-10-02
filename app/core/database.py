"""
Database connection and session management.
Handles SQLite/PostgreSQL connections via SQLAlchemy.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.models.database import Base
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=settings.debug  # Log SQL queries in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database - create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
        return False


def get_db() -> Session:
    """
    Dependency function to get database session.
    Use with FastAPI Depends().
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """Check if database connection is working"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False
