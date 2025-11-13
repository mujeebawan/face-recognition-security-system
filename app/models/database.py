"""
Database models for face recognition system.
SQLAlchemy ORM models for SQLite/PostgreSQL compatibility.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, LargeBinary, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Person(Base):
    """Person model - stores individual records"""

    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    cnic = Column(String(20), unique=True, nullable=False, index=True)  # National ID
    reference_image_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Watchlist fields for criminal/wanted person detection
    watchlist_status = Column(String(50), default='none', nullable=False, index=True)  # 'most_wanted', 'suspect', 'person_of_interest', 'banned', 'none'
    threat_level = Column(String(20), default='none', nullable=False, index=True)  # 'critical', 'high', 'medium', 'low', 'none'
    criminal_notes = Column(Text, nullable=True)  # Details about why on watchlist
    added_to_watchlist_at = Column(DateTime, nullable=True)  # When added to watchlist
    watchlist_expires_at = Column(DateTime, nullable=True)  # Optional expiry date

    # Relationship to embeddings
    embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Person(id={self.id}, name='{self.name}', cnic='{self.cnic}', watchlist='{self.watchlist_status}')>"


class FaceEmbedding(Base):
    """Face embedding model - stores face embeddings (512-D vectors)"""

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)  # Stored as blob (numpy array serialized)
    source = Column(String(50), nullable=False)  # 'original', 'augmented', 'diffusion'
    confidence = Column(Float, nullable=True)  # Detection confidence (0-1)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship to person
    person = relationship("Person", back_populates="embeddings")

    def __repr__(self):
        return f"<FaceEmbedding(id={self.id}, person_id={self.person_id}, source='{self.source}')>"


class RecognitionLog(Base):
    """Recognition log - audit trail of all recognition attempts"""

    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True, index=True)  # Null if no match
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    confidence = Column(Float, nullable=False)  # Match confidence (0-1)
    matched = Column(Integer, nullable=False, default=0)  # 1 if matched, 0 if not
    image_path = Column(String(500), nullable=True)  # Path to captured frame
    camera_source = Column(String(100), nullable=True)  # Camera identifier

    def __repr__(self):
        return f"<RecognitionLog(id={self.id}, person_id={self.person_id}, matched={self.matched}, confidence={self.confidence:.2f})>"


class Alert(Base):
    """Alert model - stores security alerts for unknown/known persons"""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # 'unknown_person', 'known_person', 'multiple_unknown'
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True, index=True)  # Null for unknown persons
    person_name = Column(String(255), nullable=True)  # Cached name for quick display
    confidence = Column(Float, nullable=True)  # Recognition confidence
    num_faces = Column(Integer, default=1)  # Number of faces in frame
    snapshot_path = Column(String(500), nullable=True)  # Path to snapshot image
    matched_embedding_id = Column(Integer, ForeignKey("face_embeddings.id"), nullable=True)  # Which embedding matched best
    video_path = Column(String(500), nullable=True)  # Path to recorded video clip

    # Original acknowledgment fields (for admin review)
    acknowledged = Column(Boolean, default=False, nullable=False, index=True)
    acknowledged_by = Column(String(100), nullable=True)  # Username who acknowledged
    acknowledged_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)  # Additional notes

    # Guard verification fields (for real-time response)
    guard_verified = Column(Boolean, default=False, nullable=False, index=True)  # Did guard verify match?
    guard_action = Column(String(50), nullable=True, index=True)  # 'confirmed', 'false_alarm', 'investigating', 'apprehended', 'escalated'
    guard_verified_by = Column(String(100), nullable=True)  # Guard username
    guard_verified_at = Column(DateTime, nullable=True)  # When guard took action
    action_notes = Column(Text, nullable=True)  # Guard's notes on action taken

    # Cached threat level for quick filtering (denormalized from Person)
    threat_level = Column(String(20), nullable=True, index=True)  # Cached from person.threat_level
    watchlist_status = Column(String(50), nullable=True, index=True)  # Cached from person.watchlist_status

    def __repr__(self):
        return f"<Alert(id={self.id}, type='{self.event_type}', threat='{self.threat_level}', verified={self.guard_verified})>"


class SystemConfiguration(Base):
    """System configuration - stores runtime configuration"""

    __tablename__ = "system_configuration"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    config_key = Column(String(100), unique=True, nullable=False, index=True)
    config_value = Column(Text, nullable=False)  # Stored as JSON string
    data_type = Column(String(50), nullable=False)  # 'float', 'int', 'bool', 'string', 'json'
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<SystemConfiguration(key='{self.config_key}', value='{self.config_value}')>"


class User(Base):
    """User model - stores admin/operator accounts for system access"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=False)  # bcrypt hashed password
    full_name = Column(String(255), nullable=True)
    role = Column(String(20), nullable=False, default='operator')  # 'admin', 'operator', 'viewer'
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"
