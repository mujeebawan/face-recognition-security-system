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

    # Relationship to embeddings
    embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Person(id={self.id}, name='{self.name}', cnic='{self.cnic}')>"


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
    acknowledged = Column(Boolean, default=False, nullable=False, index=True)
    acknowledged_by = Column(String(100), nullable=True)  # Username who acknowledged
    acknowledged_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)  # Additional notes

    def __repr__(self):
        return f"<Alert(id={self.id}, type='{self.event_type}', acknowledged={self.acknowledged})>"


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
