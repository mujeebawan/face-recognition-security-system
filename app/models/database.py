"""
Database models for face recognition system.
SQLAlchemy ORM models for SQLite/PostgreSQL compatibility.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, LargeBinary, Float
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
