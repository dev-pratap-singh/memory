"""
Database Models
SQLAlchemy models for PostgreSQL with pgvector support
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    ARRAY,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Conversation(Base):
    """Conversation storage with vector embeddings"""

    __tablename__ = "conversations"

    conversation_id = Column(String(100), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = Column(JSON, nullable=False)
    summary = Column(Text)
    importance_score = Column(Float, default=0.5, index=True)
    topics = Column(ARRAY(String))
    embedding = Column(Vector(384))  # Dimension for all-MiniLM-L6-v2
    user_id = Column(String(100), index=True)
    extra_metadata = Column(JSON)
    is_compressed = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False, index=True)

    # Relationships
    facts = relationship(
        "MemoryFact", back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Conversation(id={self.conversation_id}, created_at={self.created_at})>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "messages": self.messages,
            "summary": self.summary,
            "importance_score": self.importance_score,
            "topics": self.topics,
            "user_id": self.user_id,
            "extra_metadata": self.extra_metadata,
            "is_compressed": self.is_compressed,
            "is_archived": self.is_archived,
        }


class MemoryFact(Base):
    """Extracted facts with embeddings"""

    __tablename__ = "memory_facts"

    fact_id = Column(String(100), primary_key=True, index=True)
    conversation_id = Column(
        String(100), ForeignKey("conversations.conversation_id", ondelete="CASCADE")
    )
    fact = Column(Text, nullable=False)
    confidence = Column(Float, default=0.8)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    embedding = Column(Vector(384))
    tags = Column(ARRAY(String))
    source_type = Column(String(50), default="extraction")

    # Relationships
    conversation = relationship("Conversation", back_populates="facts")

    def __repr__(self):
        return f"<MemoryFact(id={self.fact_id}, confidence={self.confidence})>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "fact_id": self.fact_id,
            "conversation_id": self.conversation_id,
            "fact": self.fact,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "tags": self.tags,
            "source_type": self.source_type,
        }


class UserPreference(Base):
    """User preferences and instructions"""

    __tablename__ = "user_preferences"

    user_id = Column(String(100), primary_key=True, index=True)
    preferences = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id})>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TrainingHistory(Base):
    """Training history for the Small Language Model"""

    __tablename__ = "training_history"

    training_id = Column(String(100), primary_key=True, index=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    num_conversations = Column(Integer)
    adapter_path = Column(Text)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    metrics = Column(JSON)
    error_message = Column(Text)

    def __repr__(self):
        return f"<TrainingHistory(id={self.training_id}, status={self.status})>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "training_id": self.training_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "num_conversations": self.num_conversations,
            "adapter_path": self.adapter_path,
            "status": self.status,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }
