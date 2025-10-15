"""
Data Repository Layer
Provides clean interface for database operations using Repository pattern
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy import and_, desc, func, or_, text
from sqlalchemy.orm import Session

from src.database.models import Conversation, MemoryFact, TrainingHistory, UserPreference

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository for conversation operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, conversation: Conversation) -> Conversation:
        """Create a new conversation"""
        try:
            self.session.add(conversation)
            self.session.commit()
            self.session.refresh(conversation)
            logger.info(f"Created conversation: {conversation.conversation_id}")
            return conversation
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating conversation: {e}")
            raise

    def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return (
            self.session.query(Conversation)
            .filter(Conversation.conversation_id == conversation_id)
            .first()
        )

    def get_recent(
        self, limit: int = 10, user_id: Optional[str] = None
    ) -> List[Conversation]:
        """Get recent conversations"""
        query = self.session.query(Conversation).filter(
            Conversation.is_archived == False
        )

        if user_id:
            query = query.filter(Conversation.user_id == user_id)

        return query.order_by(desc(Conversation.created_at)).limit(limit).all()

    def get_important(
        self, threshold: float = 0.7, limit: int = 100
    ) -> List[Conversation]:
        """Get important conversations"""
        return (
            self.session.query(Conversation)
            .filter(
                and_(
                    Conversation.importance_score >= threshold,
                    Conversation.is_archived == False,
                )
            )
            .order_by(desc(Conversation.created_at))
            .limit(limit)
            .all()
        )

    def search_by_topic(self, topics: List[str], limit: int = 10) -> List[Conversation]:
        """Search conversations by topics"""
        return (
            self.session.query(Conversation)
            .filter(Conversation.topics.overlap(topics))
            .order_by(desc(Conversation.importance_score))
            .limit(limit)
            .all()
        )

    def search_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Conversation]:
        """Search conversations by date range"""
        return (
            self.session.query(Conversation)
            .filter(
                and_(
                    Conversation.created_at >= start_date,
                    Conversation.created_at <= end_date,
                )
            )
            .order_by(desc(Conversation.created_at))
            .all()
        )

    def semantic_search(
        self, query_embedding: np.ndarray, limit: int = 5
    ) -> List[Tuple[Conversation, float]]:
        """
        Semantic search using vector similarity

        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return

        Returns:
            List of (conversation, distance) tuples
        """
        try:
            # Convert numpy array to list for SQL
            embedding_list = query_embedding.tolist()

            # Use pgvector cosine distance operator
            query = text(
                """
                SELECT conversation_id,
                       1 - (embedding <=> :embedding) as similarity
                FROM conversations
                WHERE embedding IS NOT NULL
                  AND is_archived = FALSE
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """
            )

            result = self.session.execute(
                query, {"embedding": str(embedding_list), "limit": limit}
            )

            # Fetch conversations with similarity scores
            results = []
            for row in result:
                conv = self.get_by_id(row[0])
                if conv:
                    results.append((conv, row[1]))

            return results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def update(self, conversation: Conversation) -> Conversation:
        """Update conversation"""
        try:
            self.session.commit()
            self.session.refresh(conversation)
            logger.info(f"Updated conversation: {conversation.conversation_id}")
            return conversation
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating conversation: {e}")
            raise

    def mark_as_compressed(self, conversation_id: str) -> bool:
        """Mark conversation as compressed"""
        try:
            conv = self.get_by_id(conversation_id)
            if conv:
                conv.is_compressed = True
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error marking conversation as compressed: {e}")
            return False

    def mark_as_archived(self, conversation_id: str) -> bool:
        """Mark conversation as archived"""
        try:
            conv = self.get_by_id(conversation_id)
            if conv:
                conv.is_archived = True
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error marking conversation as archived: {e}")
            return False

    def delete(self, conversation_id: str) -> bool:
        """Delete conversation"""
        try:
            conv = self.get_by_id(conversation_id)
            if conv:
                self.session.delete(conv)
                self.session.commit()
                logger.info(f"Deleted conversation: {conversation_id}")
                return True
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting conversation: {e}")
            return False

    def count(self, user_id: Optional[str] = None) -> int:
        """Count conversations"""
        query = self.session.query(func.count(Conversation.conversation_id))
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        return query.scalar()


class FactRepository:
    """Repository for memory fact operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, fact: MemoryFact) -> MemoryFact:
        """Create a new fact"""
        try:
            self.session.add(fact)
            self.session.commit()
            self.session.refresh(fact)
            logger.info(f"Created fact: {fact.fact_id}")
            return fact
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating fact: {e}")
            raise

    def get_by_id(self, fact_id: str) -> Optional[MemoryFact]:
        """Get fact by ID"""
        return (
            self.session.query(MemoryFact)
            .filter(MemoryFact.fact_id == fact_id)
            .first()
        )

    def get_by_conversation(self, conversation_id: str) -> List[MemoryFact]:
        """Get facts for a conversation"""
        return (
            self.session.query(MemoryFact)
            .filter(MemoryFact.conversation_id == conversation_id)
            .all()
        )

    def search_by_text(self, query: str, limit: int = 10) -> List[MemoryFact]:
        """Search facts by text (simple text search)"""
        return (
            self.session.query(MemoryFact)
            .filter(MemoryFact.fact.ilike(f"%{query}%"))
            .order_by(desc(MemoryFact.confidence))
            .limit(limit)
            .all()
        )

    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[MemoryFact]:
        """Search facts by tags"""
        return (
            self.session.query(MemoryFact)
            .filter(MemoryFact.tags.overlap(tags))
            .order_by(desc(MemoryFact.confidence))
            .limit(limit)
            .all()
        )

    def semantic_search(
        self, query_embedding: np.ndarray, limit: int = 5, min_confidence: float = 0.5
    ) -> List[Tuple[MemoryFact, float]]:
        """
        Semantic search for facts using vector similarity

        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of (fact, similarity) tuples
        """
        try:
            embedding_list = query_embedding.tolist()

            query = text(
                """
                SELECT fact_id,
                       1 - (embedding <=> :embedding) as similarity
                FROM memory_facts
                WHERE embedding IS NOT NULL
                  AND confidence >= :min_confidence
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """
            )

            result = self.session.execute(
                query,
                {
                    "embedding": str(embedding_list),
                    "min_confidence": min_confidence,
                    "limit": limit,
                },
            )

            results = []
            for row in result:
                fact = self.get_by_id(row[0])
                if fact:
                    results.append((fact, row[1]))

            return results

        except Exception as e:
            logger.error(f"Error in fact semantic search: {e}")
            return []

    def delete(self, fact_id: str) -> bool:
        """Delete fact"""
        try:
            fact = self.get_by_id(fact_id)
            if fact:
                self.session.delete(fact)
                self.session.commit()
                logger.info(f"Deleted fact: {fact_id}")
                return True
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting fact: {e}")
            return False


class UserPreferenceRepository:
    """Repository for user preference operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, user_pref: UserPreference) -> UserPreference:
        """Create user preferences"""
        try:
            self.session.add(user_pref)
            self.session.commit()
            self.session.refresh(user_pref)
            logger.info(f"Created preferences for user: {user_pref.user_id}")
            return user_pref
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating user preferences: {e}")
            raise

    def get(self, user_id: str) -> Optional[UserPreference]:
        """Get user preferences"""
        return (
            self.session.query(UserPreference)
            .filter(UserPreference.user_id == user_id)
            .first()
        )

    def update(self, user_id: str, preferences: dict) -> Optional[UserPreference]:
        """Update user preferences"""
        try:
            user_pref = self.get(user_id)
            if user_pref:
                user_pref.preferences = preferences
                self.session.commit()
                self.session.refresh(user_pref)
                return user_pref
            return None
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating user preferences: {e}")
            raise


class TrainingHistoryRepository:
    """Repository for training history operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, training: TrainingHistory) -> TrainingHistory:
        """Create training record"""
        try:
            self.session.add(training)
            self.session.commit()
            self.session.refresh(training)
            logger.info(f"Created training record: {training.training_id}")
            return training
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating training record: {e}")
            raise

    def get_by_id(self, training_id: str) -> Optional[TrainingHistory]:
        """Get training record by ID"""
        return (
            self.session.query(TrainingHistory)
            .filter(TrainingHistory.training_id == training_id)
            .first()
        )

    def get_recent(self, limit: int = 10) -> List[TrainingHistory]:
        """Get recent training records"""
        return (
            self.session.query(TrainingHistory)
            .order_by(desc(TrainingHistory.started_at))
            .limit(limit)
            .all()
        )

    def get_latest_successful(self) -> Optional[TrainingHistory]:
        """Get latest successful training"""
        return (
            self.session.query(TrainingHistory)
            .filter(TrainingHistory.status == "completed")
            .order_by(desc(TrainingHistory.completed_at))
            .first()
        )

    def update_status(
        self, training_id: str, status: str, error_message: str = None
    ) -> bool:
        """Update training status"""
        try:
            training = self.get_by_id(training_id)
            if training:
                training.status = status
                if status == "completed":
                    training.completed_at = datetime.utcnow()
                if error_message:
                    training.error_message = error_message
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating training status: {e}")
            return False
