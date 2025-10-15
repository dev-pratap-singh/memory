"""
Database Connection Management
Handles PostgreSQL connections with connection pooling
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.database.models import Base
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self, database_url: str = None):
        """
        Initialize database manager

        Args:
            database_url: Database connection URL (optional, uses config if not provided)
        """
        if database_url is None:
            config = get_config()
            database_url = config.database.url

        self.database_url = database_url
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def _create_engine(self) -> Engine:
        """
        Create SQLAlchemy engine with connection pooling

        Returns:
            SQLAlchemy engine
        """
        config = get_config()

        engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,  # Set to True for SQL debugging
        )

        # Register event listeners
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Event listener for new connections"""
            logger.debug("Database connection established")
            # Enable pgvector extension (if not already enabled)
            cursor = dbapi_conn.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.close()

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Event listener for connection checkout from pool"""
            logger.debug("Connection checked out from pool")

        return engine

    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            raise

    def get_session(self) -> Session:
        """
        Get a new database session

        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope around a series of operations

        Yields:
            Database session

        Example:
            with db_manager.session_scope() as session:
                session.add(obj)
                # Automatically commits or rolls back
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """
        Test database connection

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def check_pgvector_extension(self) -> bool:
        """
        Check if pgvector extension is installed

        Returns:
            True if pgvector is available, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                )
                exists = result.scalar()
                if exists:
                    logger.info("pgvector extension is installed")
                else:
                    logger.warning("pgvector extension is NOT installed")
                return exists
        except Exception as e:
            logger.error(f"Error checking pgvector extension: {e}")
            return False

    def get_database_stats(self) -> dict:
        """
        Get database statistics

        Returns:
            Dictionary with database statistics
        """
        try:
            with self.session_scope() as session:
                stats = {}

                # Get table counts
                from src.database.models import (
                    Conversation,
                    MemoryFact,
                    TrainingHistory,
                    UserPreference,
                )

                stats["total_conversations"] = session.query(Conversation).count()
                stats["total_facts"] = session.query(MemoryFact).count()
                stats["total_trainings"] = session.query(TrainingHistory).count()
                stats["total_users"] = session.query(UserPreference).count()

                # Get database size
                result = session.execute(
                    text("SELECT pg_database_size(current_database())")
                )
                stats["database_size_bytes"] = result.scalar()
                stats["database_size_mb"] = stats["database_size_bytes"] / (
                    1024 * 1024
                )

                return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def close(self):
        """Close database connections"""
        try:
            self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Singleton instance
_db_manager: DatabaseManager = None


def get_db_manager() -> DatabaseManager:
    """
    Get database manager singleton

    Returns:
        DatabaseManager instance
    """
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()

    return _db_manager


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session

    Yields:
        Database session
    """
    db_manager = get_db_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


if __name__ == "__main__":
    # Test database connection
    logging.basicConfig(level=logging.INFO)

    db_manager = DatabaseManager()

    print("Testing database connection...")
    if db_manager.test_connection():
        print("✓ Connection successful")

    print("\nChecking pgvector extension...")
    if db_manager.check_pgvector_extension():
        print("✓ pgvector is installed")

    print("\nCreating tables...")
    db_manager.create_tables()
    print("✓ Tables created")

    print("\nDatabase statistics:")
    stats = db_manager.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    db_manager.close()
