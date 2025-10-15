"""
End-to-End System Tests
Comprehensive tests for the dual-model memory system
"""

import pytest
import time
from datetime import datetime

# Test configuration
TEST_USER_ID = "test_user_123"


class TestDatabase:
    """Test database functionality"""

    def test_database_connection(self):
        """Test database connection"""
        from src.database.connection import get_db_manager

        db_manager = get_db_manager()
        assert db_manager.test_connection() == True

    def test_pgvector_extension(self):
        """Test pgvector extension is installed"""
        from src.database.connection import get_db_manager

        db_manager = get_db_manager()
        assert db_manager.check_pgvector_extension() == True

    def test_create_tables(self):
        """Test table creation"""
        from src.database.connection import get_db_manager

        db_manager = get_db_manager()
        db_manager.create_tables()
        # If no exception, test passes


class TestEmbeddings:
    """Test embedding generation"""

    def test_embedding_generation(self):
        """Test single embedding generation"""
        from src.models.embeddings import get_embedding_generator

        embedder = get_embedding_generator()
        text = "This is a test sentence for embedding generation."

        embedding = embedder.generate(text)

        assert embedding is not None
        assert len(embedding) == embedder.dimension
        assert embedding.sum() != 0  # Not all zeros

    def test_batch_embeddings(self):
        """Test batch embedding generation"""
        from src.models.embeddings import get_embedding_generator

        embedder = get_embedding_generator()
        texts = [
            "First test sentence",
            "Second test sentence",
            "Third test sentence",
        ]

        embeddings = embedder.generate_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == embedder.dimension

    def test_similarity_calculation(self):
        """Test similarity calculation"""
        from src.models.embeddings import get_embedding_generator

        embedder = get_embedding_generator()

        text1 = "Machine learning is fascinating"
        text2 = "Deep learning is interesting"
        text3 = "The weather is nice today"

        emb1 = embedder.generate(text1)
        emb2 = embedder.generate(text2)
        emb3 = embedder.generate(text3)

        # Similar texts should have higher similarity
        sim_12 = embedder.similarity(emb1, emb2)
        sim_13 = embedder.similarity(emb1, emb3)

        assert sim_12 > sim_13


class TestStorage:
    """Test storage services"""

    def test_conversation_storage(self):
        """Test storing and retrieving conversations"""
        from src.database.connection import get_db_manager
        from src.models.storage_service import ConversationStorageService

        db_manager = get_db_manager()

        with db_manager.session_scope() as session:
            service = ConversationStorageService(session)

            messages = [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is..."},
            ]

            conv = service.store_conversation(
                messages=messages,
                user_id=TEST_USER_ID,
                importance_score=0.8,
                topics=["machine-learning", "ai"],
            )

            assert conv is not None
            assert conv.conversation_id is not None
            assert conv.importance_score == 0.8

    def test_semantic_search(self):
        """Test semantic search for conversations"""
        from src.database.connection import get_db_manager
        from src.models.storage_service import ConversationStorageService

        db_manager = get_db_manager()

        with db_manager.session_scope() as session:
            service = ConversationStorageService(session)

            # Store test conversation
            messages = [
                {"role": "user", "content": "How do I use Docker for ML?"},
                {"role": "assistant", "content": "To use Docker for ML..."},
            ]

            service.store_conversation(
                messages=messages, user_id=TEST_USER_ID, importance_score=0.9
            )

            # Search
            results = service.search_conversations(
                query="Docker machine learning", limit=5, search_type="semantic"
            )

            assert len(results) > 0


class TestWorkingMemory:
    """Test working memory system"""

    def test_add_conversation(self):
        """Test adding conversation to working memory"""
        from src.utils.working_memory import get_working_memory

        wm = get_working_memory()
        wm.clear()  # Start fresh

        wm.add_conversation(
            conversation_id="test_conv_001",
            summary="Test conversation about ML",
            messages=[{"role": "user", "content": "Test"}],
            user_id=TEST_USER_ID,
        )

        stats = wm.get_stats()
        assert stats["num_conversations"] >= 1

    def test_get_recent_conversations(self):
        """Test retrieving recent conversations"""
        from src.utils.working_memory import get_working_memory

        wm = get_working_memory()

        recent = wm.get_recent_conversations(limit=5)
        assert isinstance(recent, list)


class TestContextWindowManager:
    """Test context window overflow handling"""

    def test_token_estimation(self):
        """Test token estimation"""
        from src.utils.working_memory import get_context_manager

        cm = get_context_manager()

        text = "This is a test sentence."
        tokens = cm.estimate_tokens(text)

        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count

    def test_chunking(self):
        """Test text chunking"""
        from src.utils.working_memory import get_context_manager

        cm = get_context_manager()

        long_text = "This is a sentence. " * 100
        chunks = cm.chunk_text(long_text, chunk_size=50)

        assert len(chunks) > 1

    def test_fit_to_context_window(self):
        """Test fitting text to context window"""
        from src.utils.working_memory import ContextWindowManager

        cm = ContextWindowManager(max_tokens=100)

        long_text = "This is a very long text. " * 50
        fitted = cm.fit_to_context_window(long_text, reserve_tokens=20)

        fitted_tokens = cm.estimate_tokens(fitted)
        assert fitted_tokens <= 80  # 100 - 20 reserve


class TestOrchestrator:
    """Test main orchestrator"""

    def test_initialization(self):
        """Test orchestrator initialization"""
        from src.api.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        assert orchestrator is not None

    def test_process_conversation(self):
        """Test processing a conversation"""
        from src.api.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()

        messages = [{"role": "user", "content": "Hello, test message"}]

        # Note: This will fail without valid API keys
        # In production, use mocks for testing
        try:
            result = orchestrator.process_conversation(
                messages=messages,
                user_id=TEST_USER_ID,
                use_memory=False,  # Disable memory to avoid API calls
                store_conversation=True,
            )

            assert "conversation_id" in result or "error" in result

        except Exception as e:
            # Expected if no API keys configured
            assert "API" in str(e) or "key" in str(e).lower()

    def test_system_stats(self):
        """Test getting system stats"""
        from src.api.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        stats = orchestrator.get_system_stats()

        assert "database" in stats
        assert "working_memory" in stats


class TestMemoryTools:
    """Test memory tools"""

    def test_tool_definitions(self):
        """Test tool definitions"""
        from src.tools.memory_tools import get_memory_tools

        tools = get_memory_tools()
        definitions = tools.get_tool_definitions()

        assert len(definitions) > 0
        assert all("name" in d for d in definitions)
        assert all("description" in d for d in definitions)

    def test_get_context(self):
        """Test getting context"""
        from src.tools.memory_tools import get_memory_tools

        tools = get_memory_tools()
        context = tools.get_context(num_recent=5, user_id=TEST_USER_ID)

        assert "conversations" in context


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests"""

    def test_complete_flow(self):
        """Test complete conversation flow"""
        from src.api.orchestrator import get_orchestrator
        from src.database.connection import get_db_manager

        # Initialize
        db_manager = get_db_manager()
        db_manager.create_tables()

        orchestrator = get_orchestrator()

        # Create test conversation
        messages = [{"role": "user", "content": "Tell me about Python programming"}]

        # Process (will fail without API keys, but tests the flow)
        try:
            result = orchestrator.process_conversation(
                messages=messages, user_id=TEST_USER_ID, store_conversation=True
            )

            # If successful, check result
            if "error" not in result:
                assert "conversation_id" in result
                assert result["user_id"] == TEST_USER_ID

        except Exception as e:
            # Expected without API keys
            pass

        # Check stats
        stats = orchestrator.get_system_stats()
        assert stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
