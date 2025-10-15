"""
Main Orchestrator
Coordinates all components of the dual-model memory system
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.database.connection import get_db_manager
from src.models.primary_llm import get_primary_llm
from src.models.slm_model import get_slm
from src.models.storage_service import ConversationStorageService
from src.utils.config_loader import get_config
from src.utils.working_memory import get_context_manager, get_working_memory

logger = logging.getLogger(__name__)


class MemoryOrchestrator:
    """Orchestrates the dual-model memory system"""

    def __init__(self):
        """Initialize orchestrator"""
        self.config = get_config()
        self.db_manager = get_db_manager()
        self.primary_llm = get_primary_llm()
        self.slm = get_slm()
        self.working_memory = get_working_memory()
        self.context_manager = get_context_manager()

        logger.info("MemoryOrchestrator initialized")

    def process_conversation(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        use_memory: bool = True,
        store_conversation: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a conversation through the system

        Args:
            messages: List of conversation messages
            user_id: User identifier
            use_memory: Whether to use memory tools
            store_conversation: Whether to store this conversation

        Returns:
            Response dictionary with LLM response and metadata
        """
        try:
            logger.info(f"Processing conversation for user: {user_id}")

            # Get user context from working memory
            user_context = {}
            if user_id:
                user_context = self.working_memory.get_user_context(user_id)

            # Chat with Primary LLM
            response = self.primary_llm.chat(
                messages=messages, use_tools=use_memory, user_id=user_id
            )

            # Store conversation if requested
            conversation_id = None
            if store_conversation:
                conversation_id = self._store_conversation(
                    messages=messages, user_id=user_id, llm_response=response
                )

            # Update working memory
            if conversation_id:
                self.working_memory.add_conversation(
                    conversation_id=conversation_id,
                    summary=self._generate_summary(messages),
                    messages=messages,
                    user_id=user_id,
                )

            return {
                "conversation_id": conversation_id,
                "response": response.get("response", ""),
                "tool_calls": response.get("tool_calls", []),
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing conversation: {e}", exc_info=True)
            return {
                "error": str(e),
                "response": "I encountered an error processing your request.",
            }

    def _store_conversation(
        self, messages: List[Dict], user_id: Optional[str], llm_response: Dict
    ) -> str:
        """Store conversation in database"""
        try:
            with self.db_manager.session_scope() as session:
                storage_service = ConversationStorageService(session)

                # Calculate importance based on tool usage and length
                importance = 0.5
                if llm_response.get("tool_calls"):
                    importance += 0.2
                if len(messages) > 3:
                    importance += 0.1

                importance = min(importance, 1.0)

                # Store conversation
                conv = storage_service.store_conversation(
                    messages=messages,
                    user_id=user_id,
                    importance_score=importance,
                )

                return conv.conversation_id

        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            return None

    def _generate_summary(self, messages: List[Dict]) -> str:
        """Generate simple summary from messages"""
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if user_msgs:
            return f"User asked: {user_msgs[0][:100]}..."
        return "Conversation"

    def query_memory(
        self, query: str, user_id: Optional[str] = None, limit: int = 5
    ) -> Dict[str, Any]:
        """
        Query memory system directly

        Args:
            query: Search query
            user_id: User identifier
            limit: Max results

        Returns:
            Query results
        """
        try:
            from src.tools.memory_tools import get_memory_tools

            tools = get_memory_tools()

            results = tools.memory_search(
                query=query, query_type="semantic", limit=limit, user_id=user_id
            )

            return results

        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return {"error": str(e)}

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Database stats
            db_stats = self.db_manager.get_database_stats()

            # Working memory stats
            wm_stats = self.working_memory.get_stats()

            # Model info
            from src.models.embeddings import get_embedding_generator

            embedder = get_embedding_generator()
            embed_stats = embedder.get_cache_stats()
            model_info = self.slm.get_model_info()

            return {
                "database": db_stats,
                "working_memory": wm_stats,
                "embeddings": embed_stats,
                "slm": model_info,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

    def trigger_training(self, force: bool = False) -> Dict[str, Any]:
        """Trigger SLM training"""
        try:
            from src.tasks.celery_app import train_slm_task

            # Submit async task
            result = train_slm_task.delay(force=force)

            return {
                "status": "submitted",
                "task_id": result.id,
                "message": "Training task submitted",
            }

        except Exception as e:
            logger.error(f"Error triggering training: {e}")
            return {"status": "error", "error": str(e)}


# Singleton
_orchestrator: Optional[MemoryOrchestrator] = None


def get_orchestrator() -> MemoryOrchestrator:
    """Get MemoryOrchestrator singleton"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MemoryOrchestrator()
    return _orchestrator


if __name__ == "__main__":
    from rich import print as rprint

    logging.basicConfig(level=logging.INFO)

    orchestrator = MemoryOrchestrator()

    # Test conversation
    messages = [
        {"role": "user", "content": "Tell me about machine learning"}
    ]

    print("\n=== Processing Conversation ===")
    result = orchestrator.process_conversation(
        messages=messages, user_id="test_user", use_memory=True
    )

    rprint(result)

    # Test system stats
    print("\n=== System Statistics ===")
    stats = orchestrator.get_system_stats()
    rprint(stats)
