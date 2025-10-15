"""
Memory Tools for Primary LLM
Tool calling interface for accessing the memory system
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.database.connection import get_db_manager
from src.models.slm_model import get_slm
from src.models.storage_service import (
    ConversationStorageService,
    FactStorageService,
    UserPreferenceService,
)

logger = logging.getLogger(__name__)


class MemoryTools:
    """Tools for Primary LLM to access memory"""

    def __init__(self):
        self.db_manager = get_db_manager()
        self.slm = get_slm()

    def memory_search(
        self,
        query: str,
        query_type: str = "semantic",
        limit: int = 5,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search memory for relevant conversations and facts

        Args:
            query: Search query
            query_type: 'semantic', 'temporal', or 'factual'
            limit: Maximum number of results
            user_id: User ID to filter by

        Returns:
            Dictionary with search results
        """
        try:
            with self.db_manager.session_scope() as session:
                conv_service = ConversationStorageService(session)
                fact_service = FactStorageService(session)

                results = {
                    "query": query,
                    "query_type": query_type,
                    "slm_response": None,
                    "conversations": [],
                    "facts": [],
                }

                # Query SLM for direct response
                slm_response = self.slm.query(query)
                results["slm_response"] = slm_response

                # Search conversations
                if query_type in ["semantic", "temporal"]:
                    conv_results = conv_service.search_conversations(
                        query=query,
                        limit=limit,
                        user_id=user_id,
                        search_type="semantic" if query_type == "semantic" else "topic",
                    )

                    results["conversations"] = [
                        {
                            "conversation_id": conv.conversation_id,
                            "summary": conv.summary,
                            "importance_score": conv.importance_score,
                            "topics": conv.topics,
                            "created_at": conv.created_at.isoformat(),
                            "similarity_score": score,
                        }
                        for conv, score in conv_results
                    ]

                # Search facts
                if query_type in ["semantic", "factual"]:
                    fact_results = fact_service.search_facts(
                        query=query, limit=limit, search_type="semantic"
                    )

                    results["facts"] = [
                        {
                            "fact": fact.fact,
                            "confidence": fact.confidence,
                            "tags": fact.tags,
                            "conversation_id": fact.conversation_id,
                            "similarity_score": score,
                        }
                        for fact, score in fact_results
                    ]

                return results

        except Exception as e:
            logger.error(f"Error in memory_search: {e}")
            return {"error": str(e)}

    def memory_store(
        self,
        content: str,
        content_type: str = "conversation",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Store new information in memory

        Args:
            content: Content to store
            content_type: 'conversation' or 'fact'
            importance: Importance score (0-1)
            tags: List of tags
            user_id: User ID
            metadata: Additional metadata

        Returns:
            Dictionary with storage confirmation
        """
        try:
            with self.db_manager.session_scope() as session:
                if content_type == "conversation":
                    # Parse messages if JSON
                    try:
                        messages = json.loads(content)
                    except:
                        # Create single message
                        messages = [{"role": "user", "content": content}]

                    conv_service = ConversationStorageService(session)
                    conv = conv_service.store_conversation(
                        messages=messages,
                        user_id=user_id,
                        importance_score=importance,
                        topics=tags,
                        metadata=metadata,
                    )

                    return {
                        "status": "success",
                        "type": "conversation",
                        "id": conv.conversation_id,
                        "message": "Conversation stored successfully",
                    }

                elif content_type == "fact":
                    fact_service = FactStorageService(session)

                    # Need a conversation ID - create temporary one
                    conv_service = ConversationStorageService(session)
                    temp_messages = [{"role": "system", "content": content}]
                    conv = conv_service.store_conversation(
                        messages=temp_messages, user_id=user_id, importance_score=0.3
                    )

                    fact = fact_service.store_fact(
                        fact_text=content,
                        conversation_id=conv.conversation_id,
                        confidence=importance,
                        tags=tags or [],
                    )

                    return {
                        "status": "success",
                        "type": "fact",
                        "id": fact.fact_id,
                        "message": "Fact stored successfully",
                    }

                else:
                    return {"status": "error", "message": f"Unknown content_type: {content_type}"}

        except Exception as e:
            logger.error(f"Error in memory_store: {e}")
            return {"status": "error", "message": str(e)}

    def get_context(
        self, num_recent: int = 3, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get recent conversation context

        Args:
            num_recent: Number of recent conversations
            user_id: User ID to filter by

        Returns:
            Dictionary with recent context
        """
        try:
            with self.db_manager.session_scope() as session:
                conv_service = ConversationStorageService(session)

                recent_convs = conv_service.get_recent_conversations(
                    limit=num_recent, user_id=user_id
                )

                return {
                    "num_conversations": len(recent_convs),
                    "conversations": [
                        {
                            "conversation_id": conv.conversation_id,
                            "summary": conv.summary,
                            "topics": conv.topics,
                            "importance_score": conv.importance_score,
                            "created_at": conv.created_at.isoformat(),
                        }
                        for conv in recent_convs
                    ],
                }

        except Exception as e:
            logger.error(f"Error in get_context: {e}")
            return {"error": str(e)}

    def summarize_memory(
        self,
        time_range: str = "all",
        topics: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get summary of stored memories

        Args:
            time_range: 'day', 'week', 'month', 'all'
            topics: Filter by topics
            user_id: User ID to filter by

        Returns:
            Dictionary with memory summary
        """
        try:
            with self.db_manager.session_scope() as session:
                conv_service = ConversationStorageService(session)
                fact_service = FactStorageService(session)

                # Determine date range
                end_date = datetime.utcnow()
                if time_range == "day":
                    start_date = end_date - timedelta(days=1)
                elif time_range == "week":
                    start_date = end_date - timedelta(weeks=1)
                elif time_range == "month":
                    start_date = end_date - timedelta(days=30)
                else:
                    start_date = datetime(2000, 1, 1)

                # Get conversations
                if topics:
                    conversations = conv_service.repo.search_by_topic(topics, limit=100)
                else:
                    conversations = conv_service.repo.search_by_date_range(
                        start_date, end_date
                    )

                # Filter by user
                if user_id:
                    conversations = [c for c in conversations if c.user_id == user_id]

                # Aggregate statistics
                total_convs = len(conversations)
                avg_importance = (
                    sum(c.importance_score for c in conversations) / total_convs
                    if total_convs > 0
                    else 0
                )

                all_topics = []
                for conv in conversations:
                    if conv.topics:
                        all_topics.extend(conv.topics)

                # Count topic frequency
                topic_counts = {}
                for topic in all_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

                top_topics = sorted(
                    topic_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]

                return {
                    "time_range": time_range,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_conversations": total_convs,
                    "average_importance": avg_importance,
                    "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
                    "recent_summaries": [
                        c.summary for c in conversations[:5] if c.summary
                    ],
                }

        except Exception as e:
            logger.error(f"Error in summarize_memory: {e}")
            return {"error": str(e)}

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        try:
            with self.db_manager.session_scope() as session:
                pref_service = UserPreferenceService(session)
                prefs = pref_service.get_preferences(user_id)

                return {"user_id": user_id, "preferences": prefs}

        except Exception as e:
            logger.error(f"Error in get_user_preferences: {e}")
            return {"error": str(e)}

    def update_user_preference(
        self, user_id: str, key: str, value: Any
    ) -> Dict[str, Any]:
        """Update single user preference"""
        try:
            with self.db_manager.session_scope() as session:
                pref_service = UserPreferenceService(session)
                pref_service.add_preference(user_id, key, value)

                return {
                    "status": "success",
                    "message": f"Updated preference: {key}",
                    "user_id": user_id,
                }

        except Exception as e:
            logger.error(f"Error in update_user_preference: {e}")
            return {"status": "error", "message": str(e)}

    def get_tool_definitions(self) -> List[Dict]:
        """
        Get tool definitions for Primary LLM function calling

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "memory_search",
                "description": "Search memory for relevant conversations and facts. Use this when the user asks about previous conversations or stored information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["semantic", "temporal", "factual"],
                            "description": "Type of search to perform",
                            "default": "semantic",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_store",
                "description": "Store new information in memory. Use when user explicitly asks to remember something.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to store",
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["conversation", "fact"],
                            "default": "fact",
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score (0-1)",
                            "default": 0.8,
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "get_context",
                "description": "Get recent conversation context. Use when user asks 'what were we discussing' or similar.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_recent": {
                            "type": "integer",
                            "description": "Number of recent conversations",
                            "default": 3,
                        }
                    },
                },
            },
            {
                "name": "summarize_memory",
                "description": "Get summary of stored memories over a time period.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_range": {
                            "type": "string",
                            "enum": ["day", "week", "month", "all"],
                            "default": "week",
                        },
                        "topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by topics",
                        },
                    },
                },
            },
        ]


# Singleton instance
_memory_tools: Optional[MemoryTools] = None


def get_memory_tools() -> MemoryTools:
    """Get MemoryTools singleton"""
    global _memory_tools

    if _memory_tools is None:
        _memory_tools = MemoryTools()

    return _memory_tools


if __name__ == "__main__":
    # Test memory tools
    from rich import print as rprint

    logging.basicConfig(level=logging.INFO)

    tools = MemoryTools()

    # Get tool definitions
    print("\n=== Tool Definitions ===")
    definitions = tools.get_tool_definitions()
    for tool_def in definitions:
        rprint(f"\n{tool_def['name']}: {tool_def['description']}")

    # Test context retrieval
    print("\n=== Testing Context Retrieval ===")
    context = tools.get_context(num_recent=5)
    rprint(context)

    # Test memory summary
    print("\n=== Testing Memory Summary ===")
    summary = tools.summarize_memory(time_range="all")
    rprint(summary)
