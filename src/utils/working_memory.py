"""
Working Memory System
Real-time context file for fast access to recent conversations
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Real-time working memory for fast context access"""

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize working memory

        Args:
            file_path: Path to working memory file (uses config if not provided)
        """
        config = get_config()
        self.file_path = file_path or config.memory.working_memory["file_path"]
        self.max_size = config.memory.working_memory["max_size"]

        # Create file path if doesn't exist
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize
        self.data = self._load()

    def _load(self) -> Dict:
        """Load working memory from file"""
        try:
            if Path(self.file_path).exists():
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded working memory with {len(data.get('conversations', []))} entries")
                    return data
        except Exception as e:
            logger.warning(f"Error loading working memory: {e}")

        # Return empty structure
        return {
            "conversations": [],
            "user_context": {},
            "last_updated": datetime.utcnow().isoformat(),
        }

    def _save(self):
        """Save working memory to file"""
        try:
            self.data["last_updated"] = datetime.utcnow().isoformat()
            with open(self.file_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving working memory: {e}")

    def add_conversation(
        self, conversation_id: str, summary: str, messages: List[Dict], **kwargs
    ):
        """Add conversation to working memory"""
        entry = {
            "conversation_id": conversation_id,
            "summary": summary,
            "messages": messages,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }

        self.data["conversations"].insert(0, entry)

        # Keep only max_size entries
        if len(self.data["conversations"]) > self.max_size:
            self.data["conversations"] = self.data["conversations"][: self.max_size]

        self._save()
        logger.debug(f"Added conversation to working memory: {conversation_id}")

    def get_recent_conversations(self, limit: Optional[int] = None) -> List[Dict]:
        """Get recent conversations"""
        limit = limit or self.max_size
        return self.data["conversations"][:limit]

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get specific conversation"""
        for conv in self.data["conversations"]:
            if conv["conversation_id"] == conversation_id:
                return conv
        return None

    def update_user_context(self, user_id: str, context: Dict):
        """Update user-specific context"""
        if "user_context" not in self.data:
            self.data["user_context"] = {}

        self.data["user_context"][user_id] = {
            **context,
            "last_updated": datetime.utcnow().isoformat(),
        }
        self._save()

    def get_user_context(self, user_id: str) -> Dict:
        """Get user-specific context"""
        return self.data.get("user_context", {}).get(user_id, {})

    def clear(self):
        """Clear working memory"""
        self.data = {
            "conversations": [],
            "user_context": {},
            "last_updated": datetime.utcnow().isoformat(),
        }
        self._save()
        logger.info("Cleared working memory")

    def get_stats(self) -> Dict:
        """Get working memory statistics"""
        return {
            "num_conversations": len(self.data["conversations"]),
            "max_size": self.max_size,
            "num_users": len(self.data.get("user_context", {})),
            "last_updated": self.data.get("last_updated"),
        }


class ContextWindowManager:
    """Manages context window overflow by chunking and prioritization"""

    def __init__(self, max_tokens: int = 2048):
        """
        Initialize context window manager

        Args:
            max_tokens: Maximum context window size in tokens
        """
        self.max_tokens = max_tokens
        self.config = get_config()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return len(text) // 4

    def chunk_text(
        self, text: str, chunk_size: Optional[int] = None, overlap: int = 128
    ) -> List[str]:
        """
        Chunk text into smaller pieces

        Args:
            text: Input text
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks in tokens

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.memory.context_window.get(
            "chunk_size", 1024
        )

        # Convert tokens to characters (rough estimate)
        chunk_chars = chunk_size * 4
        overlap_chars = overlap * 4

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_chars
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap_chars

        logger.debug(f"Chunked text into {len(chunks)} pieces")
        return chunks

    def prioritize_chunks(
        self, chunks: List[str], query: str, importance_scores: Optional[List[float]] = None
    ) -> List[tuple]:
        """
        Prioritize chunks based on relevance to query

        Args:
            chunks: List of text chunks
            query: User query
            importance_scores: Optional precomputed importance scores

        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        from src.models.embeddings import get_embedding_generator

        embedder = get_embedding_generator()

        # Generate embeddings
        query_embedding = embedder.generate(query)
        chunk_embeddings = embedder.generate_batch(chunks)

        # Calculate similarities
        scored_chunks = []
        for i, (chunk, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
            similarity = embedder.similarity(query_embedding, chunk_emb)

            # Apply importance score if provided
            if importance_scores and i < len(importance_scores):
                similarity *= importance_scores[i]

            scored_chunks.append((chunk, similarity))

        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks

    def fit_to_context_window(
        self, text: str, query: Optional[str] = None, reserve_tokens: int = 500
    ) -> str:
        """
        Fit text to context window, chunking and prioritizing if needed

        Args:
            text: Input text
            query: User query for prioritization
            reserve_tokens: Tokens to reserve for response

        Returns:
            Text that fits in context window
        """
        available_tokens = self.max_tokens - reserve_tokens
        text_tokens = self.estimate_tokens(text)

        # If fits, return as is
        if text_tokens <= available_tokens:
            return text

        logger.info(f"Text too large ({text_tokens} tokens), chunking...")

        # Chunk the text
        chunks = self.chunk_text(text)

        # If query provided, prioritize chunks
        if query:
            scored_chunks = self.prioritize_chunks(chunks, query)
            chunks = [chunk for chunk, score in scored_chunks]

        # Take chunks until we hit the limit
        result_chunks = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.estimate_tokens(chunk)
            if current_tokens + chunk_tokens <= available_tokens:
                result_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                break

        result = " [...] ".join(result_chunks)
        logger.info(f"Reduced to {len(result_chunks)} chunks ({current_tokens} tokens)")

        return result


# Singletons
_working_memory: Optional[WorkingMemory] = None
_context_manager: Optional[ContextWindowManager] = None


def get_working_memory() -> WorkingMemory:
    """Get WorkingMemory singleton"""
    global _working_memory
    if _working_memory is None:
        _working_memory = WorkingMemory()
    return _working_memory


def get_context_manager() -> ContextWindowManager:
    """Get ContextWindowManager singleton"""
    global _context_manager
    if _context_manager is None:
        config = get_config()
        max_tokens = config.memory.context_window.get("max_tokens", 2048)
        _context_manager = ContextWindowManager(max_tokens=max_tokens)
    return _context_manager


if __name__ == "__main__":
    from rich import print as rprint

    logging.basicConfig(level=logging.INFO)

    # Test working memory
    print("\n=== Testing Working Memory ===")
    wm = WorkingMemory()

    wm.add_conversation(
        conversation_id="test_001",
        summary="Discussion about machine learning",
        messages=[{"role": "user", "content": "Tell me about ML"}],
    )

    rprint("Stats:", wm.get_stats())
    rprint("Recent:", wm.get_recent_conversations(limit=2))

    # Test context manager
    print("\n=== Testing Context Window Manager ===")
    cm = ContextWindowManager(max_tokens=100)

    long_text = "This is a very long text that needs to be chunked. " * 50
    fitted_text = cm.fit_to_context_window(long_text, reserve_tokens=20)
    rprint(f"Original: {cm.estimate_tokens(long_text)} tokens")
    rprint(f"Fitted: {cm.estimate_tokens(fitted_text)} tokens")
