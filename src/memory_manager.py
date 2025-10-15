"""
LLM Memory Management System
Implements context compression, memory state tracking, and intelligent memory management
Supports both external LLM approach and SLM-based approach
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import tiktoken
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from pgvector.psycopg2 import register_vector
import openai


class CompressionStrategy(Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MemoryApproach(Enum):
    EXTERNAL_LLM = "external_llm"  # Approach 1: Context partitioning with compression
    SLM = "slm"  # Approach 2: Dual-model with SLM


@dataclass
class MemoryItem:
    id: str
    session_id: str
    content_type: str
    full_content: str
    compressed_summary: str
    token_count: int
    compressed_token_count: int
    is_active: bool = True
    priority: int = 0
    metadata: Dict = None


@dataclass
class MemoryStateSnapshot:
    session_id: str
    model_name: str
    total_context_length: int
    used_context_length: int
    available_context_length: int
    context_utilization_percentage: float
    active_items: List[Dict]
    compressed_items: List[Dict]
    token_usage_stats: Dict
    performance_metrics: Dict
    timestamp: str


class LLMMemoryManager:
    """
    Comprehensive memory management system for LLMs with:
    - Context compression/decompression (Approach 1)
    - SLM-based memory management (Approach 2)
    - Memory state tracking
    - Token usage monitoring
    - Chat hygiene
    - Context quarantine
    - Task management
    - Agent management
    - Human-in-the-loop
    """

    def __init__(
        self,
        db_host: str,
        db_port: int,
        db_name: str,
        db_user: str,
        db_password: str,
        model_name: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        config_path: str = "src/utils/model_context.json",
        memory_approach: str = "external_llm",
        use_slm: bool = False,
        use_lora_finetuning: bool = False,
        slm_model_name: Optional[str] = None
    ):
        """Initialize the memory manager"""
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password
        }

        self.conn = None
        self.model_name = model_name
        self.session_id = str(uuid.uuid4())
        self.memory_approach = MemoryApproach(memory_approach)
        self.use_slm = use_slm
        self.use_lora_finetuning = use_lora_finetuning
        self.slm_model_name = slm_model_name

        # Load model configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.model_config = self.config['models'].get(model_name)
        if not self.model_config:
            raise ValueError(f"Model {model_name} not found in configuration")

        # Initialize API keys
        if openai_api_key:
            openai.api_key = openai_api_key
        if anthropic_api_key:
            self.anthropic_api_key = anthropic_api_key

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Connect to database
        self._connect_db()

        # Initialize memory state
        self._initialize_memory_state()

        # Initialize SLM if needed
        if self.use_slm and self.memory_approach == MemoryApproach.SLM:
            self._initialize_slm()

    def _connect_db(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        register_vector(self.conn)

    def _get_cursor(self):
        """Get database cursor with connection check"""
        if self.conn.closed:
            self._connect_db()
        return self.conn.cursor(cursor_factory=RealDictCursor)

    def _initialize_slm(self):
        """Initialize SLM for memory management"""
        # This would load the SLM model (Gemma-3, etc.)
        # For now, just log that SLM is enabled
        print(f"SLM enabled: {self.slm_model_name}")
        print(f"LoRA fine-tuning enabled: {self.use_lora_finetuning}")

    def _initialize_memory_state(self):
        """Initialize memory state for the session"""
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO memory_state (
                session_id, model_name, total_context_length,
                used_context_length, available_context_length,
                context_utilization_percentage, token_usage_stats,
                performance_metrics
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            self.session_id,
            self.model_name,
            self.model_config['context_length'],
            0,
            self.model_config['context_length'],
            0.0,
            Json({'input_tokens': 0, 'output_tokens': 0, 'total_cost': 0.0}),
            Json({'compressions': 0, 'decompressions': 0, 'cache_hits': 0})
        ))
        self.conn.commit()
        cursor.close()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 1536  # Return zero vector on error

    def compress_content(
        self,
        content: str,
        content_type: str,
        strategy: CompressionStrategy = CompressionStrategy.BALANCED
    ) -> str:
        """
        Compress content using LLM summarization (Approach 1)
        """
        strategy_config = self.config['compression_strategies'][strategy.value]
        token_count = self.count_tokens(content)
        target_tokens = int(token_count * strategy_config['summary_ratio'])

        prompt = f"""Summarize the following {content_type} content in approximately {target_tokens} tokens.
Maintain {strategy_config['detail_retention']} level of detail.
Focus on key information that would be needed to understand and work with this content.

Content:
{content}

Summary:"""

        try:
            # Cap max_tokens at 16,384 for gpt-4o-mini
            max_completion_tokens = min(target_tokens * 2, 16384)

            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for compression
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_completion_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error compressing content: {e}")
            # Fallback: simple truncation
            return content[:target_tokens * 4]  # Approximate chars from tokens

    def add_memory_item(
        self,
        content: str,
        content_type: str,
        priority: int = 0,
        metadata: Optional[Dict] = None,
        auto_compress: bool = True
    ) -> str:
        """
        Add a new memory item to the system
        Returns the item ID
        """
        token_count = self.count_tokens(content)

        # Check if compression is needed
        current_state = self.get_memory_state()
        should_compress = (
            auto_compress and
            current_state['context_utilization_percentage'] >
            self.model_config['compression_trigger'] * 100
        )

        if should_compress or token_count > 2000:
            compressed_summary = self.compress_content(
                content, content_type, CompressionStrategy.BALANCED
            )
        else:
            compressed_summary = content[:500] + "..." if len(content) > 500 else content

        compressed_token_count = self.count_tokens(compressed_summary)
        compression_ratio = compressed_token_count / token_count if token_count > 0 else 1.0

        # Generate embedding for semantic search
        embedding = self.get_embedding(compressed_summary)

        cursor = self._get_cursor()
        item_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO memory_items (
                id, session_id, content_type, full_content, compressed_summary,
                embedding, token_count, compressed_token_count, compression_ratio,
                priority, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            item_id, self.session_id, content_type, content, compressed_summary,
            embedding, token_count, compressed_token_count, compression_ratio,
            priority, Json(metadata or {})
        ))

        self.conn.commit()
        cursor.close()

        # Update memory state
        self._update_memory_state()

        return item_id

    def retrieve_memory_item(self, item_id: str, decompress: bool = True) -> Optional[Dict]:
        """
        Retrieve a memory item by ID
        If decompress=True, returns full content; otherwise returns compressed summary
        """
        cursor = self._get_cursor()
        cursor.execute("""
            UPDATE memory_items
            SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
            WHERE id = %s AND session_id = %s
            RETURNING *
        """, (item_id, self.session_id))

        item = cursor.fetchone()
        self.conn.commit()
        cursor.close()

        if not item:
            return None

        # Update performance metrics
        self._update_performance_metric('decompressions' if decompress else 'cache_hits', 1)

        return dict(item)

    def search_memory_semantic(
        self,
        query: str,
        limit: int = 5,
        content_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Semantic search through memory items using pgvector
        """
        query_embedding = self.get_embedding(query)

        cursor = self._get_cursor()

        if content_types:
            cursor.execute("""
                SELECT *, embedding <=> %s::vector AS distance
                FROM memory_items
                WHERE session_id = %s AND is_active = true AND content_type = ANY(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, self.session_id, content_types, query_embedding, limit))
        else:
            cursor.execute("""
                SELECT *, embedding <=> %s::vector AS distance
                FROM memory_items
                WHERE session_id = %s AND is_active = true
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, self.session_id, query_embedding, limit))

        results = cursor.fetchall()
        cursor.close()

        return [dict(r) for r in results]

    def get_working_context(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the current working context with active items
        Automatically manages what should be compressed vs. full content
        """
        if max_tokens is None:
            max_tokens = int(self.model_config['context_length'] * self.model_config['safe_threshold'])

        cursor = self._get_cursor()
        cursor.execute("""
            SELECT * FROM memory_items
            WHERE session_id = %s AND is_active = true
            ORDER BY priority DESC, last_accessed DESC
        """, (self.session_id,))

        items = cursor.fetchall()
        cursor.close()

        working_context = {
            'full_items': [],
            'compressed_items': [],
            'total_tokens': 0
        }

        current_tokens = 0

        for item in items:
            item_dict = dict(item)

            # Try to fit full content first
            if current_tokens + item_dict['token_count'] <= max_tokens:
                working_context['full_items'].append({
                    'id': item_dict['id'],
                    'type': item_dict['content_type'],
                    'content': item_dict['full_content'],
                    'tokens': item_dict['token_count']
                })
                current_tokens += item_dict['token_count']
            # Otherwise use compressed version
            elif current_tokens + item_dict['compressed_token_count'] <= max_tokens:
                working_context['compressed_items'].append({
                    'id': item_dict['id'],
                    'type': item_dict['content_type'],
                    'summary': item_dict['compressed_summary'],
                    'tokens': item_dict['compressed_token_count'],
                    'retrieval_hint': f"Use retrieve_memory_item('{item_dict['id']}') to get full content"
                })
                current_tokens += item_dict['compressed_token_count']

        working_context['total_tokens'] = current_tokens
        working_context['available_tokens'] = max_tokens - current_tokens

        return working_context

    def _update_memory_state(self):
        """Update the memory state snapshot"""
        cursor = self._get_cursor()

        # Get current memory items
        cursor.execute("""
            SELECT
                SUM(CASE WHEN is_active THEN token_count ELSE 0 END) as active_tokens,
                SUM(CASE WHEN is_active THEN compressed_token_count ELSE 0 END) as compressed_tokens,
                COUNT(*) FILTER (WHERE is_active) as active_count,
                json_agg(json_build_object(
                    'id', id,
                    'type', content_type,
                    'tokens', token_count,
                    'compressed_tokens', compressed_token_count,
                    'priority', priority
                ) ORDER BY priority DESC) FILTER (WHERE is_active) as items
            FROM memory_items
            WHERE session_id = %s
        """, (self.session_id,))

        stats = cursor.fetchone()

        used_tokens = stats['compressed_tokens'] or 0
        available_tokens = self.model_config['context_length'] - used_tokens
        utilization = (used_tokens / self.model_config['context_length']) * 100

        cursor.execute("""
            UPDATE memory_state
            SET
                used_context_length = %s,
                available_context_length = %s,
                context_utilization_percentage = %s,
                active_items = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s
        """, (
            used_tokens,
            available_tokens,
            utilization,
            Json(stats['items'] or []),
            self.session_id
        ))

        self.conn.commit()
        cursor.close()

    def _update_performance_metric(self, metric_name: str, increment: int = 1):
        """Update performance metrics"""
        cursor = self._get_cursor()
        cursor.execute("""
            UPDATE memory_state
            SET performance_metrics = jsonb_set(
                performance_metrics,
                %s,
                (COALESCE(performance_metrics->%s, '0')::int + %s)::text::jsonb
            )
            WHERE session_id = %s
        """, ([metric_name], metric_name, increment, self.session_id))
        self.conn.commit()
        cursor.close()

    def get_memory_state(self) -> Dict[str, Any]:
        """Get current memory state snapshot"""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT * FROM memory_state
            WHERE session_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
        """, (self.session_id,))

        state = cursor.fetchone()
        cursor.close()

        return dict(state) if state else {}

    def perform_chat_hygiene(self):
        """
        Perform chat hygiene: compress old items, prune inactive content
        """
        cursor = self._get_cursor()

        # Compress items that haven't been accessed recently
        threshold_date = datetime.now() - timedelta(days=7)
        cursor.execute("""
            SELECT * FROM memory_items
            WHERE session_id = %s
            AND is_active = true
            AND last_accessed < %s
            AND token_count > compressed_token_count * 2
        """, (self.session_id, threshold_date))

        items_to_compress = cursor.fetchall()

        for item in items_to_compress:
            # Already has compression, just mark for reference
            self._update_performance_metric('compressions', 1)

        # Deactivate very old items
        prune_date = datetime.now() - timedelta(
            days=self.config['memory_management']['auto_prune_after_days']
        )
        cursor.execute("""
            UPDATE memory_items
            SET is_active = false
            WHERE session_id = %s
            AND last_accessed < %s
            AND priority < 5
        """, (self.session_id, prune_date))

        self.conn.commit()
        cursor.close()

        # Update memory state
        self._update_memory_state()

    # ===== TASK MANAGEMENT =====

    def create_task(
        self,
        title: str,
        description: str = "",
        priority: int = 0,
        assigned_agent: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a new task"""
        cursor = self._get_cursor()
        task_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO tasks (
                id, session_id, title, description, priority,
                assigned_agent, dependencies, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            task_id, self.session_id, title, description, priority,
            assigned_agent, Json(dependencies or []), Json(metadata or {})
        ))

        self.conn.commit()
        cursor.close()

        return task_id

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """Update task status and progress"""
        cursor = self._get_cursor()

        updates = []
        params = []

        if status:
            updates.append("status = %s")
            params.append(status.value)
            if status == TaskStatus.COMPLETED:
                updates.append("completed_at = CURRENT_TIMESTAMP")

        if progress is not None:
            updates.append("progress = %s")
            params.append(progress)

        if metadata:
            updates.append("metadata = %s")
            params.append(Json(metadata))

        if updates:
            params.extend([task_id, self.session_id])
            cursor.execute(f"""
                UPDATE tasks
                SET {', '.join(updates)}
                WHERE id = %s AND session_id = %s
            """, params)
            self.conn.commit()

        cursor.close()

    def get_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict]:
        """Get tasks, optionally filtered by status"""
        cursor = self._get_cursor()

        if status:
            cursor.execute("""
                SELECT * FROM tasks
                WHERE session_id = %s AND status = %s
                ORDER BY priority DESC, created_at DESC
            """, (self.session_id, status.value))
        else:
            cursor.execute("""
                SELECT * FROM tasks
                WHERE session_id = %s
                ORDER BY priority DESC, created_at DESC
            """, (self.session_id,))

        tasks = cursor.fetchall()
        cursor.close()

        return [dict(t) for t in tasks]

    # ===== AGENT MANAGEMENT =====

    def create_agent(
        self,
        agent_name: str,
        agent_type: str,
        capabilities: List[str],
        assigned_tools: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a new agent"""
        cursor = self._get_cursor()
        agent_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO agents (
                id, session_id, agent_name, agent_type,
                capabilities, assigned_tools, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            agent_id, self.session_id, agent_name, agent_type,
            Json(capabilities), Json(assigned_tools or []), Json(metadata or {})
        ))

        self.conn.commit()
        cursor.close()

        return agent_id

    def get_agents(self, agent_type: Optional[str] = None) -> List[Dict]:
        """Get agents, optionally filtered by type"""
        cursor = self._get_cursor()

        if agent_type:
            cursor.execute("""
                SELECT * FROM agents
                WHERE session_id = %s AND agent_type = %s AND is_active = true
            """, (self.session_id, agent_type))
        else:
            cursor.execute("""
                SELECT * FROM agents
                WHERE session_id = %s AND is_active = true
            """, (self.session_id,))

        agents = cursor.fetchall()
        cursor.close()

        return [dict(a) for a in agents]

    # ===== CONTEXT QUARANTINE =====

    def quarantine_context(
        self,
        content: str,
        context_type: str,
        reason: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Isolate context in its own dedicated thread"""
        cursor = self._get_cursor()
        quarantine_id = str(uuid.uuid4())
        thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        token_count = self.count_tokens(content)

        cursor.execute("""
            INSERT INTO context_quarantine (
                id, session_id, thread_id, context_type,
                isolated_content, reason_for_isolation, token_count, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            quarantine_id, self.session_id, thread_id, context_type,
            content, reason, token_count, Json(metadata or {})
        ))

        self.conn.commit()
        cursor.close()

        return thread_id

    def get_quarantined_contexts(self) -> List[Dict]:
        """Get all quarantined contexts"""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT * FROM context_quarantine
            WHERE session_id = %s AND is_active = true
            ORDER BY created_at DESC
        """, (self.session_id,))

        contexts = cursor.fetchall()
        cursor.close()

        return [dict(c) for c in contexts]

    # ===== HUMAN IN THE LOOP =====

    def request_human_input(
        self,
        question: str,
        priority: int = 0,
        metadata: Optional[Dict] = None
    ) -> str:
        """Request input from human"""
        cursor = self._get_cursor()
        hil_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO hil_interactions (
                id, session_id, question, priority, metadata
            ) VALUES (%s, %s, %s, %s, %s)
        """, (hil_id, self.session_id, question, priority, Json(metadata or {})))

        self.conn.commit()
        cursor.close()

        return hil_id

    def provide_human_response(self, hil_id: str, response: str):
        """Provide response to HIL request"""
        cursor = self._get_cursor()
        cursor.execute("""
            UPDATE hil_interactions
            SET response = %s, status = 'answered', answered_at = CURRENT_TIMESTAMP
            WHERE id = %s AND session_id = %s
        """, (response, hil_id, self.session_id))

        self.conn.commit()
        cursor.close()

    def get_pending_hil_requests(self) -> List[Dict]:
        """Get pending HIL requests"""
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT * FROM hil_interactions
            WHERE session_id = %s AND status = 'pending'
            ORDER BY priority DESC, created_at ASC
        """, (self.session_id,))

        requests = cursor.fetchall()
        cursor.close()

        return [dict(r) for r in requests]

    # ===== TOKEN TRACKING =====

    def track_token_usage(self, input_tokens: int, output_tokens: int):
        """Track token usage and costs"""
        cursor = self._get_cursor()

        input_cost = (input_tokens / 1000) * self.model_config['cost_per_1k_input']
        output_cost = (output_tokens / 1000) * self.model_config['cost_per_1k_output']
        total_cost = input_cost + output_cost

        cursor.execute("""
            UPDATE memory_state
            SET token_usage_stats = jsonb_set(
                jsonb_set(
                    jsonb_set(
                        token_usage_stats,
                        '{input_tokens}',
                        ((token_usage_stats->>'input_tokens')::int + %s)::text::jsonb
                    ),
                    '{output_tokens}',
                    ((token_usage_stats->>'output_tokens')::int + %s)::text::jsonb
                ),
                '{total_cost}',
                ((token_usage_stats->>'total_cost')::float + %s)::text::jsonb
            )
            WHERE session_id = %s
        """, (input_tokens, output_tokens, total_cost, self.session_id))

        self.conn.commit()
        cursor.close()

    # ===== UTILITY METHODS =====

    def export_memory_state_json(self) -> str:
        """Export complete memory state as JSON for UI display"""
        state = self.get_memory_state()

        # Get all active memory items
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT id, content_type, compressed_summary, token_count,
                   compressed_token_count, priority, access_count, last_accessed
            FROM memory_items
            WHERE session_id = %s AND is_active = true
            ORDER BY priority DESC, last_accessed DESC
        """, (self.session_id,))

        items = [dict(item) for item in cursor.fetchall()]

        # Get tasks
        tasks = self.get_tasks()

        # Get agents
        agents = self.get_agents()

        # Get quarantined contexts
        quarantined = self.get_quarantined_contexts()

        cursor.close()

        export_data = {
            'session_id': self.session_id,
            'model': self.model_name,
            'memory_approach': self.memory_approach.value,
            'use_slm': self.use_slm,
            'use_lora_finetuning': self.use_lora_finetuning,
            'timestamp': datetime.now().isoformat(),
            'memory_state': {
                'total_context': state.get('total_context_length', 0),
                'used_context': state.get('used_context_length', 0),
                'available_context': state.get('available_context_length', 0),
                'utilization_percentage': state.get('context_utilization_percentage', 0),
                'token_usage': state.get('token_usage_stats', {}),
                'performance_metrics': state.get('performance_metrics', {})
            },
            'active_memory_items': items,
            'tasks': tasks,
            'agents': agents,
            'quarantined_contexts': quarantined
        }

        return json.dumps(export_data, indent=2, default=str)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
