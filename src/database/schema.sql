-- Database Schema for Advanced Memory Management System
-- Comprehensive schema including both existing tables and new memory management tables

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- EXISTING TABLES (from original system)
-- ============================================================================

-- Conversation storage with vector embeddings
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    messages JSONB NOT NULL,
    summary TEXT,
    importance_score FLOAT DEFAULT 0.5,
    topics TEXT[],
    embedding vector(384),
    user_id VARCHAR(100),
    extra_metadata JSONB,
    is_compressed BOOLEAN DEFAULT false,
    is_archived BOOLEAN DEFAULT false
);

-- Extracted facts with embeddings
CREATE TABLE IF NOT EXISTS memory_facts (
    fact_id VARCHAR(100) PRIMARY KEY,
    conversation_id VARCHAR(100) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    fact TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.8,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    embedding vector(384),
    tags TEXT[],
    source_type VARCHAR(50) DEFAULT 'extraction'
);

-- User preferences and instructions
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(100) PRIMARY KEY,
    preferences JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training history for Small Language Model
CREATE TABLE IF NOT EXISTS training_history (
    training_id VARCHAR(100) PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    num_conversations INTEGER,
    adapter_path TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    metrics JSONB,
    error_message TEXT
);

-- ============================================================================
-- NEW MEMORY MANAGEMENT TABLES
-- ============================================================================

-- Memory items table (stores compressed and full content)
CREATE TABLE IF NOT EXISTS memory_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    full_content TEXT NOT NULL,
    compressed_summary TEXT NOT NULL,
    embedding vector(1536),
    token_count INTEGER NOT NULL,
    compressed_token_count INTEGER NOT NULL,
    compression_ratio FLOAT,
    is_active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Memory state tracking table
CREATE TABLE IF NOT EXISTS memory_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    total_context_length INTEGER NOT NULL,
    used_context_length INTEGER NOT NULL,
    available_context_length INTEGER NOT NULL,
    context_utilization_percentage FLOAT,
    active_items JSONB DEFAULT '[]'::jsonb,
    compressed_items JSONB DEFAULT '[]'::jsonb,
    token_usage_stats JSONB DEFAULT '{}'::jsonb,
    performance_metrics JSONB DEFAULT '{}'::jsonb,
    state_snapshot TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Task management table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    assigned_agent VARCHAR(255),
    dependencies JSONB DEFAULT '[]'::jsonb,
    progress INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Agent registry table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    capabilities JSONB DEFAULT '[]'::jsonb,
    assigned_tools JSONB DEFAULT '[]'::jsonb,
    performance_stats JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Context quarantine table (isolated contexts)
CREATE TABLE IF NOT EXISTS context_quarantine (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    context_type VARCHAR(100) NOT NULL,
    isolated_content TEXT NOT NULL,
    reason_for_isolation TEXT,
    token_count INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Chat history table
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    is_compressed BOOLEAN DEFAULT false,
    parent_message_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Tool usage tracking
CREATE TABLE IF NOT EXISTS tool_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    tool_name VARCHAR(255) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Human-in-the-loop interactions
CREATE TABLE IF NOT EXISTS hil_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    question TEXT NOT NULL,
    response TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    answered_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Existing tables indexes
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_archived ON conversations(is_archived);
CREATE INDEX IF NOT EXISTS idx_conversations_importance ON conversations(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_embedding ON conversations USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_facts_conversation ON memory_facts(conversation_id);
CREATE INDEX IF NOT EXISTS idx_facts_created ON memory_facts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_embedding ON memory_facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_training_status ON training_history(status);
CREATE INDEX IF NOT EXISTS idx_training_started ON training_history(started_at DESC);

-- New tables indexes
CREATE INDEX IF NOT EXISTS idx_memory_items_session ON memory_items(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_active ON memory_items(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_memory_items_type ON memory_items(content_type);
CREATE INDEX IF NOT EXISTS idx_memory_items_embedding ON memory_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_memory_state_session ON memory_state(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_agents_session ON agents(session_id);
CREATE INDEX IF NOT EXISTS idx_context_quarantine_session ON context_quarantine(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON tool_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_hil_session ON hil_interactions(session_id);

-- ============================================================================
-- TRIGGERS AND FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates (existing tables)
DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_user_preferences_updated_at ON user_preferences;
CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Triggers for automatic timestamp updates (new tables)
DROP TRIGGER IF EXISTS update_memory_items_updated_at ON memory_items;
CREATE TRIGGER update_memory_items_updated_at BEFORE UPDATE ON memory_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_memory_state_updated_at ON memory_state;
CREATE TRIGGER update_memory_state_updated_at BEFORE UPDATE ON memory_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;
CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_agents_updated_at ON agents;
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_context_quarantine_updated_at ON context_quarantine;
CREATE TRIGGER update_context_quarantine_updated_at BEFORE UPDATE ON context_quarantine
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
