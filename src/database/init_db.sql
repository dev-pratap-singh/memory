-- Initialize PostgreSQL database with pgvector extension
-- This script runs automatically when the database is first created

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    messages JSONB NOT NULL,
    summary TEXT,
    importance_score REAL DEFAULT 0.5,
    topics TEXT[],
    embedding vector(384),  -- Dimension for all-MiniLM-L6-v2
    user_id VARCHAR(100),
    metadata JSONB,
    is_compressed BOOLEAN DEFAULT FALSE,
    is_archived BOOLEAN DEFAULT FALSE
);

-- Create facts table
CREATE TABLE IF NOT EXISTS memory_facts (
    fact_id VARCHAR(100) PRIMARY KEY,
    conversation_id VARCHAR(100) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    fact TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(384),
    tags TEXT[],
    source_type VARCHAR(50) DEFAULT 'extraction'
);

-- Create user preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(100) PRIMARY KEY,
    preferences JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create training history table
CREATE TABLE IF NOT EXISTS training_history (
    training_id VARCHAR(100) PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    num_conversations INTEGER,
    adapter_path TEXT,
    status VARCHAR(50),
    metrics JSONB,
    error_message TEXT
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_importance ON conversations(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_conversation ON memory_facts(conversation_id);
CREATE INDEX IF NOT EXISTS idx_facts_created_at ON memory_facts(created_at DESC);

-- Create vector similarity indexes using HNSW
CREATE INDEX IF NOT EXISTS idx_conversations_embedding ON conversations
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_facts_embedding ON memory_facts
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Create GIN indexes for JSONB and array columns
CREATE INDEX IF NOT EXISTS idx_conversations_topics ON conversations USING GIN(topics);
CREATE INDEX IF NOT EXISTS idx_conversations_metadata ON conversations USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_facts_tags ON memory_facts USING GIN(tags);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create view for recent important conversations
CREATE OR REPLACE VIEW recent_important_conversations AS
SELECT
    conversation_id,
    created_at,
    summary,
    importance_score,
    topics
FROM conversations
WHERE
    is_archived = FALSE
    AND importance_score >= 0.7
ORDER BY created_at DESC
LIMIT 100;

-- Create view for conversation statistics
CREATE OR REPLACE VIEW conversation_stats AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as total_conversations,
    AVG(importance_score) as avg_importance,
    COUNT(DISTINCT user_id) as unique_users
FROM conversations
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Insert initial data or configuration if needed
-- (This is optional and can be customized)

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully with pgvector extension';
END $$;
