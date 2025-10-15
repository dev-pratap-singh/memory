"""
FastAPI Application
REST API for the Dual-Model Memory System
"""

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.orchestrator import get_orchestrator
from src.database.connection import get_db_manager
from src.utils.config_loader import get_config
from src.api.memory_api import router as memory_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dual-Model Memory System",
    description="Long-term memory system for LLM interactions using a dual-model approach",
    version="1.0.0",
)

# Get configuration
config = get_config()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(memory_router)


# Pydantic models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    user_id: Optional[str] = None
    use_memory: bool = True
    store_conversation: bool = True


class MemoryQueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    limit: int = 5


class TrainingRequest(BaseModel):
    force: bool = False


# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Dual-Model Memory System...")

    # Initialize database
    db_manager = get_db_manager()
    if db_manager.test_connection():
        logger.info("Database connection successful")
        db_manager.create_tables()
    else:
        logger.error("Database connection failed!")

    # Initialize orchestrator
    orchestrator = get_orchestrator()
    logger.info("Orchestrator initialized")

    logger.info("System startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    db_manager = get_db_manager()
    db_manager.close()


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Dual-Model Memory System",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_manager = get_db_manager()
    db_healthy = db_manager.test_connection()

    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "database": "connected" if db_healthy else "disconnected",
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint - Process conversation through the system

    Args:
        request: Chat request with messages and settings

    Returns:
        Response from Primary LLM with memory integration
    """
    try:
        orchestrator = get_orchestrator()

        # Convert Pydantic models to dicts
        messages = [msg.dict() for msg in request.messages]

        # Process conversation
        result = orchestrator.process_conversation(
            messages=messages,
            user_id=request.user_id,
            use_memory=request.use_memory,
            store_conversation=request.store_conversation,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search")
async def search_memory(request: MemoryQueryRequest):
    """
    Search memory endpoint

    Args:
        request: Query request

    Returns:
        Search results from memory
    """
    try:
        orchestrator = get_orchestrator()

        results = orchestrator.query_memory(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit,
        )

        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        return results

    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/context")
async def get_context(user_id: Optional[str] = None, limit: int = 5):
    """
    Get recent context

    Args:
        user_id: User identifier
        limit: Number of recent conversations

    Returns:
        Recent conversation context
    """
    try:
        from src.tools.memory_tools import get_memory_tools

        tools = get_memory_tools()
        context = tools.get_context(num_recent=limit, user_id=user_id)

        return context

    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/summary")
async def get_summary(
    time_range: str = "week",
    user_id: Optional[str] = None,
):
    """
    Get memory summary

    Args:
        time_range: Time range (day, week, month, all)
        user_id: User identifier

    Returns:
        Memory summary statistics
    """
    try:
        from src.tools.memory_tools import get_memory_tools

        tools = get_memory_tools()
        summary = tools.summarize_memory(time_range=time_range, user_id=user_id)

        return summary

    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/trigger")
async def trigger_training(request: TrainingRequest):
    """
    Trigger SLM training

    Args:
        request: Training request with force flag

    Returns:
        Training task status
    """
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.trigger_training(force=request.force)

        return result

    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get system statistics

    Returns:
        System statistics and metrics
    """
    try:
        orchestrator = get_orchestrator()
        stats = orchestrator.get_system_stats()

        return stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/model-info")
async def get_model_info():
    """
    Get current and next model information

    Returns:
        Information about current and recommended next model
    """
    try:
        from src.utils.config_loader import get_model_info

        info = get_model_info(config)
        return info

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level=config.api.log_level,
    )
