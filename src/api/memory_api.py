"""
Memory Management API Endpoints
Provides endpoints for interacting with the comprehensive memory management system
"""

import os
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.memory_manager import LLMMemoryManager, TaskStatus

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/memory-manager", tags=["memory-manager"])


# Pydantic models for requests
class AddMemoryItemRequest(BaseModel):
    content: str
    content_type: str
    priority: int = 0
    metadata: Optional[Dict] = None
    auto_compress: bool = True


class RetrieveMemoryRequest(BaseModel):
    item_id: str
    decompress: bool = True


class SearchMemoryRequest(BaseModel):
    query: str
    limit: int = 5
    content_types: Optional[List[str]] = None


class CreateTaskRequest(BaseModel):
    title: str
    description: str = ""
    priority: int = 0
    assigned_agent: Optional[str] = None
    dependencies: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class UpdateTaskRequest(BaseModel):
    task_id: str
    status: Optional[str] = None
    progress: Optional[int] = None
    metadata: Optional[Dict] = None


class CreateAgentRequest(BaseModel):
    agent_name: str
    agent_type: str
    capabilities: List[str]
    assigned_tools: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class QuarantineContextRequest(BaseModel):
    content: str
    context_type: str
    reason: str
    metadata: Optional[Dict] = None


class HILRequest(BaseModel):
    question: str
    priority: int = 0
    metadata: Optional[Dict] = None


class HILResponseRequest(BaseModel):
    hil_id: str
    response: str


class TokenUsageRequest(BaseModel):
    input_tokens: int
    output_tokens: int


# Helper function to get memory manager
def get_memory_manager() -> LLMMemoryManager:
    """Get or create memory manager instance"""
    return LLMMemoryManager(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "memory_db"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgres"),
        model_name=os.getenv("PRIMARY_LLM_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        memory_approach=os.getenv("MEMORY_APPROACH", "external_llm"),
        use_slm=os.getenv("USE_SLM", "false").lower() == "true",
        use_lora_finetuning=os.getenv("USE_LORA_FINETUNING", "false").lower() == "true",
        slm_model_name=os.getenv("SLM_MODEL_NAME", "microsoft/phi-2")
    )


# API Endpoints

@router.post("/items/add")
async def add_memory_item(request: AddMemoryItemRequest):
    """Add a new memory item"""
    try:
        manager = get_memory_manager()
        item_id = manager.add_memory_item(
            content=request.content,
            content_type=request.content_type,
            priority=request.priority,
            metadata=request.metadata,
            auto_compress=request.auto_compress
        )
        manager.close()

        return {
            "success": True,
            "item_id": item_id,
            "message": "Memory item added successfully"
        }
    except Exception as e:
        logger.error(f"Error adding memory item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/items/retrieve")
async def retrieve_memory_item(request: RetrieveMemoryRequest):
    """Retrieve a memory item by ID"""
    try:
        manager = get_memory_manager()
        item = manager.retrieve_memory_item(
            item_id=request.item_id,
            decompress=request.decompress
        )
        manager.close()

        if not item:
            raise HTTPException(status_code=404, detail="Memory item not found")

        return {
            "success": True,
            "item": item
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving memory item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/items/search")
async def search_memory(request: SearchMemoryRequest):
    """Semantic search through memory items"""
    try:
        manager = get_memory_manager()
        results = manager.search_memory_semantic(
            query=request.query,
            limit=request.limit,
            content_types=request.content_types
        )
        manager.close()

        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state")
async def get_memory_state():
    """Get current memory state"""
    try:
        manager = get_memory_manager()
        state = manager.get_memory_state()
        manager.close()

        return {
            "success": True,
            "state": state
        }
    except Exception as e:
        logger.error(f"Error getting memory state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/export")
async def export_memory_state():
    """Export complete memory state as JSON"""
    try:
        manager = get_memory_manager()
        state_json = manager.export_memory_state_json()
        manager.close()

        return {
            "success": True,
            "state_json": state_json
        }
    except Exception as e:
        logger.error(f"Error exporting memory state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context")
async def get_working_context(max_tokens: Optional[int] = None):
    """Get working context"""
    try:
        manager = get_memory_manager()
        context = manager.get_working_context(max_tokens=max_tokens)
        manager.close()

        return {
            "success": True,
            "context": context
        }
    except Exception as e:
        logger.error(f"Error getting working context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hygiene")
async def perform_chat_hygiene():
    """Perform chat hygiene"""
    try:
        manager = get_memory_manager()
        manager.perform_chat_hygiene()
        manager.close()

        return {
            "success": True,
            "message": "Chat hygiene completed successfully"
        }
    except Exception as e:
        logger.error(f"Error performing chat hygiene: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Task Management Endpoints

@router.post("/tasks/create")
async def create_task(request: CreateTaskRequest):
    """Create a new task"""
    try:
        manager = get_memory_manager()
        task_id = manager.create_task(
            title=request.title,
            description=request.description,
            priority=request.priority,
            assigned_agent=request.assigned_agent,
            dependencies=request.dependencies,
            metadata=request.metadata
        )
        manager.close()

        return {
            "success": True,
            "task_id": task_id,
            "message": "Task created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/update")
async def update_task(request: UpdateTaskRequest):
    """Update a task"""
    try:
        manager = get_memory_manager()

        status = TaskStatus(request.status) if request.status else None

        manager.update_task(
            task_id=request.task_id,
            status=status,
            progress=request.progress,
            metadata=request.metadata
        )
        manager.close()

        return {
            "success": True,
            "message": "Task updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def get_tasks(status: Optional[str] = None):
    """Get tasks"""
    try:
        manager = get_memory_manager()

        task_status = TaskStatus(status) if status else None
        tasks = manager.get_tasks(status=task_status)
        manager.close()

        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks)
        }
    except Exception as e:
        logger.error(f"Error getting tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Agent Management Endpoints

@router.post("/agents/create")
async def create_agent(request: CreateAgentRequest):
    """Create a new agent"""
    try:
        manager = get_memory_manager()
        agent_id = manager.create_agent(
            agent_name=request.agent_name,
            agent_type=request.agent_type,
            capabilities=request.capabilities,
            assigned_tools=request.assigned_tools,
            metadata=request.metadata
        )
        manager.close()

        return {
            "success": True,
            "agent_id": agent_id,
            "message": "Agent created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def get_agents(agent_type: Optional[str] = None):
    """Get agents"""
    try:
        manager = get_memory_manager()
        agents = manager.get_agents(agent_type=agent_type)
        manager.close()

        return {
            "success": True,
            "agents": agents,
            "count": len(agents)
        }
    except Exception as e:
        logger.error(f"Error getting agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Context Quarantine Endpoints

@router.post("/quarantine")
async def quarantine_context(request: QuarantineContextRequest):
    """Quarantine a context"""
    try:
        manager = get_memory_manager()
        thread_id = manager.quarantine_context(
            content=request.content,
            context_type=request.context_type,
            reason=request.reason,
            metadata=request.metadata
        )
        manager.close()

        return {
            "success": True,
            "thread_id": thread_id,
            "message": "Context quarantined successfully"
        }
    except Exception as e:
        logger.error(f"Error quarantining context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quarantine")
async def get_quarantined_contexts():
    """Get quarantined contexts"""
    try:
        manager = get_memory_manager()
        contexts = manager.get_quarantined_contexts()
        manager.close()

        return {
            "success": True,
            "contexts": contexts,
            "count": len(contexts)
        }
    except Exception as e:
        logger.error(f"Error getting quarantined contexts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Human-in-the-Loop Endpoints

@router.post("/hil/request")
async def request_human_input(request: HILRequest):
    """Request human input"""
    try:
        manager = get_memory_manager()
        hil_id = manager.request_human_input(
            question=request.question,
            priority=request.priority,
            metadata=request.metadata
        )
        manager.close()

        return {
            "success": True,
            "hil_id": hil_id,
            "message": "Human input requested successfully"
        }
    except Exception as e:
        logger.error(f"Error requesting human input: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hil/respond")
async def provide_human_response(request: HILResponseRequest):
    """Provide human response"""
    try:
        manager = get_memory_manager()
        manager.provide_human_response(
            hil_id=request.hil_id,
            response=request.response
        )
        manager.close()

        return {
            "success": True,
            "message": "Human response recorded successfully"
        }
    except Exception as e:
        logger.error(f"Error providing human response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hil/pending")
async def get_pending_hil_requests():
    """Get pending HIL requests"""
    try:
        manager = get_memory_manager()
        requests = manager.get_pending_hil_requests()
        manager.close()

        return {
            "success": True,
            "requests": requests,
            "count": len(requests)
        }
    except Exception as e:
        logger.error(f"Error getting pending HIL requests: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Token Tracking Endpoints

@router.post("/tokens/track")
async def track_token_usage(request: TokenUsageRequest):
    """Track token usage"""
    try:
        manager = get_memory_manager()
        manager.track_token_usage(
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens
        )
        manager.close()

        return {
            "success": True,
            "message": "Token usage tracked successfully"
        }
    except Exception as e:
        logger.error(f"Error tracking token usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
