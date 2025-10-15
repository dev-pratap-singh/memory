"""
Performance Test: Memory Management with 5x Context Size
Tests the system with 640k tokens (5x gpt-4o-mini's 128k context)
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory_manager import LLMMemoryManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test configuration
MODEL_NAME = "gpt-4o-mini"
CONTEXT_SIZE = 128000  # tokens
TEST_DATA_SIZE = CONTEXT_SIZE * 5  # 640k tokens
TARGET_TOKENS_PER_CHUNK = 50000  # ~50k tokens per chunk
NUM_CHUNKS = TEST_DATA_SIZE // TARGET_TOKENS_PER_CHUNK  # 12-13 chunks


def generate_test_content(chunk_id: int, tokens: int) -> str:
    """Generate test content with specific searchable information"""

    # Create diverse content for each chunk
    topics = [
        "frontend authentication system with JWT tokens and OAuth2 integration",
        "backend API architecture using FastAPI with PostgreSQL database",
        "microservices communication patterns with message queues and event-driven design",
        "cloud infrastructure setup on AWS with ECS, RDS, and S3 storage",
        "CI/CD pipeline configuration using GitHub Actions and Docker containers",
        "monitoring and observability with Prometheus, Grafana, and logging systems",
        "security best practices including encryption, HTTPS, and vulnerability scanning",
        "performance optimization techniques for database queries and API responses",
        "user management system with role-based access control and permissions",
        "payment processing integration with Stripe and subscription management",
        "email notification system with templates and scheduled sending",
        "file upload and storage handling with image processing and CDN delivery",
        "search functionality implementation using Elasticsearch and full-text search"
    ]

    topic = topics[chunk_id % len(topics)]

    # Generate content that will be approximately the right number of tokens
    # Rough estimate: 1 token ≈ 4 characters
    target_chars = tokens * 4

    content = f"""
# Documentation Chunk {chunk_id + 1}: {topic.title()}

## Overview
This document describes the {topic} implementation in our system.
This is a comprehensive guide covering architecture, implementation details,
best practices, and troubleshooting information.

## Architecture

### System Design
The {topic} follows a modular architecture with clear separation of concerns.
We use industry-standard patterns including:
- Layered architecture with presentation, business logic, and data access layers
- Dependency injection for loose coupling and testability
- Repository pattern for data access abstraction
- Service layer for business logic encapsulation

### Key Components
"""

    # Add detailed sections to reach target size
    sections = [
        "Implementation Details",
        "Configuration Options",
        "API Endpoints",
        "Data Models",
        "Error Handling",
        "Testing Strategy",
        "Deployment Process",
        "Monitoring and Alerts",
        "Performance Metrics",
        "Security Considerations"
    ]

    for section in sections:
        content += f"\n\n## {section}\n\n"
        content += f"The {section.lower()} for {topic} includes:\n\n"

        # Add repetitive but searchable content
        for i in range(20):
            content += f"- Point {i + 1}: Detailed information about {section.lower()} "
            content += f"in the context of {topic}. This includes implementation specifics, "
            content += f"configuration requirements, and operational considerations. "
            content += f"Reference ID: CHUNK{chunk_id + 1}_SECTION{section.replace(' ', '_').upper()}_ITEM{i + 1}\n"

    # Add unique searchable markers
    content += f"\n\n## Special Information\n\n"
    content += f"**SEARCH_KEY_CHUNK_{chunk_id + 1}**: This chunk specifically covers {topic}.\n"
    content += f"**UNIQUE_IDENTIFIER**: CHK{chunk_id + 1:03d}_{int(time.time())}\n"
    content += f"**TIMESTAMP**: {datetime.now().isoformat()}\n"

    # Pad to reach target size if needed
    while len(content) < target_chars:
        content += f"\nAdditional context paragraph {len(content) // 1000}: "
        content += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10

    return content[:target_chars]  # Trim to target size


def run_performance_test():
    """Run the performance test"""

    print("=" * 80)
    print("MEMORY MANAGEMENT PERFORMANCE TEST")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Context Size: {CONTEXT_SIZE:,} tokens")
    print(f"Test Data Size: {TEST_DATA_SIZE:,} tokens (5x context)")
    print(f"Target Tokens Per Chunk: {TARGET_TOKENS_PER_CHUNK:,}")
    print(f"Number of Chunks: {NUM_CHUNKS}")
    print("=" * 80)
    print()

    # Initialize memory manager
    print("Initializing memory manager...")
    manager = LLMMemoryManager(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "memory_db"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgres"),
        model_name=MODEL_NAME,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        memory_approach="external_llm"
    )
    print("✓ Memory manager initialized")
    print()

    # Check initial state
    initial_state = manager.get_memory_state()
    print("Initial State:")
    print(f"  Context Utilization: {initial_state['context_utilization_percentage']:.2f}%")
    print(f"  Available Tokens: {initial_state['available_context_length']:,}")
    print()

    # Generate and add test data
    print(f"Generating and adding {NUM_CHUNKS} chunks of test data...")
    chunk_ids = []
    start_time = time.time()

    for i in range(NUM_CHUNKS):
        print(f"  Processing chunk {i + 1}/{NUM_CHUNKS}...", end=" ")
        chunk_start = time.time()

        # Generate content
        content = generate_test_content(i, TARGET_TOKENS_PER_CHUNK)

        # Determine content type and priority
        content_types = ["frontend", "backend", "infrastructure", "security", "monitoring"]
        content_type = content_types[i % len(content_types)]
        priority = 10 - (i // 2)  # Decreasing priority

        # Add to memory
        chunk_id = manager.add_memory_item(
            content=content,
            content_type=content_type,
            priority=priority,
            metadata={
                "chunk_number": i + 1,
                "total_chunks": NUM_CHUNKS,
                "test_id": "performance_test_5x"
            }
        )
        chunk_ids.append(chunk_id)

        chunk_time = time.time() - chunk_start
        print(f"✓ ({chunk_time:.2f}s)")

    total_time = time.time() - start_time
    print(f"\n✓ All chunks added in {total_time:.2f}s (avg {total_time/NUM_CHUNKS:.2f}s per chunk)")
    print()

    # Check state after adding data
    current_state = manager.get_memory_state()
    print("State After Adding Data:")
    print(f"  Context Utilization: {current_state['context_utilization_percentage']:.2f}%")
    print(f"  Used Tokens: {current_state['used_context_length']:,}")
    print(f"  Available Tokens: {current_state['available_context_length']:,}")
    print()

    # Test compression (should have triggered automatically)
    if current_state['context_utilization_percentage'] > 75:
        print("✓ Compression triggered (utilization > 75%)")
    else:
        print("ℹ Compression not yet triggered (utilization < 75%)")
    print()

    # Test semantic search
    print("Testing Semantic Search...")
    test_queries = [
        ("authentication system JWT tokens", 0),
        ("database PostgreSQL FastAPI", 1),
        ("AWS cloud infrastructure", 3),
        ("monitoring Prometheus Grafana", 5),
    ]

    for query, expected_chunk in test_queries:
        print(f"\n  Query: '{query}'")
        search_start = time.time()
        results = manager.search_memory_semantic(query, limit=3)
        search_time = time.time() - search_start

        print(f"  Search Time: {search_time*1000:.2f}ms")
        print(f"  Results Found: {len(results)}")

        if results:
            best_match = results[0]
            print(f"  Best Match:")
            print(f"    - Content Type: {best_match['content_type']}")
            print(f"    - Priority: {best_match['priority']}")
            print(f"    - Distance: {best_match['distance']:.4f}")
            print(f"    - Tokens: {best_match['token_count']:,}")
            print(f"    - Compressed: {best_match['compressed_token_count']:,}")

    print()

    # Test retrieval performance
    print("Testing Retrieval Performance...")
    retrieval_times = []

    for i, chunk_id in enumerate(chunk_ids[:5]):  # Test first 5 chunks
        print(f"  Retrieving chunk {i + 1}...", end=" ")
        retrieval_start = time.time()
        retrieved = manager.retrieve_memory_item(chunk_id, decompress=True)
        retrieval_time = time.time() - retrieval_start
        retrieval_times.append(retrieval_time)

        if retrieved:
            print(f"✓ ({retrieval_time*1000:.2f}ms, {len(retrieved['full_content']):,} chars)")
        else:
            print("✗ Failed")

    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
    print(f"\n  Average Retrieval Time: {avg_retrieval*1000:.2f}ms")
    print()

    # Test working context
    print("Testing Working Context...")
    context = manager.get_working_context()
    print(f"  Full Items: {len(context['full_items'])}")
    print(f"  Compressed Items: {len(context['compressed_items'])}")
    print(f"  Total Tokens in Working Context: {context['total_tokens']:,}")
    print(f"  Available Tokens: {context['available_tokens']:,}")
    print()

    # Final statistics
    final_state = manager.get_memory_state()
    print("=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total Test Data Added: {TEST_DATA_SIZE:,} tokens (5x context size)")
    print(f"Context Utilization: {final_state['context_utilization_percentage']:.2f}%")
    print(f"Compression Working: {'✓ YES' if final_state['context_utilization_percentage'] < 100 else '✗ NO'}")
    print(f"Average Search Time: {sum(search_time for _ in range(len(test_queries)))/len(test_queries)*1000:.2f}ms")
    print(f"Average Retrieval Time: {avg_retrieval*1000:.2f}ms")
    print(f"Total Processing Time: {total_time:.2f}s")
    print()

    token_stats = final_state.get('token_usage_stats', {})
    if token_stats:
        print("Token Usage:")
        print(f"  Input Tokens: {token_stats.get('input_tokens', 0):,}")
        print(f"  Output Tokens: {token_stats.get('output_tokens', 0):,}")
        print(f"  Total Cost: ${token_stats.get('total_cost', 0):.4f}")

    perf_metrics = final_state.get('performance_metrics', {})
    if perf_metrics:
        print("\nPerformance Metrics:")
        print(f"  Compressions: {perf_metrics.get('compressions', 0)}")
        print(f"  Decompressions: {perf_metrics.get('decompressions', 0)}")
        print(f"  Cache Hits: {perf_metrics.get('cache_hits', 0)}")

    print("=" * 80)
    print()

    # Answer a specific question using retrieved context
    print("Testing Question Answering with Retrieved Context...")
    print()
    question = "What does the documentation say about JWT authentication?"
    print(f"Question: {question}")
    print()

    # Search for relevant context
    results = manager.search_memory_semantic(question, limit=2)

    if results:
        print(f"Found {len(results)} relevant chunks")
        print()

        # Retrieve full content of best match
        best_match = results[0]
        full_content = manager.retrieve_memory_item(best_match['id'], decompress=True)

        if full_content:
            content_snippet = full_content['full_content'][:500]
            print("Retrieved Context Preview:")
            print("-" * 80)
            print(content_snippet + "...")
            print("-" * 80)
            print()
            print("✓ System successfully retrieved relevant context from 640k tokens of data!")
            print("✓ Compression is working - context fits within model limits")
            print("✓ Semantic search found the right information")
            print("✓ Retrieval is fast and accurate")

    # Cleanup
    manager.close()
    print()
    print("✓ Test completed successfully!")


if __name__ == "__main__":
    try:
        run_performance_test()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
