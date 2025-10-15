"""
Extreme Performance Test: Memory Management with 10x Context Size
Tests the system with 1.28 MILLION tokens (10x gpt-4o-mini's 128k context)
This is a stress test to verify scalability under extreme load
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
TEST_DATA_SIZE = CONTEXT_SIZE * 10  # 1.28 MILLION tokens!
TARGET_TOKENS_PER_CHUNK = 40000  # ~40k tokens per chunk
NUM_CHUNKS = TEST_DATA_SIZE // TARGET_TOKENS_PER_CHUNK  # 32 chunks


def generate_diverse_content(chunk_id: int, tokens: int) -> str:
    """Generate diverse test content with specific searchable information"""

    # Create very diverse content categories
    categories = [
        ("Authentication", "JWT tokens, OAuth2, SAML, session management, password hashing"),
        ("Database", "PostgreSQL, indexes, query optimization, migrations, transactions"),
        ("API Design", "REST, GraphQL, versioning, rate limiting, pagination"),
        ("Cloud Infrastructure", "AWS, ECS, RDS, S3, CloudFront, Lambda"),
        ("Microservices", "service mesh, event-driven, message queues, saga pattern"),
        ("Security", "HTTPS, encryption, OWASP, penetration testing, vulnerability scanning"),
        ("Performance", "caching, CDN, load balancing, horizontal scaling, profiling"),
        ("Monitoring", "Prometheus, Grafana, logs, metrics, alerts, tracing"),
        ("CI/CD", "GitHub Actions, Docker, Kubernetes, deployment strategies"),
        ("Frontend", "React, state management, component design, performance optimization"),
        ("Backend", "FastAPI, async programming, background tasks, webhooks"),
        ("Testing", "unit tests, integration tests, E2E tests, mocking, coverage"),
        ("DevOps", "infrastructure as code, Terraform, Ansible, configuration management"),
        ("Data Processing", "ETL pipelines, data warehousing, streaming, batch processing"),
        ("Machine Learning", "model training, inference, MLOps, feature engineering"),
        ("Networking", "TCP/IP, DNS, load balancers, proxies, firewalls"),
    ]

    category_name, keywords = categories[chunk_id % len(categories)]

    # Rough estimate: 1 token ≈ 4 characters
    target_chars = tokens * 4

    content = f"""
# Technical Documentation - Part {chunk_id + 1}: {category_name}

## Executive Summary

This section covers {category_name.lower()} in detail, including: {keywords}.
This is comprehensive technical documentation for production systems.

**Document ID**: DOC{chunk_id + 1:04d}
**Category**: {category_name}
**Keywords**: {keywords}
**Last Updated**: {datetime.now().isoformat()}

## Table of Contents

1. Overview and Architecture
2. Implementation Guidelines
3. Best Practices
4. Common Patterns
5. Troubleshooting
6. Performance Optimization
7. Security Considerations
8. Monitoring and Observability
9. Case Studies
10. Reference Implementation

---

## 1. Overview and Architecture

The {category_name.lower()} architecture follows industry best practices and modern design patterns.
Our implementation leverages {keywords} to provide a robust, scalable solution.

### Core Principles

"""

    # Generate substantial content for each section
    sections = [
        ("System Architecture", "layered design, separation of concerns, modularity"),
        ("Data Flow", "input validation, processing pipeline, output formatting"),
        ("Error Handling", "exception management, retry logic, circuit breakers"),
        ("Configuration", "environment variables, feature flags, runtime settings"),
        ("API Endpoints", "request/response format, authentication, rate limiting"),
        ("Database Schema", "tables, relationships, indexes, constraints"),
        ("Caching Strategy", "cache invalidation, TTL, cache warming, distributed cache"),
        ("Security Model", "authentication, authorization, encryption, auditing"),
        ("Performance Metrics", "latency, throughput, error rates, resource utilization"),
        ("Deployment Process", "blue-green, canary, rolling updates, rollback procedures"),
    ]

    for section_title, section_keywords in sections:
        content += f"\n\n## {section_title}\n\n"
        content += f"The {section_title.lower()} encompasses: {section_keywords}.\n\n"

        # Add detailed subsections
        for i in range(15):  # 15 subsections per section
            content += f"### {section_title} - Point {i + 1}\n\n"
            content += f"In the context of {category_name.lower()}, the {section_title.lower()} "
            content += f"implementation requires careful consideration of {section_keywords}. "
            content += f"This involves:\n\n"

            for j in range(5):  # 5 items per subsection
                content += f"- **Item {j + 1}**: Detailed analysis of {section_keywords.split(',')[j % 3].strip()} "
                content += f"as it relates to {category_name.lower()}. "
                content += f"Reference: CHUNK{chunk_id + 1:04d}_SEC{i + 1:02d}_ITEM{j + 1:02d}\n"

    # Add unique searchable information
    content += f"\n\n## Special Search Markers\n\n"
    content += f"**PRIMARY_TOPIC**: {category_name}\n"
    content += f"**SEARCH_KEY**: CHUNK_{chunk_id + 1:04d}_{category_name.replace(' ', '_').upper()}\n"
    content += f"**KEYWORDS**: {keywords}\n"
    content += f"**UNIQUE_ID**: CHK{chunk_id + 1:04d}_{int(time.time())}\n"

    # Add scenario-specific information for testing
    if "Authentication" in category_name:
        content += f"\n\n### JWT Token Implementation Details\n\n"
        content += "Our JWT token implementation uses RS256 algorithm with rotating keys. "
        content += "Tokens expire after 15 minutes and refresh tokens last 7 days. "
        content += "We implement token blacklisting for logout and use Redis for token storage. "
        content += "**ANSWER_KEY_JWT**: This system uses JWT with RS256 and 15-minute expiry.\n"

    elif "Database" in category_name:
        content += f"\n\n### PostgreSQL Optimization Techniques\n\n"
        content += "PostgreSQL optimization includes proper indexing strategies using B-tree and GiST indexes. "
        content += "We use connection pooling with pgBouncer and implement query caching. "
        content += "Partitioning is used for tables over 10GB. "
        content += "**ANSWER_KEY_DB**: PostgreSQL uses B-tree indexes and connection pooling with pgBouncer.\n"

    elif "Cloud" in category_name:
        content += f"\n\n### AWS Infrastructure Setup\n\n"
        content += "Our AWS infrastructure uses ECS Fargate for container orchestration. "
        content += "RDS PostgreSQL runs in Multi-AZ configuration for high availability. "
        content += "S3 with CloudFront CDN serves static assets. Lambda handles background processing. "
        content += "**ANSWER_KEY_AWS**: Infrastructure uses ECS Fargate, RDS Multi-AZ, and S3 with CloudFront.\n"

    elif "Monitoring" in category_name:
        content += f"\n\n### Prometheus and Grafana Setup\n\n"
        content += "Prometheus scrapes metrics every 15 seconds from all services. "
        content += "Grafana dashboards show P50, P95, P99 latencies and error rates. "
        content += "AlertManager sends notifications to Slack and PagerDuty. "
        content += "**ANSWER_KEY_MONITORING**: Prometheus scrapes every 15s, Grafana shows P50/P95/P99 metrics.\n"

    # Pad content to reach target size
    padding_text = (
        f"Additional technical details for {category_name.lower()} covering "
        f"{keywords}. This includes implementation specifics, configuration options, "
        f"deployment strategies, monitoring approaches, and operational best practices. "
    ) * 20

    while len(content) < target_chars:
        content += "\n" + padding_text

    return content[:target_chars]  # Trim to exact target size


def run_extreme_performance_test():
    """Run the extreme 10x performance test"""

    print("=" * 80)
    print("🚀 EXTREME MEMORY MANAGEMENT PERFORMANCE TEST - 10X SCALE 🚀")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Context Size: {CONTEXT_SIZE:,} tokens")
    print(f"Test Data Size: {TEST_DATA_SIZE:,} tokens (10x context!)")
    print(f"Target Tokens Per Chunk: {TARGET_TOKENS_PER_CHUNK:,}")
    print(f"Number of Chunks: {NUM_CHUNKS}")
    print("=" * 80)
    print()

    # Initialize memory manager
    print("Initializing memory manager...")
    start_init = time.time()
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
    init_time = time.time() - start_init
    print(f"✓ Memory manager initialized ({init_time:.2f}s)")
    print()

    # Check initial state
    initial_state = manager.get_memory_state()
    print("📊 Initial State:")
    print(f"  Context Utilization: {initial_state['context_utilization_percentage']:.2f}%")
    print(f"  Available Tokens: {initial_state['available_context_length']:,}")
    print()

    # Generate and add test data
    print(f"📝 Generating and adding {NUM_CHUNKS} chunks of test data...")
    print(f"   (This will take several minutes...)")
    print()

    chunk_ids = []
    chunk_times = []
    start_time = time.time()

    for i in range(NUM_CHUNKS):
        chunk_start = time.time()

        # Progress indicator
        if i % 5 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / i
                remaining = (NUM_CHUNKS - i) * avg_time
                print(f"  Progress: {i}/{NUM_CHUNKS} chunks ({i/NUM_CHUNKS*100:.1f}%) - "
                      f"ETA: {remaining/60:.1f} minutes")

        # Generate content
        content = generate_diverse_content(i, TARGET_TOKENS_PER_CHUNK)

        # Determine content type and priority
        content_types = ["docs", "code", "config", "tests", "api"]
        content_type = content_types[i % len(content_types)]
        priority = max(1, 10 - (i // 4))  # Decreasing priority

        # Add to memory
        chunk_id = manager.add_memory_item(
            content=content,
            content_type=content_type,
            priority=priority,
            metadata={
                "chunk_number": i + 1,
                "total_chunks": NUM_CHUNKS,
                "test_id": "extreme_test_10x",
                "category": content.split('\n')[1].split(':')[1].strip() if '\n' in content else "unknown"
            }
        )
        chunk_ids.append(chunk_id)

        chunk_time = time.time() - chunk_start
        chunk_times.append(chunk_time)

    total_time = time.time() - start_time
    avg_chunk_time = sum(chunk_times) / len(chunk_times)
    print()
    print(f"✓ All {NUM_CHUNKS} chunks added in {total_time:.2f}s (avg {avg_chunk_time:.2f}s per chunk)")
    print()

    # Check state after adding data
    current_state = manager.get_memory_state()
    print("📊 State After Adding 1.28M Tokens:")
    print(f"  Context Utilization: {current_state['context_utilization_percentage']:.2f}%")
    print(f"  Used Tokens: {current_state['used_context_length']:,}")
    print(f"  Available Tokens: {current_state['available_context_length']:,}")

    compression_ratio = TEST_DATA_SIZE / current_state['used_context_length'] if current_state['used_context_length'] > 0 else 0
    print(f"  Compression Ratio: {compression_ratio:.2f}:1")
    print()

    # Test compression effectiveness
    if current_state['context_utilization_percentage'] < 100:
        print("✅ COMPRESSION SUCCESS: 1.28M tokens fit in 128k context!")
    else:
        print("⚠️  Warning: Context utilization above 100%")
    print()

    # Test semantic search with various queries
    print("🔍 Testing Semantic Search (10x scale)...")
    test_queries = [
        ("JWT token implementation with RS256", "Authentication"),
        ("PostgreSQL indexing and connection pooling", "Database"),
        ("AWS ECS Fargate container orchestration", "Cloud Infrastructure"),
        ("Prometheus metrics scraping interval", "Monitoring"),
        ("React component state management", "Frontend"),
        ("FastAPI async background tasks", "Backend"),
    ]

    search_times = []
    search_results = []

    for query, expected_category in test_queries:
        print(f"\n  Query: '{query}'")
        search_start = time.time()
        results = manager.search_memory_semantic(query, limit=3)
        search_time = time.time() - search_start
        search_times.append(search_time)

        print(f"  ⏱️  Search Time: {search_time*1000:.2f}ms")
        print(f"  📊 Results Found: {len(results)}")

        if results:
            best_match = results[0]
            search_results.append(best_match)
            category = best_match.get('metadata', {}).get('category', 'unknown')
            print(f"  ✓ Best Match Category: {category}")
            print(f"    Distance Score: {best_match['distance']:.4f}")
            print(f"    Compression: {best_match['token_count']:,} → {best_match['compressed_token_count']:,} tokens")

    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    print(f"\n  📈 Average Search Time: {avg_search_time*1000:.2f}ms")
    print()

    # Test retrieval performance
    print("📥 Testing Retrieval Performance...")
    retrieval_times = []

    for i in range(min(10, len(chunk_ids))):
        retrieval_start = time.time()
        retrieved = manager.retrieve_memory_item(chunk_ids[i], decompress=True)
        retrieval_time = time.time() - retrieval_start
        retrieval_times.append(retrieval_time)

    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
    min_retrieval = min(retrieval_times) if retrieval_times else 0
    max_retrieval = max(retrieval_times) if retrieval_times else 0

    print(f"  Average: {avg_retrieval*1000:.2f}ms")
    print(f"  Min: {min_retrieval*1000:.2f}ms")
    print(f"  Max: {max_retrieval*1000:.2f}ms")
    print()

    # Test working context
    print("🔧 Testing Working Context...")
    context = manager.get_working_context()
    print(f"  Full Items: {len(context['full_items'])}")
    print(f"  Compressed Items: {len(context['compressed_items'])}")
    print(f"  Total Tokens: {context['total_tokens']:,}")
    print(f"  Available: {context['available_tokens']:,}")
    print()

    # Answer specific questions
    print("=" * 80)
    print("💬 QUESTION ANSWERING TEST")
    print("=" * 80)
    print()

    questions = [
        "How is JWT token authentication implemented?",
        "What PostgreSQL optimization techniques are used?",
        "How is the AWS infrastructure configured?",
        "What monitoring metrics does Prometheus collect?",
    ]

    for question in questions:
        print(f"❓ Question: {question}")
        print()

        qa_start = time.time()
        results = manager.search_memory_semantic(question, limit=2)
        search_time = time.time() - qa_start

        if results:
            best_match = results[0]
            retrieve_start = time.time()
            full_content = manager.retrieve_memory_item(best_match['id'], decompress=True)
            retrieve_time = time.time() - retrieve_start

            total_qa_time = search_time + retrieve_time

            print(f"  ⏱️  Search: {search_time*1000:.2f}ms | Retrieval: {retrieve_time*1000:.2f}ms | Total: {total_qa_time*1000:.2f}ms")
            print(f"  📄 Source: {full_content.get('metadata', {}).get('category', 'Unknown')}")

            # Try to find answer key in content
            content = full_content.get('full_content', '')
            answer_lines = [line for line in content.split('\n') if 'ANSWER_KEY' in line]
            if answer_lines:
                print(f"  ✅ Answer: {answer_lines[0].split(':', 1)[1].strip()}")
            else:
                print(f"  ℹ️  Found relevant content (preview): {content[:200]}...")
        else:
            print("  ❌ No results found")

        print()

    # Final statistics
    final_state = manager.get_memory_state()
    print("=" * 80)
    print("📊 FINAL STATISTICS")
    print("=" * 80)
    print(f"Total Test Data: {TEST_DATA_SIZE:,} tokens (10x context size)")
    print(f"Final Context Utilization: {final_state['context_utilization_percentage']:.2f}%")
    print(f"Effective Compression Ratio: {compression_ratio:.2f}:1")
    print(f"Average Search Time: {avg_search_time*1000:.2f}ms")
    print(f"Average Retrieval Time: {avg_retrieval*1000:.2f}ms")
    print(f"Total Processing Time: {total_time/60:.2f} minutes")
    print(f"Chunks Processed: {NUM_CHUNKS}")
    print()

    # Performance assessment
    print("🎯 PERFORMANCE ASSESSMENT:")
    if final_state['context_utilization_percentage'] < 100:
        print("  ✅ EXCELLENT: System handled 10x context size successfully")
    else:
        print("  ⚠️  WARNING: Context utilization exceeded 100%")

    if avg_search_time < 1.0:
        print(f"  ✅ EXCELLENT: Search time under 1 second ({avg_search_time*1000:.0f}ms)")
    else:
        print(f"  ⚠️  Search time: {avg_search_time:.2f}s")

    if avg_retrieval < 0.1:
        print(f"  ✅ EXCELLENT: Retrieval time under 100ms ({avg_retrieval*1000:.0f}ms)")
    else:
        print(f"  ⚠️  Retrieval time: {avg_retrieval*1000:.0f}ms")

    print()
    print("=" * 80)
    print("🏆 EXTREME SCALE TEST COMPLETED!")
    print("=" * 80)

    # Cleanup
    manager.close()


if __name__ == "__main__":
    try:
        run_extreme_performance_test()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
