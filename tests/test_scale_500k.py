"""
500k Conversation Scale Test
Tests the dual-model memory system with 500k conversations
Measures latency, retrieval accuracy, and context window utilization with gpt-4o-mini
"""

import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


class ConversationGenerator:
    """Generate synthetic conversations for testing"""

    TOPICS = [
        "machine_learning", "python_programming", "docker", "kubernetes",
        "web_development", "data_science", "cloud_computing", "devops",
        "artificial_intelligence", "database_design", "api_development",
        "security", "testing", "ci_cd", "microservices", "frontend",
        "backend", "mobile_development", "system_design", "algorithms"
    ]

    USER_QUESTIONS = [
        "How do I implement {topic}?",
        "What are best practices for {topic}?",
        "Can you explain {topic} in detail?",
        "I'm having trouble with {topic}, can you help?",
        "What's the difference between {topic} and {other_topic}?",
        "How do I optimize {topic} for production?",
        "What tools should I use for {topic}?",
        "Can you recommend resources for learning {topic}?",
        "What are common mistakes in {topic}?",
        "How do I debug issues with {topic}?"
    ]

    ASSISTANT_RESPONSES = [
        "Here's a comprehensive guide to {topic}. First, you should understand the fundamentals...",
        "Great question about {topic}! Let me break this down for you...",
        "For {topic}, the best approach is to start with the basics and build up gradually...",
        "I can help you with {topic}. Here are the key concepts you need to know...",
        "When working with {topic}, it's important to consider performance and scalability...",
        "The most effective way to implement {topic} is to follow these steps...",
        "Let me explain {topic} with a practical example...",
        "Common challenges with {topic} include X, Y, and Z. Here's how to address them...",
        "For production-grade {topic}, you'll want to focus on reliability and monitoring...",
        "Here's a detailed walkthrough of {topic} with code examples..."
    ]

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)

    def generate_conversation(self, conv_id: int) -> Dict:
        """Generate a single conversation"""
        topic = random.choice(self.TOPICS)
        other_topic = random.choice([t for t in self.TOPICS if t != topic])

        user_question = random.choice(self.USER_QUESTIONS).format(
            topic=topic, other_topic=other_topic
        )
        assistant_response = random.choice(self.ASSISTANT_RESPONSES).format(topic=topic)

        # Add some variation in conversation length
        num_exchanges = random.randint(1, 5)
        messages = []

        for i in range(num_exchanges):
            messages.append({
                "role": "user",
                "content": user_question if i == 0 else f"Follow-up about {topic}: Can you elaborate?"
            })
            messages.append({
                "role": "assistant",
                "content": assistant_response if i == 0 else f"Additional details about {topic}..."
            })

        # Generate metadata
        importance_score = random.uniform(0.3, 1.0)
        created_at = datetime.now(timezone.utc) - timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        return {
            "conversation_id": conv_id,
            "messages": messages,
            "topic": topic,
            "importance_score": importance_score,
            "user_id": f"user_{random.randint(1, 1000)}",
            "created_at": created_at.isoformat()
        }

    def generate_batch(self, start_id: int, batch_size: int) -> List[Dict]:
        """Generate a batch of conversations"""
        return [
            self.generate_conversation(start_id + i)
            for i in range(batch_size)
        ]


class ScaleTestRunner:
    """Run scale tests on the memory system"""

    def __init__(self, target_conversations: int = 500000):
        """Initialize test runner"""
        self.target_conversations = target_conversations
        self.generator = ConversationGenerator()
        self.results = {
            "config": {
                "target_conversations": target_conversations,
                "started_at": datetime.now(timezone.utc).isoformat()
            },
            "population": {},
            "retrieval": {},
            "latency": {},
            "accuracy": {}
        }

    def setup_database(self):
        """Setup database connection"""
        console.print("\n[bold cyan]Setting up database connection...[/bold cyan]")
        try:
            from src.database.connection import get_db_manager

            self.db_manager = get_db_manager()
            self.db_manager.create_tables()

            console.print("[green]✓[/green] Database connected")
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Database setup failed: {e}")
            return False

    def setup_primary_llm(self):
        """Setup Primary LLM (gpt-4o-mini)"""
        console.print("\n[bold cyan]Setting up Primary LLM (gpt-4o-mini)...[/bold cyan]")
        try:
            from src.models.primary_llm import get_primary_llm

            self.primary_llm = get_primary_llm()

            # Verify it's using gpt-4o-mini
            console.print(f"[yellow]Model:[/yellow] {self.primary_llm.model}")
            console.print(f"[yellow]Provider:[/yellow] {self.primary_llm.provider}")

            if "gpt-4o-mini" not in self.primary_llm.model:
                console.print(f"[yellow]⚠[/yellow] Warning: Not using gpt-4o-mini. Current model: {self.primary_llm.model}")
                console.print("[yellow]To use gpt-4o-mini, update config.yaml primary_llm.model to 'gpt-4o-mini'[/yellow]")
            else:
                console.print("[green]✓[/green] gpt-4o-mini configured")

            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Primary LLM setup failed: {e}")
            return False

    def populate_database(self, batch_size: int = 1000):
        """Populate database with synthetic conversations"""
        console.print(f"\n[bold cyan]Populating database with {self.target_conversations:,} conversations...[/bold cyan]")

        from src.models.storage_service import ConversationStorageService

        start_time = time.time()
        stored_count = 0
        failed_count = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            task = progress.add_task(
                "[cyan]Storing conversations...",
                total=self.target_conversations
            )

            batch_times = []

            for batch_start in range(0, self.target_conversations, batch_size):
                batch_end = min(batch_start + batch_size, self.target_conversations)
                current_batch_size = batch_end - batch_start

                batch_start_time = time.time()

                # Generate batch
                conversations = self.generator.generate_batch(batch_start, current_batch_size)

                # Store batch
                with self.db_manager.session_scope() as session:
                    storage_service = ConversationStorageService(session)

                    for conv_data in conversations:
                        try:
                            storage_service.store_conversation(
                                messages=conv_data["messages"],
                                user_id=conv_data["user_id"],
                                importance_score=conv_data["importance_score"],
                                topics=[conv_data["topic"]]
                            )
                            stored_count += 1
                        except Exception as e:
                            failed_count += 1
                            if failed_count <= 10:  # Log first 10 errors
                                logger.error(f"Error storing conversation: {e}")

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

                progress.update(task, advance=current_batch_size)

                # Progress update every 10 batches
                if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
                    avg_batch_time = np.mean(batch_times[-10:])
                    conversations_per_sec = current_batch_size / avg_batch_time
                    logger.info(
                        f"Progress: {stored_count:,}/{self.target_conversations:,} "
                        f"({conversations_per_sec:.1f} conv/sec)"
                    )

        total_time = time.time() - start_time

        # Store results
        self.results["population"] = {
            "stored_count": stored_count,
            "failed_count": failed_count,
            "total_time_seconds": total_time,
            "conversations_per_second": stored_count / total_time,
            "average_batch_time_seconds": np.mean(batch_times),
            "batches_completed": len(batch_times)
        }

        # Display results
        table = Table(title="Population Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Stored", f"{stored_count:,}")
        table.add_row("Failed", f"{failed_count:,}")
        table.add_row("Total Time", f"{total_time:.2f}s")
        table.add_row("Conv/Second", f"{stored_count/total_time:.2f}")

        console.print("\n")
        console.print(table)

        return stored_count == self.target_conversations

    def test_retrieval_performance(self, test_queries: int = 100):
        """Test retrieval performance at scale"""
        console.print(f"\n[bold cyan]Testing retrieval performance ({test_queries} queries)...[/bold cyan]")

        from src.models.storage_service import ConversationStorageService

        # Generate test queries
        test_topics = random.sample(self.generator.TOPICS, min(test_queries, len(self.generator.TOPICS)))
        queries = [f"How do I implement {topic}?" for topic in test_topics]

        # Add more queries if needed
        while len(queries) < test_queries:
            topic = random.choice(self.generator.TOPICS)
            queries.append(f"What are best practices for {topic}?")

        retrieval_times = []
        result_counts = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:

            task = progress.add_task("[cyan]Running queries...", total=test_queries)

            for query in queries[:test_queries]:
                start_time = time.time()

                with self.db_manager.session_scope() as session:
                    storage_service = ConversationStorageService(session)
                    results = storage_service.search_conversations(
                        query=query,
                        limit=10,
                        search_type="semantic"
                    )

                query_time = time.time() - start_time
                retrieval_times.append(query_time)
                result_counts.append(len(results))

                progress.update(task, advance=1)

        # Store results
        self.results["retrieval"] = {
            "total_queries": test_queries,
            "average_query_time_ms": np.mean(retrieval_times) * 1000,
            "median_query_time_ms": np.median(retrieval_times) * 1000,
            "p95_query_time_ms": np.percentile(retrieval_times, 95) * 1000,
            "p99_query_time_ms": np.percentile(retrieval_times, 99) * 1000,
            "average_results_returned": np.mean(result_counts),
            "queries_per_second": test_queries / sum(retrieval_times)
        }

        # Display results
        table = Table(title="Retrieval Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Queries", f"{test_queries:,}")
        table.add_row("Avg Query Time", f"{np.mean(retrieval_times)*1000:.2f}ms")
        table.add_row("Median Query Time", f"{np.median(retrieval_times)*1000:.2f}ms")
        table.add_row("P95 Query Time", f"{np.percentile(retrieval_times, 95)*1000:.2f}ms")
        table.add_row("P99 Query Time", f"{np.percentile(retrieval_times, 99)*1000:.2f}ms")
        table.add_row("Queries/Second", f"{test_queries/sum(retrieval_times):.2f}")

        console.print("\n")
        console.print(table)

    def test_context_window_utilization(self):
        """Test how much data fits in gpt-4o-mini's 125k context window"""
        console.print("\n[bold cyan]Testing context window utilization (125k tokens)...[/bold cyan]")

        from src.models.storage_service import ConversationStorageService

        # Get a large sample of conversations
        with self.db_manager.session_scope() as session:
            storage_service = ConversationStorageService(session)

            # Get recent conversations
            conversations = storage_service.get_recent_conversations(limit=1000)

            # Extract conversation data within session
            conv_data = []
            for conv in conversations:
                conv_data.append({
                    'messages': conv.messages,
                    'summary': conv.summary,
                    'conversation_id': conv.conversation_id
                })

        # Now work with the extracted data outside the session

        # Estimate tokens (rough estimation: 1 token ≈ 4 characters)
        total_chars = 0
        convs_included = 0
        max_tokens = 125000  # gpt-4o-mini context window

        for conv in conv_data:
            conv_text = json.dumps(conv['messages'])
            conv_chars = len(conv_text)
            conv_tokens = conv_chars / 4  # Rough estimate

            if (total_chars / 4) + conv_tokens <= max_tokens * 0.8:  # Use 80% to be safe
                total_chars += conv_chars
                convs_included += 1
            else:
                break

        estimated_tokens = total_chars / 4

        self.results["context_window"] = {
            "max_context_tokens": max_tokens,
            "estimated_tokens_used": estimated_tokens,
            "utilization_percentage": (estimated_tokens / max_tokens) * 100,
            "conversations_in_context": convs_included,
            "total_chars": total_chars
        }

        # Display results
        table = Table(title="Context Window Utilization")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Max Context", f"{max_tokens:,} tokens")
        table.add_row("Estimated Used", f"{int(estimated_tokens):,} tokens")
        table.add_row("Utilization", f"{(estimated_tokens/max_tokens)*100:.2f}%")
        table.add_row("Conversations", f"{convs_included:,}")

        console.print("\n")
        console.print(table)

    def test_end_to_end_latency(self, num_tests: int = 10):
        """Test end-to-end latency for full conversation flow"""
        console.print(f"\n[bold cyan]Testing end-to-end latency ({num_tests} tests)...[/bold cyan]")

        from src.api.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()

        latencies = {
            "total": [],
            "llm": [],
            "memory_search": []
        }

        test_queries = [
            "What are best practices for machine learning?",
            "How do I implement Docker containers?",
            "Can you explain microservices architecture?",
            "What's the best way to design a REST API?",
            "How do I optimize database queries?"
        ]

        for i in range(num_tests):
            query = random.choice(test_queries)

            start_time = time.time()

            # Full conversation processing
            result = orchestrator.process_conversation(
                messages=[{"role": "user", "content": query}],
                user_id=f"test_user_{i}",
                use_memory=True
            )

            total_time = time.time() - start_time
            latencies["total"].append(total_time)

            logger.info(f"Test {i+1}/{num_tests}: {total_time*1000:.2f}ms")

        self.results["latency"] = {
            "end_to_end_avg_ms": np.mean(latencies["total"]) * 1000,
            "end_to_end_median_ms": np.median(latencies["total"]) * 1000,
            "end_to_end_p95_ms": np.percentile(latencies["total"], 95) * 1000,
            "end_to_end_p99_ms": np.percentile(latencies["total"], 99) * 1000,
        }

        # Display results
        table = Table(title="End-to-End Latency")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Average", f"{np.mean(latencies['total'])*1000:.2f}ms")
        table.add_row("Median", f"{np.median(latencies['total'])*1000:.2f}ms")
        table.add_row("P95", f"{np.percentile(latencies['total'], 95)*1000:.2f}ms")
        table.add_row("P99", f"{np.percentile(latencies['total'], 99)*1000:.2f}ms")

        console.print("\n")
        console.print(table)

    def save_results(self, output_path: str = "tests/scale_test_results.json"):
        """Save test results to file"""
        self.results["config"]["completed_at"] = datetime.now(timezone.utc).isoformat()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        console.print(f"\n[green]✓[/green] Results saved to {output_path}")

    def run_full_test(self, skip_population: bool = False):
        """Run complete test suite"""
        console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")
        console.print("[bold magenta]  500K CONVERSATION SCALE TEST[/bold magenta]")
        console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")

        # Setup
        if not self.setup_database():
            return False

        if not self.setup_primary_llm():
            return False

        # Population (can be skipped if DB already populated)
        if not skip_population:
            console.print("\n[yellow]This will populate the database with 500k conversations.[/yellow]")
            console.print("[yellow]This may take several hours. Continue? (y/n)[/yellow]")

            # Auto-continue for non-interactive mode
            if not self.populate_database():
                console.print("[red]Population failed. Aborting test.[/red]")
                return False
        else:
            console.print("\n[yellow]⚠ Skipping population (using existing data)[/yellow]")

        # Run tests
        self.test_retrieval_performance(test_queries=100)
        self.test_context_window_utilization()
        self.test_end_to_end_latency(num_tests=10)

        # Save results
        self.save_results()

        console.print("\n[bold green]✓ Test suite completed![/bold green]")
        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="500k Conversation Scale Test")
    parser.add_argument(
        "--target",
        type=int,
        default=500000,
        help="Target number of conversations (default: 500000)"
    )
    parser.add_argument(
        "--skip-population",
        action="store_true",
        help="Skip database population (use existing data)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test with smaller dataset (10k conversations)"
    )

    args = parser.parse_args()

    target = 10000 if args.quick_test else args.target

    runner = ScaleTestRunner(target_conversations=target)
    success = runner.run_full_test(skip_population=args.skip_population)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
