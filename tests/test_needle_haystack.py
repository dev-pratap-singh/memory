"""
Needle in a Haystack Test for Dual-Model Memory System
Adapted from chroma-core/context-rot methodology

Tests if the dual-model system can retrieve specific facts (needles)
from a large database of conversations (haystack) using semantic search.

This addresses the "context rot" problem identified by Chroma Research:
https://research.trychroma.com/context-rot
"""

import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


class NeedleInHaystackTest:
    """
    Test retrieval accuracy of the dual-model system with needle-in-haystack methodology
    """

    # Define 20 needle facts with questions and expected answers
    NEEDLES = [
        {
            "fact": "The user's favorite programming language is Rust because it combines memory safety with high performance.",
            "query": "What is my favorite programming language and why?",
            "expected_keywords": ["rust", "memory safety", "performance"],
            "category": "preferences"
        },
        {
            "fact": "The project deployment is scheduled for March 15th, 2025 at 3:00 PM EST.",
            "query": "When is the project deployment scheduled?",
            "expected_keywords": ["march 15", "2025", "3:00 pm", "est"],
            "category": "dates"
        },
        {
            "fact": "The database password was last updated on January 3rd, 2025 by the security team.",
            "query": "When was the database password last changed?",
            "expected_keywords": ["january 3", "2025", "security team"],
            "category": "security"
        },
        {
            "fact": "The recommended API rate limit is 1000 requests per minute for premium tier users.",
            "query": "What is the API rate limit for premium users?",
            "expected_keywords": ["1000", "requests", "minute", "premium"],
            "category": "technical"
        },
        {
            "fact": "The company's annual revenue target for Q2 2025 is $2.5 million with a 15% growth rate.",
            "query": "What is our Q2 2025 revenue target?",
            "expected_keywords": ["2.5 million", "q2", "2025", "15%"],
            "category": "business"
        },
        {
            "fact": "The recommended Docker base image for production is alpine:3.19 for minimal size.",
            "query": "Which Docker base image should we use in production?",
            "expected_keywords": ["alpine", "3.19", "production"],
            "category": "devops"
        },
        {
            "fact": "The user's preferred code editor is Neovim with the LazyVim configuration.",
            "query": "What code editor do I prefer?",
            "expected_keywords": ["neovim", "lazyvim"],
            "category": "preferences"
        },
        {
            "fact": "The backup system runs every night at 2:30 AM and retains snapshots for 30 days.",
            "query": "When does the backup system run and how long are backups kept?",
            "expected_keywords": ["2:30 am", "night", "30 days"],
            "category": "operations"
        },
        {
            "fact": "The optimal batch size for the machine learning model training is 32 with a learning rate of 0.001.",
            "query": "What are the optimal training parameters for our ML model?",
            "expected_keywords": ["batch size", "32", "learning rate", "0.001"],
            "category": "ml"
        },
        {
            "fact": "The user's birthday is June 18th, 1990 and their favorite cake flavor is chocolate raspberry.",
            "query": "When is my birthday and what's my favorite cake?",
            "expected_keywords": ["june 18", "1990", "chocolate raspberry"],
            "category": "personal"
        },
        {
            "fact": "The database connection pool size is set to 20 connections with a 30-second timeout.",
            "query": "What are our database connection pool settings?",
            "expected_keywords": ["20", "connections", "30", "timeout"],
            "category": "database"
        },
        {
            "fact": "The team's daily standup is at 9:30 AM Pacific Time on Zoom with meeting ID 123-456-789.",
            "query": "When is our daily standup and where?",
            "expected_keywords": ["9:30 am", "pacific", "zoom", "123-456-789"],
            "category": "meetings"
        },
        {
            "fact": "The production SSL certificate expires on December 31st, 2025 and needs renewal 30 days before.",
            "query": "When does our SSL certificate expire?",
            "expected_keywords": ["december 31", "2025", "30 days"],
            "category": "security"
        },
        {
            "fact": "The recommended PostgreSQL version for production is 16.2 with the pgvector extension version 0.6.0.",
            "query": "Which PostgreSQL version should we use?",
            "expected_keywords": ["16.2", "pgvector", "0.6.0"],
            "category": "database"
        },
        {
            "fact": "The user's home office is located in Austin, Texas with a dedicated fiber connection at 1 Gbps.",
            "query": "Where is my home office and what's the internet speed?",
            "expected_keywords": ["austin", "texas", "1 gbps", "fiber"],
            "category": "personal"
        },
        {
            "fact": "The CI/CD pipeline typically completes in 8 minutes with 4 parallel test runners.",
            "query": "How long does our CI/CD pipeline take?",
            "expected_keywords": ["8 minutes", "4", "parallel", "test"],
            "category": "devops"
        },
        {
            "fact": "The customer support SLA is 4 hours for critical issues and 24 hours for standard requests.",
            "query": "What is our support SLA for critical issues?",
            "expected_keywords": ["4 hours", "critical", "24 hours"],
            "category": "support"
        },
        {
            "fact": "The user's preferred vacation destination is Kyoto, Japan during the cherry blossom season in April.",
            "query": "Where do I like to vacation and when?",
            "expected_keywords": ["kyoto", "japan", "cherry blossom", "april"],
            "category": "personal"
        },
        {
            "fact": "The Kubernetes cluster runs on 5 nodes with 32 GB RAM each and uses the containerd runtime.",
            "query": "What are our Kubernetes cluster specifications?",
            "expected_keywords": ["5 nodes", "32 gb", "containerd"],
            "category": "infrastructure"
        },
        {
            "fact": "The monitoring system sends alerts to Slack channel #alerts and PagerDuty for P0 incidents.",
            "query": "Where do monitoring alerts go?",
            "expected_keywords": ["slack", "#alerts", "pagerduty", "p0"],
            "category": "operations"
        }
    ]

    def __init__(self):
        """Initialize needle-in-haystack test"""
        self.results = {
            "config": {
                "test_type": "needle_in_haystack",
                "num_needles": len(self.NEEDLES),
                "started_at": datetime.now(timezone.utc).isoformat()
            },
            "accuracy": {},
            "needle_results": []
        }
        self.needle_conversation_ids = []

    def setup_database(self):
        """Setup database connection"""
        console.print("\n[bold cyan]Setting up database connection...[/bold cyan]")
        try:
            from src.database.connection import get_db_manager

            self.db_manager = get_db_manager()
            console.print("[green]✓[/green] Database connected")
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Database setup failed: {e}")
            return False

    def setup_primary_llm(self):
        """Setup Primary LLM"""
        console.print("\n[bold cyan]Setting up Primary LLM...[/bold cyan]")
        try:
            from src.models.primary_llm import get_primary_llm

            self.primary_llm = get_primary_llm()
            console.print(f"[yellow]Model:[/yellow] {self.primary_llm.model}")
            console.print("[green]✓[/green] Primary LLM configured")
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Primary LLM setup failed: {e}")
            return False

    def insert_needles(self):
        """Insert needle conversations into the database"""
        console.print(f"\n[bold cyan]Inserting {len(self.NEEDLES)} needle conversations...[/bold cyan]")

        from src.models.storage_service import ConversationStorageService

        inserted_count = 0
        failed_count = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task("[cyan]Inserting needles...", total=len(self.NEEDLES))

            for needle in self.NEEDLES:
                try:
                    # Create a conversation with the needle fact
                    messages = [
                        {
                            "role": "user",
                            "content": f"I want to tell you something important: {needle['fact']}"
                        },
                        {
                            "role": "assistant",
                            "content": f"Thank you for sharing that information. I've recorded that {needle['fact']} I'll remember this for future reference."
                        }
                    ]

                    with self.db_manager.session_scope() as session:
                        storage_service = ConversationStorageService(session)

                        # Store with high importance and category tag
                        conv_id = storage_service.store_conversation(
                            messages=messages,
                            user_id="needle_test_user",
                            importance_score=1.0,  # Maximum importance
                            topics=[needle['category'], "needle_fact"]
                        )

                        self.needle_conversation_ids.append({
                            "conversation_id": conv_id,
                            "needle": needle
                        })

                        inserted_count += 1
                        logger.info(f"Inserted needle {inserted_count}: {needle['category']}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to insert needle: {e}")

                progress.update(task, advance=1)

        console.print(f"\n[green]✓[/green] Inserted {inserted_count} needles")
        if failed_count > 0:
            console.print(f"[yellow]⚠[/yellow] Failed to insert {failed_count} needles")

        return inserted_count

    def test_needle_retrieval(self):
        """Test if Primary LLM can retrieve needle facts via semantic search"""
        console.print(f"\n[bold cyan]Testing needle retrieval ({len(self.NEEDLES)} queries)...[/bold cyan]")

        from src.api.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()

        results = []
        retrieval_times = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task("[cyan]Testing retrieval...", total=len(self.NEEDLES))

            for i, needle in enumerate(self.NEEDLES):
                start_time = time.time()

                try:
                    # Query with Primary LLM using memory
                    result = orchestrator.process_conversation(
                        messages=[{"role": "user", "content": needle["query"]}],
                        user_id="needle_test_user",
                        use_memory=True
                    )

                    retrieval_time = time.time() - start_time
                    retrieval_times.append(retrieval_time)

                    # Extract response text from result dict
                    response = result.get("response", "")
                    if not isinstance(response, str):
                        response = str(response)

                    # Check if response contains expected keywords
                    response_lower = response.lower()
                    keywords_found = []
                    keywords_missing = []

                    for keyword in needle["expected_keywords"]:
                        if keyword.lower() in response_lower:
                            keywords_found.append(keyword)
                        else:
                            keywords_missing.append(keyword)

                    # Calculate accuracy for this needle
                    keyword_accuracy = len(keywords_found) / len(needle["expected_keywords"])
                    is_correct = keyword_accuracy >= 0.5  # At least 50% of keywords must be present

                    result = {
                        "needle_id": i,
                        "category": needle["category"],
                        "query": needle["query"],
                        "expected_keywords": needle["expected_keywords"],
                        "keywords_found": keywords_found,
                        "keywords_missing": keywords_missing,
                        "keyword_accuracy": keyword_accuracy,
                        "is_correct": is_correct,
                        "retrieval_time_ms": retrieval_time * 1000,
                        "response_preview": response[:200] + "..." if len(response) > 200 else response
                    }

                    results.append(result)

                    logger.info(
                        f"Needle {i+1}: {needle['category']} - "
                        f"{'✓' if is_correct else '✗'} "
                        f"({len(keywords_found)}/{len(needle['expected_keywords'])} keywords)"
                    )

                except Exception as e:
                    logger.error(f"Error testing needle {i}: {e}")
                    results.append({
                        "needle_id": i,
                        "category": needle["category"],
                        "query": needle["query"],
                        "error": str(e),
                        "is_correct": False
                    })

                progress.update(task, advance=1)

        # Calculate overall accuracy
        correct_count = sum(1 for r in results if r.get("is_correct", False))
        total_count = len(results)
        accuracy_rate = (correct_count / total_count) * 100 if total_count > 0 else 0

        # Calculate average keyword accuracy
        avg_keyword_accuracy = sum(r.get("keyword_accuracy", 0) for r in results) / total_count if total_count > 0 else 0

        self.results["accuracy"] = {
            "total_needles": total_count,
            "needles_found": correct_count,
            "needles_missed": total_count - correct_count,
            "accuracy_percentage": accuracy_rate,
            "average_keyword_accuracy": avg_keyword_accuracy * 100,
            "average_retrieval_time_ms": sum(retrieval_times) / len(retrieval_times) * 1000 if retrieval_times else 0
        }

        self.results["needle_results"] = results

        # Display results
        table = Table(title="Needle in a Haystack - Accuracy Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Needles", str(total_count))
        table.add_row("Needles Found", f"{correct_count} ({accuracy_rate:.1f}%)")
        table.add_row("Needles Missed", str(total_count - correct_count))
        table.add_row("Avg Keyword Match", f"{avg_keyword_accuracy*100:.1f}%")
        table.add_row("Avg Retrieval Time", f"{sum(retrieval_times)/len(retrieval_times)*1000:.2f}ms")

        console.print("\n")
        console.print(table)

        # Display per-category accuracy
        category_stats = {}
        for result in results:
            cat = result.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0}
            category_stats[cat]["total"] += 1
            if result.get("is_correct", False):
                category_stats[cat]["correct"] += 1

        category_table = Table(title="Accuracy by Category")
        category_table.add_column("Category", style="cyan")
        category_table.add_column("Accuracy", style="green")

        for cat, stats in sorted(category_stats.items()):
            accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            category_table.add_row(cat, f"{stats['correct']}/{stats['total']} ({accuracy:.0f}%)")

        console.print("\n")
        console.print(category_table)

        return accuracy_rate

    def save_results(self, output_path: str = "tests/needle_haystack_results.json"):
        """Save test results to file"""
        self.results["config"]["completed_at"] = datetime.now(timezone.utc).isoformat()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        console.print(f"\n[green]✓[/green] Results saved to {output_path}")

    def run_full_test(self):
        """Run complete needle-in-haystack test"""
        console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")
        console.print("[bold magenta]  NEEDLE IN A HAYSTACK TEST[/bold magenta]")
        console.print("[bold magenta]  Dual-Model Memory System[/bold magenta]")
        console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")

        # Setup
        if not self.setup_database():
            return False

        if not self.setup_primary_llm():
            return False

        # Insert needles
        inserted = self.insert_needles()
        if inserted == 0:
            console.print("[red]✗[/red] Failed to insert needles. Aborting test.")
            return False

        # Wait a moment for embeddings to be generated
        console.print("\n[yellow]Waiting 2 seconds for embeddings...[/yellow]")
        time.sleep(2)

        # Test retrieval
        accuracy = self.test_needle_retrieval()

        # Save results
        self.save_results()

        # Final grade
        console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
        if accuracy >= 90:
            console.print("[bold green]GRADE: A+ (Excellent)[/bold green]")
        elif accuracy >= 80:
            console.print("[bold green]GRADE: A (Very Good)[/bold green]")
        elif accuracy >= 70:
            console.print("[bold yellow]GRADE: B (Good)[/bold yellow]")
        elif accuracy >= 60:
            console.print("[bold yellow]GRADE: C (Fair)[/bold yellow]")
        else:
            console.print("[bold red]GRADE: D (Needs Improvement)[/bold red]")

        console.print(f"[bold]Accuracy: {accuracy:.1f}%[/bold]")
        console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")

        console.print("\n[bold green]✓ Needle in a Haystack test completed![/bold green]")
        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Needle in a Haystack Test")
    parser.add_argument(
        "--output",
        type=str,
        default="tests/needle_haystack_results.json",
        help="Output path for results JSON"
    )

    args = parser.parse_args()

    runner = NeedleInHaystackTest()
    success = runner.run_full_test()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
