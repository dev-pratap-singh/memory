"""
Train SLM on database conversations
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from src.database.connection import get_db_manager
from src.models.slm_model import get_slm
from src.models.storage_service import ConversationStorageService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


def train_slm_on_conversations(max_conversations: int = 5000):
    """Train SLM on database conversations"""

    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  SLM TRAINING[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")

    # Get database manager
    console.print("\n[bold]Step 1: Fetching conversations from database...[/bold]")
    db_manager = get_db_manager()

    conversations = []
    with db_manager.session_scope() as session:
        storage_service = ConversationStorageService(session)
        db_conversations = storage_service.get_recent_conversations(limit=max_conversations)

        # Extract conversation data within session to avoid DetachedInstanceError
        conv_list = []
        for conv in db_conversations:
            # Create a dict with all needed data
            conv_data = {
                'conversation_id': conv.conversation_id,
                'user_id': conv.user_id,
                'messages': conv.messages,
                'summary': conv.summary,
                'topics': conv.topics,
                'importance_score': conv.importance_score,
                'created_at': conv.created_at,
                'updated_at': conv.updated_at,
            }
            conv_list.append(conv_data)

        conversations = conv_list

    console.print(f"[green]✓[/green] Fetched {len(conversations)} conversations")

    if len(conversations) == 0:
        console.print("[red]✗[/red] No conversations found. Please populate database first.")
        return None

    # Get SLM
    console.print("\n[bold]Step 2: Initializing SLM...[/bold]")
    slm = get_slm()
    console.print(f"[green]✓[/green] SLM initialized: {slm.model_name}")

    # Show model info
    info = slm.get_model_info()
    console.print(f"  Total parameters: {info['total_params']:,}")
    console.print(f"  Trainable parameters: {info['trainable_params']:,}")

    # Train
    console.print(f"\n[bold]Step 3: Training on {len(conversations)} conversations...[/bold]")
    console.print("[yellow]This may take several minutes...[/yellow]")

    try:
        adapter_path = slm.fine_tune(conversations)

        if adapter_path:
            console.print(f"\n[green]✓[/green] Training completed successfully!")
            console.print(f"[green]✓[/green] Adapter saved to: {adapter_path}")
            return adapter_path
        else:
            console.print("[red]✗[/red] Training failed - no adapter path returned")
            return None

    except Exception as e:
        console.print(f"[red]✗[/red] Training failed: {e}")
        logger.error(f"Training error: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SLM on conversations")
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=5000,
        help="Maximum conversations to train on (default: 5000)"
    )

    args = parser.parse_args()

    adapter_path = train_slm_on_conversations(max_conversations=args.max_conversations)

    if adapter_path:
        console.print("\n[bold green]Training completed successfully![/bold green]")
        console.print(f"[bold]Adapter path: {adapter_path}[/bold]")
        sys.exit(0)
    else:
        console.print("\n[bold red]Training failed![/bold red]")
        sys.exit(1)
