"""
Celery Application for Background Tasks
Handles periodic training, compression, and maintenance
"""

import logging
from datetime import datetime, timedelta

from celery import Celery
from celery.schedules import crontab

from src.utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize Celery app
app = Celery(
    "memory_system",
    broker=f"redis://{config.database.host if hasattr(config.database, 'host') else 'localhost'}:6379/0",
    backend=f"redis://{config.database.host if hasattr(config.database, 'host') else 'localhost'}:6379/0",
)

# Celery configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    worker_prefetch_multiplier=1,
)


@app.task(name="tasks.train_slm")
def train_slm_task(force: bool = False):
    """
    Task to train/fine-tune the SLM

    Args:
        force: Force training even if threshold not met
    """
    try:
        logger.info("Starting SLM training task...")

        from src.database.connection import get_db_manager
        from src.database.models import TrainingHistory
        from src.database.repository import ConversationRepository, TrainingHistoryRepository
        from src.models.slm_model import get_slm

        db_manager = get_db_manager()

        with db_manager.session_scope() as session:
            conv_repo = ConversationRepository(session)
            train_repo = TrainingHistoryRepository(session)

            # Check if we should train
            threshold = config.slm.retraining.get("conversation_threshold", 25)

            # Count recent important conversations
            important_convs = conv_repo.get_important(
                threshold=config.slm.retraining.get("importance_threshold", 0.7),
                limit=1000,
            )

            # Check last training time
            last_training = train_repo.get_latest_successful()
            time_since_training = None

            if last_training:
                time_since_training = datetime.utcnow() - last_training.completed_at
                hours_since = time_since_training.total_seconds() / 3600
            else:
                hours_since = float("inf")

            time_threshold = config.slm.retraining.get("time_threshold_hours", 24)

            # Decide whether to train
            should_train = (
                force
                or len(important_convs) >= threshold
                or hours_since >= time_threshold
            )

            if not should_train:
                logger.info(
                    f"Training not needed. {len(important_convs)}/{threshold} conversations, "
                    f"{hours_since:.1f}/{time_threshold} hours"
                )
                return {"status": "skipped", "reason": "threshold not met"}

            # Create training record
            training_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            training_record = TrainingHistory(
                training_id=training_id,
                started_at=datetime.utcnow(),
                num_conversations=len(important_convs),
                status="running",
            )
            train_repo.create(training_record)

        # Train the model (outside session scope to avoid timeout)
        try:
            slm = get_slm()
            adapter_path = slm.fine_tune(important_convs)

            # Update training record
            with db_manager.session_scope() as session:
                train_repo = TrainingHistoryRepository(session)
                train_repo.update_status(
                    training_id, "completed", error_message=None
                )

                # Update adapter path
                training = train_repo.get_by_id(training_id)
                if training:
                    training.adapter_path = adapter_path
                    training.completed_at = datetime.utcnow()
                    session.commit()

            logger.info(f"Training completed successfully: {adapter_path}")

            return {
                "status": "success",
                "training_id": training_id,
                "adapter_path": adapter_path,
                "num_conversations": len(important_convs),
            }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)

            # Update training record with error
            with db_manager.session_scope() as session:
                train_repo = TrainingHistoryRepository(session)
                train_repo.update_status(training_id, "failed", error_message=str(e))

            return {"status": "failed", "error": str(e)}

    except Exception as e:
        logger.error(f"Error in training task: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


@app.task(name="tasks.compress_old_conversations")
def compress_old_conversations_task():
    """Task to compress old conversations to save space"""
    try:
        logger.info("Starting conversation compression task...")

        from src.database.connection import get_db_manager
        from src.database.repository import ConversationRepository

        db_manager = get_db_manager()
        compress_after_days = config.memory.retention.get("compress_after_days", 30)

        with db_manager.session_scope() as session:
            conv_repo = ConversationRepository(session)

            # Find old uncompressed conversations
            cutoff_date = datetime.utcnow() - timedelta(days=compress_after_days)
            conversations = conv_repo.repo.search_by_date_range(
                datetime(2000, 1, 1), cutoff_date
            )

            compressed_count = 0
            for conv in conversations:
                if not conv.is_compressed:
                    # Keep only summary, discard full messages
                    conv.messages = []
                    conv.is_compressed = True
                    compressed_count += 1

            session.commit()

        logger.info(f"Compressed {compressed_count} conversations")

        return {"status": "success", "compressed_count": compressed_count}

    except Exception as e:
        logger.error(f"Error in compression task: {e}")
        return {"status": "error", "error": str(e)}


@app.task(name="tasks.backup_database")
def backup_database_task():
    """Task to backup the database"""
    try:
        logger.info("Starting database backup task...")

        import subprocess
        from pathlib import Path

        backup_path = Path(config.memory.storage.get("backup_path", "./data/backups"))
        backup_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"backup_{timestamp}.sql"

        # PostgreSQL backup command
        cmd = [
            "pg_dump",
            "-h", config.database.host,
            "-U", config.database.user,
            "-d", config.database.name,
            "-f", str(backup_file),
        ]

        subprocess.run(cmd, check=True)

        logger.info(f"Database backup created: {backup_file}")

        return {"status": "success", "backup_file": str(backup_file)}

    except Exception as e:
        logger.error(f"Error in backup task: {e}")
        return {"status": "error", "error": str(e)}


# Periodic task schedule
app.conf.beat_schedule = {
    "train-slm-daily": {
        "task": "tasks.train_slm",
        "schedule": crontab(hour=2, minute=0),  # 2 AM daily
        "args": (False,),  # force=False
    },
    "compress-old-conversations-weekly": {
        "task": "tasks.compress_old_conversations",
        "schedule": crontab(hour=3, minute=0, day_of_week=0),  # Sunday 3 AM
    },
    "backup-database-daily": {
        "task": "tasks.backup_database",
        "schedule": crontab(hour=1, minute=0),  # 1 AM daily
    },
}


if __name__ == "__main__":
    # Run worker
    app.worker_main()
