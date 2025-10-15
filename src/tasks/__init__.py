"""Background tasks module"""

from src.tasks.celery_app import app, train_slm_task, compress_old_conversations_task, backup_database_task

__all__ = ["app", "train_slm_task", "compress_old_conversations_task", "backup_database_task"]
