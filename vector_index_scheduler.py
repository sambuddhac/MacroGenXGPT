#!/usr/bin/env python3
"""
Vector Index Auto-Update Scheduler

This module provides automatic weekly updates for the vector index.
It uses APScheduler to run periodic updates based on configuration.
"""

import os
import sys
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Import the RAG system
from rag_system_ollama import DocumentationRAG


class VectorIndexScheduler:
    """Manages automatic updates of the vector index on a weekly schedule."""

    def __init__(self, config_path: str = 'scheduler_config.yaml'):
        """
        Initialize the scheduler with configuration.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.scheduler = BlockingScheduler()
        self.rag_system: Optional[DocumentationRAG] = None

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        log_config = self.config.get('logging', {})
        log_file = log_config.get('log_file', 'scheduler.log')
        log_level = log_config.get('level', 'INFO')
        console_output = log_config.get('console_output', True)

        # Create logger
        logger = logging.getLogger('VectorIndexScheduler')
        logger.setLevel(getattr(logging, log_level))

        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def _create_backup(self, output_path: str) -> bool:
        """
        Create a backup of existing vector index files.

        Args:
            output_path: Path to the vector index files

        Returns:
            True if backup was created successfully, False otherwise
        """
        backup_config = self.config.get('backup', {})
        if not backup_config.get('enabled', True):
            self.logger.info("Backup is disabled in configuration")
            return True

        backup_dir = Path(backup_config.get('backup_dir', 'backups'))
        backup_dir.mkdir(exist_ok=True)

        # Create timestamp-based backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        pkl_file = Path(output_path)
        faiss_file = Path(f"{output_path}.faiss")

        if not pkl_file.exists():
            self.logger.info(f"No existing index to backup at {output_path}")
            return True

        try:
            # Backup pickle file
            backup_pkl = backup_dir / f"{pkl_file.stem}_{timestamp}{pkl_file.suffix}"
            shutil.copy2(pkl_file, backup_pkl)
            self.logger.info(f"Backed up {pkl_file} to {backup_pkl}")

            # Backup FAISS file if it exists
            if faiss_file.exists():
                backup_faiss = backup_dir / f"{pkl_file.stem}_{timestamp}{pkl_file.suffix}.faiss"
                shutil.copy2(faiss_file, backup_faiss)
                self.logger.info(f"Backed up {faiss_file} to {backup_faiss}")

            # Clean old backups
            self._clean_old_backups(backup_dir, backup_config.get('retention_days', 30))

            return True

        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False

    def _clean_old_backups(self, backup_dir: Path, retention_days: int):
        """
        Remove backups older than retention period.

        Args:
            backup_dir: Directory containing backups
            retention_days: Number of days to keep backups
        """
        try:
            current_time = time.time()
            retention_seconds = retention_days * 86400

            for backup_file in backup_dir.iterdir():
                if backup_file.is_file():
                    file_age = current_time - backup_file.stat().st_mtime
                    if file_age > retention_seconds:
                        backup_file.unlink()
                        self.logger.info(f"Removed old backup: {backup_file}")

        except Exception as e:
            self.logger.error(f"Failed to clean old backups: {e}")

    def _initialize_rag_system(self) -> DocumentationRAG:
        """
        Initialize the RAG system based on configuration.

        Returns:
            Initialized DocumentationRAG instance
        """
        vector_config = self.config.get('vector_index', {})
        embedding_model = vector_config.get('embedding_model', 'all-MiniLM-L6-v2')
        llm_backend = vector_config.get('llm_backend', 'ollama')

        if llm_backend == 'ollama':
            ollama_config = vector_config.get('ollama', {})
            ollama_model = ollama_config.get('model', 'llama3.2')

            self.logger.info(f"Initializing RAG with Ollama model: {ollama_model}")
            rag = DocumentationRAG(
                embedding_model=embedding_model,
                ollama_model=ollama_model
            )
        else:
            # Use OpenAI/Portkey backend
            from rag_system import DocumentationRAG as OpenAIRAG
            self.logger.info("Initializing RAG with OpenAI/Portkey")
            rag = OpenAIRAG(embedding_model=embedding_model)

        return rag

    def update_vector_index(self):
        """
        Main function to update the vector index.
        This is the job that gets scheduled to run weekly.
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting vector index update")
        self.logger.info("=" * 80)

        error_config = self.config.get('error_handling', {})
        max_retries = error_config.get('max_retries', 3)
        retry_delay = error_config.get('retry_delay', 300)

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)

                # Get configuration
                vector_config = self.config.get('vector_index', {})
                output_path = vector_config.get('output_path', 'energy_packages_rag.pkl')
                docs_urls = vector_config.get('documentation_urls', [])
                max_pages = vector_config.get('max_pages', 50)

                if not docs_urls:
                    raise ValueError("No documentation URLs configured")

                # Create backup
                self.logger.info("Creating backup of existing index...")
                self._create_backup(output_path)

                # Initialize RAG system
                self.logger.info("Initializing RAG system...")
                self.rag_system = self._initialize_rag_system()

                # Build the index
                self.logger.info(f"Building index from {len(docs_urls)} documentation sources")
                self.logger.info(f"Documentation URLs: {docs_urls}")

                start_time = time.time()
                self.rag_system.build_index(docs_urls, max_pages=max_pages)
                elapsed_time = time.time() - start_time

                # Save the index
                self.logger.info(f"Saving index to {output_path}...")
                self.rag_system.save(output_path)

                # Log statistics
                num_docs = len(self.rag_system.documents) if self.rag_system.documents else 0
                self.logger.info(f"Index update completed successfully!")
                self.logger.info(f"Total documents: {num_docs}")
                self.logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
                self.logger.info("=" * 80)

                return

            except Exception as e:
                self.logger.error(f"Error updating vector index (attempt {attempt + 1}/{max_retries}): {e}")
                self.logger.exception("Full traceback:")

                if attempt == max_retries - 1:
                    self.logger.error("Max retries reached. Update failed.")
                    if error_config.get('notify_on_failure', False):
                        self._send_failure_notification(str(e))

    def _send_failure_notification(self, error_message: str):
        """
        Send notification on failure (placeholder for future implementation).

        Args:
            error_message: The error message to include in notification
        """
        self.logger.warning("Failure notification not implemented yet")
        # TODO: Implement email, Slack, or other notification service

    def start(self):
        """Start the scheduler and run in blocking mode."""
        schedule_config = self.config.get('schedule', {})

        day_of_week = schedule_config.get('day_of_week', 0)  # 0 = Monday
        hour = schedule_config.get('hour', 2)
        minute = schedule_config.get('minute', 0)
        timezone = schedule_config.get('timezone', 'UTC')

        # Map numeric day to cron format
        day_names = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        day_name = day_names[day_of_week] if 0 <= day_of_week < 7 else 'mon'

        self.logger.info("Vector Index Scheduler Started")
        self.logger.info(f"Schedule: Every {day_name.capitalize()} at {hour:02d}:{minute:02d} {timezone}")
        self.logger.info(f"Configuration file: {self.config_path}")
        self.logger.info("-" * 80)

        # Add the job to the scheduler
        trigger = CronTrigger(
            day_of_week=day_name,
            hour=hour,
            minute=minute,
            timezone=timezone
        )

        self.scheduler.add_job(
            self.update_vector_index,
            trigger=trigger,
            id='vector_index_update',
            name='Vector Index Weekly Update',
            replace_existing=True
        )

        # Print next run time
        next_run = self.scheduler.get_jobs()[0].next_run_time
        self.logger.info(f"Next scheduled update: {next_run}")
        self.logger.info("-" * 80)

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("Scheduler stopped by user")
            self.scheduler.shutdown()

    def run_now(self):
        """Run an immediate update without starting the scheduler."""
        self.logger.info("Running immediate vector index update...")
        self.update_vector_index()


def main():
    """Main entry point for the scheduler."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Vector Index Auto-Update Scheduler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the scheduler with default config
  python vector_index_scheduler.py

  # Start with custom config file
  python vector_index_scheduler.py --config my_config.yaml

  # Run an immediate update without scheduling
  python vector_index_scheduler.py --now

  # Show next scheduled run time
  python vector_index_scheduler.py --show-next
        """
    )

    parser.add_argument(
        '--config',
        default='scheduler_config.yaml',
        help='Path to configuration file (default: scheduler_config.yaml)'
    )

    parser.add_argument(
        '--now',
        action='store_true',
        help='Run an immediate update without starting the scheduler'
    )

    parser.add_argument(
        '--show-next',
        action='store_true',
        help='Show the next scheduled run time and exit'
    )

    args = parser.parse_args()

    try:
        scheduler = VectorIndexScheduler(config_path=args.config)

        if args.now:
            scheduler.run_now()
        elif args.show_next:
            schedule_config = scheduler.config.get('schedule', {})
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week = schedule_config.get('day_of_week', 0)
            hour = schedule_config.get('hour', 2)
            minute = schedule_config.get('minute', 0)
            timezone = schedule_config.get('timezone', 'UTC')

            if not isinstance(day_of_week, int) or not (0 <= day_of_week < len(day_names)):
                print(f"Error: day_of_week value '{day_of_week}' is out of range (0-6).", file=sys.stderr)
                sys.exit(1)
            print(f"Next update: Every {day_names[day_of_week]} at {hour:02d}:{minute:02d} {timezone}")
        else:
            scheduler.start()

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
