import subprocess
import sqlite3
from datetime import datetime
import os
import sys
import shlex
import json
import numpy as np

def log_run(method, dataset, metrics: dict, config: dict = None, sweep_id: str = None):
    """
    Log a benchmark run to the SQLite database.

    Args:
        method: Quantization method name (e.g., "pq", "sq")
        dataset: Dataset name (e.g., "dummy", "huggingface")
        metrics: Dictionary of metric values
        config: Optional configuration dictionary (e.g., num_chunks, num_clusters)
        sweep_id: Optional sweep identifier to group related runs together
    """
    os.makedirs("logs", exist_ok=True)
    db_path = "logs/benchmark_runs.db"

    try:
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_branch = "unknown"
        git_commit = "unknown"

    try:
        from importlib.metadata import version
        pkg_version = version("haag-vq")
    except Exception:
        pkg_version = "dev"

    cli_command = " ".join(shlex.quote(arg) for arg in sys.argv)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_json = json.dumps(convert_to_native(metrics))
    config_json = json.dumps(convert_to_native(config)) if config else "{}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with initial schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            git_branch TEXT,
            git_commit TEXT,
            package_version TEXT,
            method TEXT,
            dataset TEXT,
            cli_command TEXT,
            metrics_json TEXT
        )
    """)

    # Add config_json column if it doesn't exist (for backwards compatibility)
    try:
        cursor.execute("ALTER TABLE runs ADD COLUMN config_json TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Add sweep_id column if it doesn't exist (for backwards compatibility)
    try:
        cursor.execute("ALTER TABLE runs ADD COLUMN sweep_id TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists
        pass

    cursor.execute("""
        INSERT INTO runs (timestamp, git_branch, git_commit, package_version, method, dataset, cli_command, metrics_json, config_json, sweep_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        git_branch,
        git_commit,
        pkg_version,
        method,
        dataset,
        cli_command,
        metrics_json,
        config_json,
        sweep_id
    ))

    conn.commit()
    conn.close()