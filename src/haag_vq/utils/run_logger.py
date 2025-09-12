import subprocess
import sqlite3
from datetime import datetime
import os
import sys
import shlex
import json

def log_run(method, dataset, metrics: dict):
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
    metrics_json = json.dumps(metrics)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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
    cursor.execute("""
        INSERT INTO runs (timestamp, git_branch, git_commit, package_version, method, dataset, cli_command, metrics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        git_branch,
        git_commit,
        pkg_version,
        method,
        dataset,
        cli_command,
        metrics_json
    ))

    conn.commit()
    conn.close()