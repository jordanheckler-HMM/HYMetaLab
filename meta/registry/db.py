import json
import os
import sqlite3
import subprocess
import time

DB_PATH = os.environ.get("METOLAB_DB", "metolab.sqlite")
SCHEMA = """CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, study TEXT, params TEXT, git TEXT, status TEXT, export_dir TEXT
);"""


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def connect():
    con = sqlite3.connect(DB_PATH)
    con.execute(SCHEMA)
    return con


def log_run(study: str, params: dict, status: str, export_dir: str = ""):
    con = connect()
    con.execute(
        "INSERT INTO runs(ts,study,params,git,status,export_dir) VALUES(?,?,?,?,?,?)",
        (
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            study,
            json.dumps(params),
            _git_hash(),
            status,
            export_dir,
        ),
    )
    con.commit()
    con.close()
