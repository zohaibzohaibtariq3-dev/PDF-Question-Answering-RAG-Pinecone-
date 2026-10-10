"""Application logging + JSONL query log (one line per user question)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_configured = False


def setup_logging(log_dir: Path) -> None:
    global _configured
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_dir / "app.log", encoding="utf-8"), logging.StreamHandler()],
    )
    _configured = True


def log_query(log_dir: Path, record: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with (log_dir / "queries.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
