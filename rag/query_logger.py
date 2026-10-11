
"""Application logging + JSONL query log (one line per user question)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

_configured = False


def setup_logging(log_dir: Path) -> None:
    global _configured
    if _configured:
        return

    is_vercel = os.environ.get("VERCEL") == "1"

    handlers = [logging.StreamHandler()]

    if not is_vercel:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.insert(
            0,
            logging.FileHandler(log_dir / "app.log", encoding="utf-8"),
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    _configured = True


def log_query(log_dir: Path, record: dict) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}

    if os.environ.get("VERCEL") == "1":
        logging.getLogger(__name__).info(
            "Query log: %s",
            json.dumps(record, ensure_ascii=False),
        )
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "queries.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

