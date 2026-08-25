#!/usr/bin/env python3
"""Start EmAtSy: camera workers + web UI.  Usage: python scripts/run.py"""
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import uvicorn
from app.config import settings

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s", datefmt="%H:%M:%S")

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host=settings.host, port=settings.port,
                log_level="warning", access_log=False)
