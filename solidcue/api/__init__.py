"""HTTP delivery layer for solidcue.

This package exposes the existing ``solidcue.services`` functions over HTTP via
FastAPI. It is a thin transport wrapper — all business logic lives in
``solidcue.services`` and is shared with the CLI (``solidcue.app``). Nothing in
this package should contain domain or orchestration logic.
"""

from solidcue.api.main import create_app

__all__ = ["create_app"]
