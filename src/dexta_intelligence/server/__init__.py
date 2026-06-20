"""dexta serve - a local web GUI over the findings store.

A thin, server-rendered FastAPI app: dashboard, wiki, goals, chat, settings.
Stack is deliberately tiny - fastapi + uvicorn + jinja2 + vendored HTMX, no
build step. Everything is a projection of the same SQLite store the CLI reads.
"""

from __future__ import annotations

from dexta_intelligence.server.app import create_app

__all__ = ["create_app"]
