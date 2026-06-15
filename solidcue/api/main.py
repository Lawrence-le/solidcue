"""FastAPI application factory and entrypoint.

The app is a thin HTTP surface over ``solidcue.services``. Run it with::

    uvicorn solidcue.api.main:app --reload

or via the ``api`` console script (``api``), which calls :func:`run`.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from solidcue.api.routes import agents, mcp, profile, state, tools

_DEFAULT_ORIGINS = [
    "http://localhost:5173",  # Vite (studio) dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]


def _allowed_origins() -> list[str]:
    configured = os.getenv("SOLIDCUE_API_CORS_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return _DEFAULT_ORIGINS


def create_app() -> FastAPI:
    app = FastAPI(
        title="solidcue API",
        description="HTTP surface over solidcue agent services.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "name": "solidcue API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(agents.router)
    app.include_router(tools.router)
    app.include_router(mcp.router)
    app.include_router(state.router)
    app.include_router(profile.router)

    return app


app = create_app()


def run() -> None:
    """Console-script entrypoint: serve the API with uvicorn."""
    import uvicorn

    host = os.getenv("SOLIDCUE_API_HOST", "127.0.0.1")
    port = int(os.getenv("SOLIDCUE_API_PORT", "8000"))
    uvicorn.run("solidcue.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
