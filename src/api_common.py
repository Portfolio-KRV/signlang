"""Shared FastAPI plumbing: app factory, rate limiter, error handlers.

Every project's `api/app.py` was carrying ~100 lines of identical boilerplate
(CORS, slowapi setup, six exception handlers). This module collapses that
into a `create_app(...)` factory plus a `register_error_handlers(...)`
helper so each project's `app.py` becomes about routes, not glue.

Usage::

    from src.api_common import create_app, register_error_handlers, limiter
    from src.exceptions import ModelNotLoadedError, DataValidationError

    app = create_app(
        title="My API",
        description="...",
        version=__version__,
    )
    register_error_handlers(app, {
        ModelNotLoadedError: (503, "model_not_loaded"),
        DataValidationError: (422, "validation_error"),
    })

    @app.get("/health")
    @limiter.limit("60/minute")
    async def health(request: Request): ...
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Default origins cover the local Next.js dev (3000), an alt port (3001),
# and FastAPI's own auto-generated docs (8000). Override in production via
# the ALLOWED_ORIGINS env var (comma-separated list).
_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:3001,http://localhost:8000"


# A single Limiter shared across all endpoints in the app. Memory-based by
# default — for multi-worker deployments swap `storage_uri` to Redis via
# the SLOWAPI_STORAGE env var.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"],
    storage_uri=os.getenv("SLOWAPI_STORAGE", "memory://"),
)


def create_app(
    *,
    title: str,
    description: str,
    version: str,
    allowed_methods: tuple[str, ...] = ("GET", "POST", "OPTIONS"),
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]]
    | Callable[[FastAPI], AsyncIterator[None]]
    | None = None,
) -> FastAPI:
    """Build a FastAPI app pre-wired with CORS + slowapi rate-limiting.

    Pass ``lifespan`` (an async context manager returned by
    ``contextlib.asynccontextmanager``) to run startup/shutdown logic. This
    is the modern replacement for the deprecated ``@app.on_event(...)`` API.
    """
    app = FastAPI(
        title=title, description=description, version=version, lifespan=lifespan
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    origins = os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_credentials=True,
        allow_methods=list(allowed_methods),
        allow_headers=["*"],
    )
    return app


def register_error_handlers(
    app: FastAPI,
    handlers: dict[type[Exception], tuple[int, str]],
    *,
    expose_message: tuple[type[Exception], ...] = (),
) -> None:
    """Register a generic JSON error handler for each exception class.

    Args:
        app: The FastAPI app.
        handlers: Map of `ExceptionClass -> (http_status, error_type_string)`.
            On an instance of that class being raised, the API returns
            `{"detail": "...", "error_type": "<error_type_string>"}`.
        expose_message: Exception classes whose `str(exc)` should be sent
            back to the caller (use only for validation errors etc.;
            internal errors should keep generic messages).
    """
    for exc_cls, (status_code, error_type) in handlers.items():
        # Capture in default args to avoid the late-binding closure trap.
        async def handler(request: Request, exc: Exception, *,
                           _status=status_code, _type=error_type,
                           _expose=exc_cls in expose_message) -> JSONResponse:
            detail = str(exc) if _expose else _default_message_for(_type)
            return JSONResponse(
                status_code=_status,
                content={"detail": detail, "error_type": _type},
            )

        app.add_exception_handler(exc_cls, handler)


def _default_message_for(error_type: str) -> str:
    """Generic non-leaking messages keyed by error_type."""
    return {
        "model_not_loaded": "Model service is not available. Please try again later.",
        "model_load_error": "Failed to load model. Please contact support.",
        "model_error": "An error occurred processing your request.",
        "data_error": "An error occurred processing data.",
        "configuration_error": "Service configuration error. Please contact support.",
        "validation_error": "Request validation failed.",
    }.get(error_type, "An error occurred processing your request.")
