"""Exception handlers.

* Validation errors never echo the submitted input back (v2-style 422s would reflect passwords
  and camera URLs into responses): only location, message and type are returned.
* Unhandled exceptions are handled by app.middleware.CatchAllMiddleware (logged with a stack
  trace, answered with a generic 500 + request id, still carrying the security headers).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def _validation(_: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    safe = [
        {"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))} for e in errors
    ]
    return JSONResponse(status_code=422, content={"detail": "validation_error", "errors": safe})


def install(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation)
