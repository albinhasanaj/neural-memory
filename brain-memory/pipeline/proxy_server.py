"""
FastAPI proxy server — intercepts OpenAI-format chat requests and
injects brain-memory context before forwarding to the real LLM.

Usage::

    # Start the proxy server
    brain-proxy
    # or
    uvicorn pipeline.proxy_server:app --host 0.0.0.0 --port 8800

Then point any OpenAI-compatible chat UI at ``http://localhost:8800``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config.settings import settings
from pipeline.memory_pipeline import BrainMemoryPipeline

logger = logging.getLogger(__name__)

# ── Pipeline singleton ──────────────────────────────────────────────

_pipeline: BrainMemoryPipeline | None = None


def _get_pipeline() -> BrainMemoryPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BrainMemoryPipeline()
    return _pipeline


# ── FastAPI lifespan ────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start / stop the memory pipeline with the server."""
    logger.info("Starting Brain Memory proxy server on %s:%d", settings.proxy_host, settings.proxy_port)
    _get_pipeline()
    yield
    pipeline = _get_pipeline()
    pipeline.shutdown()
    logger.info("Brain Memory proxy server shut down.")


app = FastAPI(
    title="Brain Memory Proxy",
    description="OpenAI-compatible proxy that injects brain-inspired memory context.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── health check ────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── model listing (required by some UIs) ────────────────────────────


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Return a minimal model list so chat UIs can discover the proxy."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.llm_model,
                "object": "model",
                "owned_by": "brain-memory-proxy",
            }
        ],
    }


# ── chat completions ────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Intercept an OpenAI-format chat request, inject memory, forward, return.

    Steps
    -----
    1. Parse the incoming request body.
    2. Run the memory pipeline's ``pipe()`` to inject context.
    3. Forward the modified request to the real LLM API.
    4. Observe the LLM's response for future memory encoding.
    5. Return the response to the client.
    """
    body: dict[str, Any] = await request.json()
    pipeline = _get_pipeline()

    # 1-2. Memory observation + injection
    modified_body = pipeline.pipe(body)

    # 3. Determine target URL
    if settings.llm_provider == "openai":
        base = settings.llm_base_url or "https://api.openai.com/v1"
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
    elif settings.llm_provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        return JSONResponse(
            {"error": f"Unknown LLM provider: {settings.llm_provider}"},
            status_code=400,
        )

    is_stream = modified_body.get("stream", False)

    async with httpx.AsyncClient(timeout=120.0) as client:
        if is_stream:
            # Stream: forward as a streaming response
            async def _stream() -> AsyncGenerator[bytes, None]:
                import json as _json

                full_text_parts: list[str] = []
                async with client.stream(
                    "POST", url, headers=headers, json=modified_body
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                        # Parse SSE chunks to accumulate response text
                        try:
                            text = chunk.decode("utf-8", errors="ignore")
                            for line in text.splitlines():
                                line = line.strip()
                                if not line.startswith("data:"):
                                    continue
                                payload = line[len("data:"):].strip()
                                if payload == "[DONE]":
                                    continue
                                obj = _json.loads(payload)
                                # OpenAI format
                                for choice in obj.get("choices", []):
                                    delta = choice.get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        full_text_parts.append(content)
                                # Anthropic format
                                if obj.get("type") == "content_block_delta":
                                    delta = obj.get("delta", {})
                                    content = delta.get("text")
                                    if content:
                                        full_text_parts.append(content)
                        except Exception:
                            pass
                # Observe the assembled response
                full_text = "".join(full_text_parts)
                if full_text:
                    pipeline.observe_response(full_text)

            return StreamingResponse(_stream(), media_type="text/event-stream")

        else:
            # Non-stream: forward and return
            resp = await client.post(url, headers=headers, json=modified_body)
            resp_data = resp.json()

            # 4. Observe response
            try:
                if settings.llm_provider == "openai":
                    content = resp_data["choices"][0]["message"]["content"]
                else:
                    content = resp_data["content"][0]["text"]
                pipeline.observe_response(content)
            except (KeyError, IndexError):
                pass

            # 5. Return
            return JSONResponse(content=resp_data, status_code=resp.status_code)


# ── entry point ─────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for ``brain-proxy``."""
    import uvicorn

    uvicorn.run(
        "pipeline.proxy_server:app",
        host=settings.proxy_host,
        port=settings.proxy_port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
