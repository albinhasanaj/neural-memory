"""FastAPI server for 3D brain visualization."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

app = FastAPI()

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "brain_viz.html")


# WebSocket connections
connections: list[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            # Keep connection alive, receive any client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connections:
            connections.remove(websocket)


async def broadcast_event(event_data: dict):
    """Send event to all connected visualization clients."""
    message = json.dumps(event_data)
    dead = []
    for ws in connections:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connections:
            connections.remove(ws)


def start_viz_server(event_bus, observer, host="127.0.0.1", port=7860):
    """Start the visualization server with event bus integration."""
    import uvicorn

    # Bridge: event_bus → WebSocket broadcast
    loop = asyncio.new_event_loop()

    def on_event(event):
        """Called by event_bus when a neural event occurs."""
        event_data = {
            "type": event.event_type,
            "timestamp": event.timestamp,
            "data": {},
        }
        # Serialize event data (convert non-JSON-serializable types)
        for k, v in event.data.items():
            if isinstance(v, (str, int, float, bool, list)):
                event_data["data"][k] = v
            elif isinstance(v, dict):
                event_data["data"][k] = v
            else:
                event_data["data"][k] = str(v)

        # Include module summary data for write/retrieve/gate events
        if event.event_type in ("write", "retrieve", "gate_decision"):
            try:
                hopfield = getattr(observer, "_hopfield", None)
                if hopfield is not None and hasattr(hopfield, "module_summary"):
                    summaries = hopfield.module_summary()
                    event_data["modules"] = [
                        {
                            "id": s["module_index"],
                            "write_count": s.get("write_count", 0),
                            "w_key_norm": s.get("w_key_norm", 0.0),
                            "occupancy": s.get("occupancy", 0.0),
                        }
                        for s in summaries
                    ]
            except Exception:
                pass

        asyncio.run_coroutine_threadsafe(broadcast_event(event_data), loop)

    event_bus.subscribe(on_event)

    def run_server():
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(
            app, host=host, port=port, log_level="warning", loop="asyncio"
        )
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread
