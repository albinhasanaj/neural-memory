"""Event system for neural activity visualization."""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable


@dataclass
class NeuralEvent:
    """A single neural activity event."""
    timestamp: float
    event_type: str  # 'encode', 'gate_decision', 'write', 'retrieve', 'activate', 'inject', 'consolidate', 'forget'
    data: dict[str, Any] = field(default_factory=dict)


class NeuralEventBus:
    """Thread-safe event bus for neural activity."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._events: deque[NeuralEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._listeners: list[Callable[[NeuralEvent], Any]] = []

    def emit(self, event_type: str, **data: Any) -> None:
        event = NeuralEvent(timestamp=time.time(), event_type=event_type, data=data)
        with self._lock:
            self._events.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

    def subscribe(self, listener: Callable[[NeuralEvent], Any]) -> None:
        self._listeners.append(listener)

    def recent(self, n: int = 50) -> list[NeuralEvent]:
        with self._lock:
            return list(self._events)[-n:]

    def events_since(self, timestamp: float) -> list[NeuralEvent]:
        with self._lock:
            return [e for e in self._events if e.timestamp > timestamp]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


# Global singleton
event_bus = NeuralEventBus()
