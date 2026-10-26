"""MQTT subscriber that turns ``nv.Frame`` payloads into batched rows.

The recorder subscribes to the pipeline's ``mv3dt/<LOCATION_ID>/sv3d`` topic,
decodes each message with :func:`decode.decode_frame`, and hands the rows to a
``sink(rows)`` callable in batches. One flush thread owns every sink call; it
flushes every ``flush_interval_s`` or as soon as the buffer reaches
``max_batch`` rows. A bad payload or a failing sink is logged and counted,
never fatal. paho-mqtt reconnects automatically after a broker drop, and the
subscription is renewed on every (re)connect.

Constructor: ``Recorder(sink, topic, host, port, *, flush_interval_s=0.25,
max_batch=2000, client_id="", keepalive=30, client_factory=None)``.
``client_factory`` returns a paho-compatible client and exists for tests;
by default a paho-mqtt VERSION2 client is built inside ``start()``. A failed
``start()`` re-raises and leaves the recorder stopped, so it can be retried.

``stats()`` keys: ``connected``, ``messages`` (all received, including bad
ones), ``rows`` (decoded), ``rows_flushed`` (accepted by the sink),
``rows_dropped`` (lost to sink errors), ``rows_buffered`` (awaiting flush),
``decode_errors``, ``sink_errors``, ``last_message_ms`` (receive time of the
latest message, epoch ms, or None).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .decode import DecodeError, decode_frame

__all__ = ["Recorder"]

log = logging.getLogger(__name__)

Sink = Callable[[list], None]


def _now_ms() -> int:
    return int(time.time() * 1000)


class Recorder:
    """Subscribe to ``topic`` on ``host:port`` and feed decoded rows to ``sink``."""

    def __init__(
        self,
        sink: Sink,
        topic: str,
        host: str = "127.0.0.1",
        port: int = 1883,
        *,
        flush_interval_s: float = 0.25,
        max_batch: int = 2000,
        client_id: str = "",
        keepalive: int = 30,
        client_factory: Callable[[], object] | None = None,
    ) -> None:
        self._sink = sink
        self.topic = topic
        self.host = host
        self.port = port
        self.flush_interval_s = flush_interval_s
        self.max_batch = max_batch
        self._client_id = client_id
        self._keepalive = keepalive
        self._client_factory = client_factory

        self._lock = threading.Lock()
        self._buffer: list[dict] = []
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._flusher: threading.Thread | None = None
        self._client = None

        self._messages = 0
        self._rows = 0
        self._rows_flushed = 0
        self._rows_dropped = 0
        self._decode_errors = 0
        self._sink_errors = 0
        self._last_message_ms: int | None = None
        self._connected = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Connect the MQTT client, then start the flush thread.

        On failure nothing is left running and the error is re-raised, so a
        later ``start()`` can retry.
        """
        if self._flusher is not None:
            return
        client = self._client_factory() if self._client_factory else self._make_paho_client()
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        loop_started = False
        try:
            if hasattr(client, "reconnect_delay_set"):
                client.reconnect_delay_set(min_delay=1, max_delay=30)
            # connect_async + loop_start: a broker that is down at start-up is
            # retried by paho's loop rather than raising here.
            client.connect_async(self.host, self.port, keepalive=self._keepalive)
            client.loop_start()
            loop_started = True
            self._stopping.clear()
            self._wake.clear()
            flusher = threading.Thread(
                target=self._flush_loop, name="mv3dt-recorder-flush", daemon=True
            )
            flusher.start()
        except BaseException:
            if loop_started:
                try:
                    client.loop_stop()
                except Exception:  # noqa: BLE001
                    log.exception("error while cleaning up MQTT client")
            self._client = None
            self._flusher = None
            raise
        self._client = client
        self._flusher = flusher

    def stop(self, timeout: float = 5.0) -> None:
        """Disconnect, then flush whatever rows are still buffered."""
        self._stopping.set()
        client, self._client = self._client, None
        if client is not None:
            try:
                client.disconnect()
                client.loop_stop()
            except Exception:  # noqa: BLE001
                log.exception("error while stopping MQTT client")
        flusher, self._flusher = self._flusher, None
        self._wake.set()
        if flusher is not None:
            flusher.join(timeout)
        # The flush thread has exited, so this is still the only sink caller;
        # it picks up rows that arrived while the client was shutting down.
        if flusher is None or not flusher.is_alive():
            self._flush()

    def _make_paho_client(self):
        import paho.mqtt.client as mqtt  # lazy: decode tests need no paho

        return mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self._client_id,
        )

    # -- MQTT callbacks (paho VERSION2 signatures) -------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "is_failure", False):
            log.warning("MQTT connect to %s:%s failed: %s", self.host, self.port, reason_code)
            return
        self._connected = True
        log.info("MQTT connected to %s:%s, subscribing to %s", self.host, self.port, self.topic)
        client.subscribe(self.topic, qos=0)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self._connected = False
        if not self._stopping.is_set():
            log.warning("MQTT disconnected (%s); paho will reconnect", reason_code)

    def _on_message(self, client, userdata, msg) -> None:
        recv_ms = _now_ms()
        try:
            rows = decode_frame(msg.payload, recv_ms=recv_ms)
        except DecodeError as exc:
            with self._lock:
                self._messages += 1
                self._decode_errors += 1
                self._last_message_ms = recv_ms
            log.warning("dropping undecodable payload on %s: %s", getattr(msg, "topic", "?"), exc)
            return
        except Exception:  # noqa: BLE001  never let a payload kill the loop
            with self._lock:
                self._messages += 1
                self._decode_errors += 1
                self._last_message_ms = recv_ms
            log.exception("unexpected error decoding payload")
            return
        with self._lock:
            self._messages += 1
            self._last_message_ms = recv_ms
            self._rows += len(rows)
            self._buffer.extend(rows)
            full = len(self._buffer) >= self.max_batch
        if full:
            self._wake.set()

    # -- flushing ----------------------------------------------------------

    def _flush_loop(self) -> None:
        while not self._stopping.is_set():
            self._wake.wait(self.flush_interval_s)
            self._wake.clear()
            self._flush()
        self._flush()

    def _flush(self) -> None:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return
        try:
            self._sink(batch)
        except Exception:  # noqa: BLE001  a failing sink is counted, not fatal
            with self._lock:
                self._sink_errors += 1
                self._rows_dropped += len(batch)
            log.exception("sink failed; dropped %d rows", len(batch))
            return
        with self._lock:
            self._rows_flushed += len(batch)

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected,
                "messages": self._messages,
                "rows": self._rows,
                "rows_flushed": self._rows_flushed,
                "rows_dropped": self._rows_dropped,
                "rows_buffered": len(self._buffer),
                "decode_errors": self._decode_errors,
                "sink_errors": self._sink_errors,
                "last_message_ms": self._last_message_ms,
            }
