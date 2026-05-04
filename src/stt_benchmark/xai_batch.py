"""xAI realtime STT, used in 'batch' style for the multilingual benchmark.

xAI exposes a batch HTTP STT endpoint at /v1/audio/transcriptions, but our
team's API key is currently scoped out of it (HTTP 403 'Team is not
authorized'). The realtime websocket at /v1/stt is authorized, so this module
opens a fresh WS per sample and runs many in parallel — same pattern as the
Scribe and Speechmatics batch runners.

Writes results to the `results` table with service_name='xai_batch' so the
regular `wer` and `report` commands consume them.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from loguru import logger
from pipecat.transcriptions.language import Language

from stt_benchmark.config import BenchmarkConfig, get_config
from stt_benchmark.models import AudioSample, BenchmarkResult, ServiceName
from stt_benchmark.services import parse_language
from stt_benchmark.storage.database import Database

XAI_STT_WS = "wss://api.x.ai/v1/stt"
DEFAULT_MODEL = "grok-stt"
SEND_CHUNK_MS = 200          # ~6.4 KB per chunk at 16kHz int16
SEND_PACE_FACTOR = 5.0       # send 5x faster than realtime
READY_TIMEOUT_SECS = 10.0
QUIET_DEADLINE_SECS = 2.5    # consider session done after this many sec of WS silence
SESSION_TIMEOUT_SECS = 60.0  # absolute cap per sample
TRAILING_SILENCE_MS = 1500   # pad audio so endpointer fires


# xAI accepts ISO 639-1 codes (e.g. "en", "no", "da"). Pipecat's language
# values like "nb-NO" need the head split off.
_LANG_TO_XAI: dict[Language, str] = {
    Language.EN: "en",
    Language.EN_US: "en",
    Language.NB: "no",
    Language.NB_NO: "no",
    Language.NO: "no",
    Language.DA: "da",
    Language.DA_DK: "da",
    Language.DE: "de",
    Language.DE_DE: "de",
    Language.FR: "fr",
    Language.FR_FR: "fr",
    Language.IT: "it",
    Language.IT_IT: "it",
    Language.ES: "es",
    Language.ES_ES: "es",
    Language.ES_419: "es",
}


def xai_lang_code(language: Language) -> str:
    code = _LANG_TO_XAI.get(language)
    if code is None:
        # Fall back to the 2-letter prefix; xAI may or may not accept it.
        return str(language).split("-")[0].lower()
    return code


class _XAISession:
    """One WS session per sample.

    xAI's WS protocol returns many `transcript.partial` events; ones with
    `is_final=true` are committed phrase-level transcripts. There's no
    `transcript.done` for the whole session, so we collect every final
    segment and decide we're done when no new events arrive for a while.
    """

    def __init__(self, ws: aiohttp.ClientWebSocketResponse):
        self._ws = ws
        self._ready = asyncio.Event()
        self._final_segments: list[str] = []
        self._error: str | None = None
        self._closed = False
        self._last_event_at: float = 0.0

    async def receive_loop(self) -> None:
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        ev = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    t = ev.get("type", "")
                    self._last_event_at = time.monotonic()
                    if t == "transcript.created":
                        self._ready.set()
                    elif t == "transcript.partial":
                        if ev.get("is_final"):
                            txt = (ev.get("text") or ev.get("transcript") or "").strip()
                            if txt:
                                self._final_segments.append(txt)
                    elif t == "transcript.done":
                        # If xAI ever sends this, treat it as a hard end.
                        txt = (ev.get("text") or ev.get("transcript") or "").strip()
                        if txt:
                            self._final_segments.append(txt)
                        self._closed = True
                        return
                    elif t == "error":
                        err = ev.get("error", ev)
                        if isinstance(err, dict):
                            self._error = err.get("message") or json.dumps(err)[:200]
                        else:
                            self._error = str(err)[:200]
                        self._closed = True
                        self._ready.set()
                        return
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    self._closed = True
                    return
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self._error = f"ws error: {self._ws.exception()}"
                    self._closed = True
                    return
        except Exception as e:
            self._error = f"recv exception: {e}"
            self._closed = True

    @property
    def transcript(self) -> str:
        return " ".join(self._final_segments).strip()


class XAIBatchRunner:
    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        max_concurrency: int = 4,
        model: str = DEFAULT_MODEL,
    ):
        self.config = config or get_config()
        if not self.config.xai_api_key:
            raise ValueError("XAI_API_KEY not set")
        self.api_key = self.config.xai_api_key
        self.model = model
        self._sem = asyncio.Semaphore(max_concurrency)

    async def transcribe_sample(
        self,
        sample: AudioSample,
        session: aiohttp.ClientSession,
    ) -> BenchmarkResult:
        async with self._sem:
            t0 = time.monotonic()
            try:
                pcm = Path(sample.audio_path).read_bytes()
                lang = xai_lang_code(parse_language(sample.language))
                params = {
                    "sample_rate": "16000",
                    "encoding": "pcm",
                    "interim_results": "false",
                    "endpointing": "10",
                    "language": lang,
                }
                ws = await session.ws_connect(
                    XAI_STT_WS,
                    params=params,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    autoping=True,
                    heartbeat=None,
                )
                sess = _XAISession(ws)
                recv_task = asyncio.create_task(sess.receive_loop())

                try:
                    await asyncio.wait_for(sess._ready.wait(), timeout=READY_TIMEOUT_SECS)
                    if sess._error:
                        raise RuntimeError(sess._error)

                    # Stream PCM in chunks faster than realtime, then pad
                    # with silence so xAI's endpointer fires the last segment.
                    bytes_per_chunk = int(16000 * 2 * SEND_CHUNK_MS / 1000)  # 16-bit
                    silence_bytes = b"\x00" * int(16000 * 2 * TRAILING_SILENCE_MS / 1000)
                    payload = pcm + silence_bytes
                    sleep_per_chunk = (SEND_CHUNK_MS / 1000) / SEND_PACE_FACTOR
                    for i in range(0, len(payload), bytes_per_chunk):
                        await ws.send_bytes(payload[i:i + bytes_per_chunk])
                        if sleep_per_chunk > 0:
                            await asyncio.sleep(sleep_per_chunk)

                    # xAI doesn't send a session-end event, so wait until the
                    # WS has been quiet for QUIET_DEADLINE_SECS (or hit the
                    # absolute cap).
                    sess._last_event_at = time.monotonic()
                    deadline = time.monotonic() + SESSION_TIMEOUT_SECS
                    while time.monotonic() < deadline:
                        if sess._closed:
                            break
                        if time.monotonic() - sess._last_event_at >= QUIET_DEADLINE_SECS:
                            break
                        await asyncio.sleep(0.2)
                finally:
                    recv_task.cancel()
                    try:
                        await recv_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    if not ws.closed:
                        await ws.close()

                if sess._error:
                    raise RuntimeError(sess._error or "unknown xAI session error")
                transcript = sess.transcript
                if not transcript:
                    raise RuntimeError(
                        f"xai returned no final segments (ready={sess._ready.is_set()}, "
                        f"closed={sess._closed})"
                    )

                elapsed = time.monotonic() - t0
                return BenchmarkResult(
                    sample_id=sample.sample_id,
                    service_name=ServiceName.XAI_BATCH,
                    model_name=self.model,
                    ttfb_seconds=elapsed,
                    transcription=transcript,
                    audio_duration_seconds=sample.duration_seconds,
                    timestamp=datetime.now(timezone.utc),
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                logger.error(f"xai_batch {sample.sample_id}: {err}")
                return BenchmarkResult(
                    sample_id=sample.sample_id,
                    service_name=ServiceName.XAI_BATCH,
                    model_name=self.model,
                    audio_duration_seconds=sample.duration_seconds,
                    error=err[:500],
                    timestamp=datetime.now(timezone.utc),
                )


async def run_and_persist(
    db: Database,
    samples: list[AudioSample],
    max_concurrency: int = 4,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[BenchmarkResult]:
    runner = XAIBatchRunner(max_concurrency=max_concurrency)
    results: list[BenchmarkResult] = []
    async with aiohttp.ClientSession() as session:
        async def _one(s: AudioSample) -> None:
            r = await runner.transcribe_sample(s, session)
            results.append(r)
            await db.insert_result(r)
            if progress_callback:
                progress_callback(len(results), len(samples), s.sample_id)
        await asyncio.gather(*(_one(s) for s in samples))
    return results
