"""xAI realtime STT service for the benchmark pipeline."""

import asyncio
import json
from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

_READY_TIMEOUT_SECS = 10.0


class XAIRealtimeSTTService(STTService):
    """Streaming STT service backed by xAI's `/v1/stt` websocket API."""

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "wss://api.x.ai/v1/stt",
        language: Language | None = Language.EN,
        interim_results: bool = True,
        endpointing_ms: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._aiohttp_session = aiohttp_session
        self._base_url = base_url
        self._language = self._language_to_code(language) if language else None
        self._interim_results = interim_results
        self._endpointing_ms = endpointing_ms

        self.set_model_name("grok-stt")

        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._session_error: Exception | None = None
        self._last_final_text: str | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio:
            yield None
            return

        await self._wait_until_ready()
        if self._session_error:
            raise self._session_error

        assert self._websocket is not None
        await self._websocket.send_bytes(audio)
        yield None

    async def set_language(self, language: Language):
        self._language = self._language_to_code(language)

    async def _connect(self) -> None:
        self._ready.clear()
        self._session_error = None
        self._last_final_text = None

        params = {
            "sample_rate": str(self.sample_rate),
            "encoding": "pcm",
            "interim_results": "true" if self._interim_results else "false",
            "endpointing": str(self._endpointing_ms),
        }
        if self._language:
            params["language"] = self._language

        self._websocket = await self._aiohttp_session.ws_connect(
            self._base_url,
            params=params,
            headers={"Authorization": f"Bearer {self._api_key}"},
            autoping=True,
            heartbeat=None,
        )
        self._receive_task = self.create_task(self._receive_messages(), name="xai_stt_receive")
        await self._wait_until_ready()

    async def _disconnect(self) -> None:
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None

        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
        self._websocket = None
        self._ready.clear()

    async def _wait_until_ready(self) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=_READY_TIMEOUT_SECS)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Timed out waiting for xAI STT websocket readiness") from exc

    async def _receive_messages(self) -> None:
        assert self._websocket is not None

        async for message in self._websocket:
            if message.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(message.data)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse xAI STT websocket message")
                    continue
                await self._handle_event(event)
            elif message.type == aiohttp.WSMsgType.ERROR:
                error = self._websocket.exception()
                if error:
                    self._session_error = error
                    self._ready.set()
                    await self.push_error(error_msg=f"xAI STT websocket error: {error}")
                break
            elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                break

    async def _handle_event(self, event: dict) -> None:
        event_type = event.get("type", "")

        if event_type == "transcript.created":
            self._ready.set()
            return
        if event_type == "transcript.partial":
            await self._handle_transcript_partial(event)
            return
        if event_type == "transcript.done":
            await self._handle_transcript_done(event)
            return
        if event_type == "error":
            await self._handle_error(event)
            return

        logger.trace(f"Unhandled xAI STT event: {event_type}")

    async def _handle_transcript_partial(self, event: dict) -> None:
        text = self._extract_text(event)
        if not text:
            return

        is_final = bool(event.get("is_final", False))
        speech_final = bool(event.get("speech_final", False))

        if not is_final:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    result=event,
                )
            )
            return

        frame = TranscriptionFrame(
            text,
            self._user_id,
            time_now_iso8601(),
            result=event,
        )
        frame.finalized = speech_final
        await self.push_frame(frame)
        self._last_final_text = text

    async def _handle_transcript_done(self, event: dict) -> None:
        text = self._extract_text(event)
        if text and text != self._last_final_text:
            frame = TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                result=event,
            )
            frame.finalized = True
            await self.push_frame(frame)
            self._last_final_text = text

    async def _handle_error(self, event: dict) -> None:
        error = event.get("error", event)
        if isinstance(error, dict):
            message = error.get("message", "Unknown xAI STT error")
            code = error.get("code", "")
            error_message = f"xAI STT error [{code}]: {message}" if code else f"xAI STT error: {message}"
        else:
            error_message = f"xAI STT error: {error}"

        self._session_error = RuntimeError(error_message)
        self._ready.set()
        await self.push_error(error_msg=error_message)

    @staticmethod
    def _extract_text(event: dict) -> str:
        for key in ("text", "transcript", "delta"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _language_to_code(language: Language) -> str:
        return str(language).split("-")[0].lower()
