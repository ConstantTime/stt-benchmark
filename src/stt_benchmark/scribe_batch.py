"""ElevenLabs Scribe v2 batch transcription via REST.

We already have `elevenlabs_http` as a Pipecat-wrapped batch service, but the
Pipecat pipeline runs samples one at a time (each with full pipeline setup),
which made a 1000-sample multilingual run take ~8 hours wall clock. This module
hits the same REST endpoint directly with a concurrency semaphore — same model,
same audio, just async parallel — turning the run into ~5 minutes.

Writes results to the `results` table with service_name='elevenlabs_batch' so
the regular `wer` and `report` commands consume them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from loguru import logger
from pipecat.transcriptions.language import Language

from stt_benchmark.config import BenchmarkConfig, get_config
from stt_benchmark.ground_truth.gemini_transcriber import pcm_to_wav
from stt_benchmark.models import AudioSample, BenchmarkResult, ServiceName
from stt_benchmark.services import parse_language
from stt_benchmark.storage.database import Database

SCRIBE_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_MODEL = "scribe_v2"


# Scribe v2 expects ISO 639-3 codes (e.g. "eng", "nor", "dan").
_LANG_TO_SCRIBE: dict[Language, str] = {
    Language.EN: "eng",
    Language.EN_US: "eng",
    Language.NB: "nor",
    Language.NB_NO: "nor",
    Language.NO: "nor",
    Language.DA: "dan",
    Language.DA_DK: "dan",
    Language.DE: "deu",
    Language.DE_DE: "deu",
    Language.FR: "fra",
    Language.FR_FR: "fra",
    Language.IT: "ita",
    Language.IT_IT: "ita",
    Language.ES: "spa",
    Language.ES_ES: "spa",
    Language.ES_419: "spa",
}


def scribe_lang_code(language: Language) -> str:
    code = _LANG_TO_SCRIBE.get(language)
    if code is None:
        raise ValueError(f"No Scribe language code mapped for {language}")
    return code


class ScribeBatchRunner:
    """Submits audio samples to Scribe v2 batch HTTP API in parallel."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        model: str = DEFAULT_MODEL,
        max_concurrency: int = 8,
    ):
        self.config = config or get_config()
        if not self.config.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")
        self.api_key = self.config.elevenlabs_api_key
        self.model = model
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _transcribe(
        self,
        session: aiohttp.ClientSession,
        wav_bytes: bytes,
        lang_code: str,
    ) -> str:
        data = aiohttp.FormData()
        data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        data.add_field("model_id", self.model)
        data.add_field("language_code", lang_code)
        data.add_field("tag_audio_events", "false")
        data.add_field("timestamps_granularity", "none")
        async with session.post(
            SCRIBE_API_URL,
            data=data,
            headers={"xi-api-key": self.api_key},
        ) as r:
            if r.status != 200:
                body = await r.text()
                raise RuntimeError(f"status={r.status} body={body[:300]}")
            payload = await r.json()
            return (payload.get("text") or "").strip()

    async def transcribe_sample(
        self,
        sample: AudioSample,
        session: aiohttp.ClientSession,
    ) -> BenchmarkResult:
        async with self._sem:
            t0 = time.monotonic()
            try:
                pcm_bytes = Path(sample.audio_path).read_bytes()
                wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1)
                lang_code = scribe_lang_code(parse_language(sample.language))
                text = await self._transcribe(session, wav_bytes, lang_code)
                elapsed = time.monotonic() - t0
                return BenchmarkResult(
                    sample_id=sample.sample_id,
                    service_name=ServiceName.ELEVENLABS_BATCH,
                    model_name=self.model,
                    ttfb_seconds=elapsed,
                    transcription=text,
                    audio_duration_seconds=sample.duration_seconds,
                    timestamp=datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.error(f"elevenlabs_batch {sample.sample_id}: {e}")
                return BenchmarkResult(
                    sample_id=sample.sample_id,
                    service_name=ServiceName.ELEVENLABS_BATCH,
                    model_name=self.model,
                    audio_duration_seconds=sample.duration_seconds,
                    error=str(e)[:500],
                    timestamp=datetime.now(timezone.utc),
                )


async def run_and_persist(
    db: Database,
    samples: list[AudioSample],
    model: str = DEFAULT_MODEL,
    max_concurrency: int = 8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[BenchmarkResult]:
    runner = ScribeBatchRunner(model=model, max_concurrency=max_concurrency)
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
