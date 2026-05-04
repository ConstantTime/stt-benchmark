"""Speechmatics batch transcription via REST.

Pipecat ships only the Speechmatics realtime (WebSocket) service. The Noteless
prospect is moving to batch processing, so the benchmark needs batch numbers.
This module talks to the Speechmatics batch REST API directly:

  1. POST /v2/jobs (multipart audio + JSON config) -> job_id
  2. GET /v2/jobs/{id}                              -> poll until status = done
  3. GET /v2/jobs/{id}/transcript?format=txt        -> plain transcript

Results are written into the same `results` table the rest of the pipeline
uses (service_name = 'speechmatics_batch'), so `wer` and `report` work
unchanged.
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

DEFAULT_BATCH_URL = "https://asr.api.speechmatics.com/v2"
POLL_INTERVAL_SECS = 2.0
JOB_TIMEOUT_SECS = 300.0  # 5 min per clip; FLEURS clips are 10-15s so plenty


# Speechmatics expects ISO 639-1 codes. They use "no" for Norwegian.
_LANG_TO_SPEECHMATICS: dict[Language, str] = {
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


def speechmatics_lang_code(language: Language) -> str:
    code = _LANG_TO_SPEECHMATICS.get(language)
    if code is None:
        raise ValueError(f"No Speechmatics language code mapped for {language}")
    return code


class SpeechmaticsBatchRunner:
    """Submits audio samples to the Speechmatics batch API in parallel."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        operating_point: str = "enhanced",  # "standard" or "enhanced"
        max_concurrency: int = 8,
    ):
        self.config = config or get_config()
        if not self.config.speechmatics_api_key:
            raise ValueError("SPEECHMATICS_API_KEY not set")

        # Match what the realtime factory reads, with a sensible REST default.
        import os
        self.base_url = (
            os.getenv("SPEECHMATICS_BATCH_URL")
            or DEFAULT_BATCH_URL
        )
        self.api_key = self.config.speechmatics_api_key
        self.operating_point = operating_point
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _submit(
        self,
        session: aiohttp.ClientSession,
        wav_bytes: bytes,
        lang_code: str,
    ) -> str:
        """Submit a job and return its id."""
        config_json = (
            '{"type": "transcription", "transcription_config": '
            f'{{"language": "{lang_code}", "operating_point": "{self.operating_point}"}}}}'
        )
        data = aiohttp.FormData()
        data.add_field("data_file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        data.add_field("config", config_json, content_type="application/json")
        async with session.post(
            f"{self.base_url}/jobs",
            data=data,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ) as resp:
            if resp.status not in (200, 201):
                body = await resp.text()
                raise RuntimeError(f"submit failed status={resp.status} body={body[:300]}")
            payload = await resp.json()
            return payload["id"]

    async def _wait_for_done(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> None:
        """Poll until the job leaves the running state."""
        deadline = time.monotonic() + JOB_TIMEOUT_SECS
        headers = {"Authorization": f"Bearer {self.api_key}"}
        while time.monotonic() < deadline:
            async with session.get(f"{self.base_url}/jobs/{job_id}", headers=headers) as r:
                if r.status != 200:
                    body = await r.text()
                    raise RuntimeError(f"poll failed status={r.status} body={body[:300]}")
                payload = await r.json()
                status = payload.get("job", {}).get("status")
                if status == "done":
                    return
                if status == "rejected":
                    err = payload.get("job", {}).get("errors", "rejected")
                    raise RuntimeError(f"job rejected: {err}")
            await asyncio.sleep(POLL_INTERVAL_SECS)
        raise asyncio.TimeoutError(f"job {job_id} not done after {JOB_TIMEOUT_SECS}s")

    async def _fetch_transcript(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> str:
        async with session.get(
            f"{self.base_url}/jobs/{job_id}/transcript",
            params={"format": "txt"},
            headers={"Authorization": f"Bearer {self.api_key}"},
        ) as r:
            if r.status != 200:
                body = await r.text()
                raise RuntimeError(f"fetch failed status={r.status} body={body[:300]}")
            text = await r.text()
            return text.strip()

    async def transcribe_sample(
        self,
        sample: AudioSample,
        session: aiohttp.ClientSession,
    ) -> BenchmarkResult:
        """Transcribe a single sample. Returns BenchmarkResult (success or error)."""
        async with self._sem:
            t0 = time.monotonic()
            try:
                pcm_bytes = Path(sample.audio_path).read_bytes()
                wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1)
                lang_code = speechmatics_lang_code(parse_language(sample.language))
                job_id = await self._submit(session, wav_bytes, lang_code)
                await self._wait_for_done(session, job_id)
                transcript = await self._fetch_transcript(session, job_id)
                elapsed = time.monotonic() - t0
                return BenchmarkResult(
                    sample_id=sample.sample_id,
                    service_name=ServiceName.SPEECHMATICS_BATCH,
                    model_name=self.operating_point,
                    ttfb_seconds=elapsed,  # for batch this is total wall time
                    transcription=transcript,
                    audio_duration_seconds=sample.duration_seconds,
                    timestamp=datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.error(f"speechmatics_batch {sample.sample_id}: {e}")
                return BenchmarkResult(
                    sample_id=sample.sample_id,
                    service_name=ServiceName.SPEECHMATICS_BATCH,
                    model_name=self.operating_point,
                    audio_duration_seconds=sample.duration_seconds,
                    error=str(e)[:500],
                    timestamp=datetime.now(timezone.utc),
                )

    async def run(
        self,
        samples: list[AudioSample],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[BenchmarkResult]:
        """Transcribe all samples in parallel, return BenchmarkResult list."""
        results: list[BenchmarkResult] = []
        completed = 0

        async with aiohttp.ClientSession() as session:
            async def _one(sample: AudioSample) -> None:
                nonlocal completed
                r = await self.transcribe_sample(sample, session)
                results.append(r)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(samples), sample.sample_id)

            await asyncio.gather(*(_one(s) for s in samples))

        return results


async def run_and_persist(
    db: Database,
    samples: list[AudioSample],
    operating_point: str = "enhanced",
    max_concurrency: int = 8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[BenchmarkResult]:
    """Convenience: transcribe samples and write each result to the DB."""
    runner = SpeechmaticsBatchRunner(
        operating_point=operating_point,
        max_concurrency=max_concurrency,
    )
    results: list[BenchmarkResult] = []

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(sample: AudioSample) -> None:
            async with sem:
                r = await runner.transcribe_sample(sample, session)
            results.append(r)
            await db.insert_result(r)
            if progress_callback:
                progress_callback(len(results), len(samples), sample.sample_id)

        await asyncio.gather(*(_one(s) for s in samples))

    return results
