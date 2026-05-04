"""Download FLEURS audio + reference transcripts for our 6 target languages.

FLEURS (https://huggingface.co/datasets/google/fleurs) is Google's multilingual
speech eval dataset. Each sample is ~10s of read speech with a human-verified
reference transcript. This sidesteps the bias problem of generating ground truth
with one of the systems we're benchmarking.

For each language subset:
  - load test split (smallest, intended for eval)
  - take the first --limit-per-lang samples
  - convert audio float32 [-1, 1] -> int16 PCM little-endian @ 16kHz mono
  - write PCM file to stt_benchmark_data/audio/<sample_id>.pcm
  - insert sample into `samples` table with language=<subset>
  - insert reference transcript into `ground_truth` table with model_used="fleurs"

Sample ids are namespaced as `fleurs_<subset>_<orig_id>` to avoid collision with
the existing English smart-turn-data samples.

Usage:
  uv run python scripts/download_fleurs.py --limit-per-lang 200
  uv run python scripts/download_fleurs.py --limit-per-lang 50 --languages nb_no,da_dk
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset
from loguru import logger

from stt_benchmark.config import get_config
from stt_benchmark.models import AudioSample, GroundTruth
from stt_benchmark.storage.database import Database

DEFAULT_LANGUAGES = ["nb_no", "da_dk", "de_de", "fr_fr", "it_it", "es_es"]


def _float_to_pcm16(arr: np.ndarray) -> bytes:
    """Convert FLEURS float audio (-1..1) to little-endian int16 PCM bytes."""
    a = np.asarray(arr, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


async def load_one_language(
    db: Database,
    audio_dir: Path,
    subset: str,
    limit: int,
    dataset_index_offset: int,
) -> tuple[int, int]:
    """Load one FLEURS language subset.

    Returns (samples_inserted, gts_inserted).
    """
    logger.info(f"Loading FLEURS subset '{subset}' (test split, limit={limit})")
    ds = load_dataset(
        "google/fleurs",
        subset,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    samples: list[AudioSample] = []
    gts: list[GroundTruth] = []

    for i, ex in enumerate(ds):
        if i >= limit:
            break

        orig_id = ex.get("id")
        sample_id = f"fleurs_{subset}_{orig_id}"
        audio = ex.get("audio") or {}
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is None or sr != 16000:
            logger.warning(f"Skipping {sample_id}: bad audio (sr={sr})")
            continue

        pcm = _float_to_pcm16(arr)
        out_path = audio_dir / f"{sample_id}.pcm"
        out_path.write_bytes(pcm)

        duration = len(arr) / sr
        ref = (ex.get("transcription") or "").strip()
        if not ref:
            logger.warning(f"Skipping {sample_id}: empty transcription")
            continue

        samples.append(
            AudioSample(
                sample_id=sample_id,
                audio_path=str(out_path),
                duration_seconds=duration,
                language=subset,
                dataset_index=dataset_index_offset + i,
            )
        )
        gts.append(
            GroundTruth(
                sample_id=sample_id,
                text=ref,
                model_used="fleurs",
                generated_at=datetime.now(timezone.utc),
                verified_by="fleurs_dataset",
                verified_at=datetime.now(timezone.utc),
                original_text=None,
            )
        )

    if samples:
        await db.insert_samples_batch(samples)
        for gt in gts:
            await db.insert_ground_truth(gt)
    return len(samples), len(gts)


async def amain() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help=f"Comma-separated FLEURS subsets (default: {','.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument(
        "--limit-per-lang",
        type=int,
        default=200,
        help="Max samples per language (default: 200)",
    )
    args = parser.parse_args()

    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    config = get_config()
    config.ensure_dirs()
    audio_dir = config.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    db = Database()
    await db.initialize()

    total_samples = 0
    total_gts = 0
    # Use a high offset to avoid colliding with the English smart-turn dataset_indexes
    base_offset = 1_000_000

    for i, subset in enumerate(languages):
        try:
            n_s, n_g = await load_one_language(
                db,
                audio_dir,
                subset,
                args.limit_per_lang,
                base_offset + i * 100_000,
            )
            logger.info(f"  {subset}: inserted {n_s} samples, {n_g} ground truths")
            total_samples += n_s
            total_gts += n_g
        except Exception as e:
            logger.error(f"  {subset}: FAILED — {e}")

    await db.close()
    logger.info(
        f"Done: {total_samples} samples and {total_gts} ground truths across "
        f"{len(languages)} languages."
    )


if __name__ == "__main__":
    asyncio.run(amain())
