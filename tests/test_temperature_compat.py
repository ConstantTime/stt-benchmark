"""The judge must not be pinned to models that accept `temperature`.

Newer models (e.g. opus-4.x) return a 400 'temperature is deprecated for this
model'. _api_call should drop the param and retry, then stop sending it.
"""
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest

from stt_benchmark.evaluation.semantic_wer import SemanticWEREvaluator


def _temp_400():
    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com"))
    return anthropic.BadRequestError(
        "temperature is deprecated for this model", response=resp, body=None
    )


@pytest.mark.asyncio
async def test_drops_temperature_and_retries():
    ev = SemanticWEREvaluator(db_path=Path(tempfile.mkdtemp()) / "e.db", max_concurrency=1)
    ok = object()
    ev.client.messages.create = AsyncMock(side_effect=[_temp_400(), ok])

    result = await ev._api_call({"model": "claude-opus-4-8", "temperature": 0, "messages": []}, "f")

    assert result is ok
    assert ev.client.messages.create.await_count == 2
    # the retry must not include temperature
    assert "temperature" not in ev.client.messages.create.await_args_list[1].kwargs
    assert ev._temperature_unsupported is True


@pytest.mark.asyncio
async def test_other_400_still_raises():
    ev = SemanticWEREvaluator(db_path=Path(tempfile.mkdtemp()) / "e.db", max_concurrency=1)
    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com"))
    ev.client.messages.create = AsyncMock(side_effect=anthropic.BadRequestError(
        "max_tokens too large", response=resp, body=None))

    with pytest.raises(anthropic.BadRequestError):
        await ev._api_call({"model": "x", "temperature": 0, "messages": []}, "f")
