"""CLI for running batch (non-realtime) STT services that don't fit Pipecat's pipeline.

Right now this only knows about `speechmatics_batch`, but the structure is set up
so other batch REST services (e.g. an OpenAI Whisper batch path, AssemblyAI batch)
can be added later.
"""

import asyncio

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from stt_benchmark.config import get_config
from stt_benchmark.models import ServiceName
from stt_benchmark.scribe_batch import run_and_persist as scribe_run_and_persist
from stt_benchmark.speechmatics_batch import run_and_persist as speechmatics_run_and_persist
from stt_benchmark.storage.database import Database
from stt_benchmark.xai_batch import run_and_persist as xai_run_and_persist

_BATCH_SERVICES = {"speechmatics_batch", "elevenlabs_batch", "xai_batch"}

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def run_batch(
    services: str = typer.Option(
        "speechmatics_batch,elevenlabs_batch",
        "--services",
        "-s",
        help="Comma-separated list of batch services (speechmatics_batch, elevenlabs_batch)",
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Limit number of samples"
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="ISO code (e.g. 'no', 'da_dk', 'eng'). When set, only samples with that "
        "language are transcribed. Default: all samples.",
    ),
    operating_point: str = typer.Option(
        "enhanced",
        "--operating-point",
        help="Speechmatics operating point: 'standard' or 'enhanced' (default: enhanced)",
    ),
    concurrency: int = typer.Option(
        8, "--concurrency", "-c", help="Concurrent in-flight jobs"
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--no-skip-existing", help="Skip samples already transcribed"
    ),
):
    """Run batch STT services that talk to REST APIs (no Pipecat pipeline)."""
    requested = [s.strip().lower() for s in services.split(",") if s.strip()]
    unknown = [s for s in requested if s not in _BATCH_SERVICES]
    if unknown:
        console.print(
            f"[red]Unsupported batch services: {unknown}. "
            f"Supported: {sorted(_BATCH_SERVICES)}[/red]"
        )
        raise typer.Exit(1)

    config = get_config()
    if "speechmatics_batch" in requested and not config.speechmatics_api_key:
        console.print("[red]SPEECHMATICS_API_KEY not set in environment[/red]")
        raise typer.Exit(1)
    if "elevenlabs_batch" in requested and not config.elevenlabs_api_key:
        console.print("[red]ELEVENLABS_API_KEY not set in environment[/red]")
        raise typer.Exit(1)
    if "xai_batch" in requested and not config.xai_api_key:
        console.print("[red]XAI_API_KEY not set in environment[/red]")
        raise typer.Exit(1)

    async def _run_one(
        db: Database,
        svc_name: str,
    ) -> None:
        """Run one batch service against all eligible samples."""
        if svc_name == "speechmatics_batch":
            svc_enum = ServiceName.SPEECHMATICS_BATCH
            run_and_persist = speechmatics_run_and_persist
            kwargs = dict(operating_point=operating_point, max_concurrency=concurrency)
            label = f"Speechmatics batch (operating_point={operating_point})"
            persist_model = operating_point
        elif svc_name == "elevenlabs_batch":
            svc_enum = ServiceName.ELEVENLABS_BATCH
            run_and_persist = scribe_run_and_persist
            kwargs = dict(max_concurrency=concurrency)
            label = "ElevenLabs Scribe v2 batch"
            persist_model = "scribe_v2"
        elif svc_name == "xai_batch":
            svc_enum = ServiceName.XAI_BATCH
            run_and_persist = xai_run_and_persist
            # xAI is a per-sample WebSocket; cap parallelism to avoid trips on
            # rate limits / connection ceilings.
            kwargs = dict(max_concurrency=min(concurrency, 4))
            label = "xAI realtime STT (used as batch)"
            persist_model = "grok-stt"
        else:
            raise RuntimeError(f"unreachable: {svc_name}")

        if skip_existing:
            samples_to_run = await db.get_samples_without_results(
                svc_enum, model_name=persist_model
            )
        else:
            samples_to_run = await db.get_all_samples()
        if language:
            target = language.lower().replace("-", "_")
            samples_to_run = [
                s for s in samples_to_run
                if (s.language or "").lower().replace("-", "_") == target
            ]
        if limit is not None:
            samples_to_run = samples_to_run[:limit]

        if not samples_to_run:
            console.print(f"[yellow]{label}: nothing to do[/yellow]")
            return

        console.print(
            f"\n[bold blue]{label}[/bold blue] | samples={len(samples_to_run)} "
            f"| concurrency={concurrency}"
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Transcribing...", total=len(samples_to_run))

            def cb(done: int, total: int, _sid: str) -> None:
                progress.update(task_id, completed=done)

            results = await run_and_persist(
                db, samples_to_run, progress_callback=cb, **kwargs
            )

        n_ok = sum(1 for r in results if r.error is None)
        n_err = len(results) - n_ok
        console.print(f"  [green]{n_ok} ok[/green] / [red]{n_err} errors[/red]")

    async def main() -> None:
        db = Database()
        await db.initialize()
        try:
            for svc in requested:
                await _run_one(db, svc)
        finally:
            await db.close()

    asyncio.run(main())
