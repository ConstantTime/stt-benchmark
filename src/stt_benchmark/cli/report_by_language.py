"""Per-language WER report.

The default `report` command aggregates across all samples regardless of
language. For the multilingual benchmark we want a 6-row × N-service table
broken down by language, which is what gets pasted into Slack.
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from stt_benchmark.config import get_config
from stt_benchmark.storage.database import Database

app = typer.Typer()
console = Console()


# Pretty names for the 6 target languages (plus English fallback).
_LANG_LABELS = {
    "nb_no": "Norwegian",
    "no": "Norwegian",
    "nb": "Norwegian",
    "da_dk": "Danish",
    "da": "Danish",
    "de_de": "German",
    "de": "German",
    "fr_fr": "French",
    "fr": "French",
    "it_it": "Italian",
    "it": "Italian",
    "es_es": "Spanish",
    "es_419": "Spanish (LatAm)",
    "es": "Spanish",
    "eng": "English",
    "en": "English",
    "en_us": "English",
}


def _label(code: str | None) -> str:
    if not code:
        return "(unknown)"
    return _LANG_LABELS.get(code.lower(), code)


@app.callback(invoke_without_command=True)
def report_by_language(
    wer_label: str | None = typer.Option(
        None,
        "--wer-label",
        help="Restrict to wer_metrics rows with this model_name (the --wer-label "
        "you used when running the wer command). If unset, all WER rows are included.",
    ),
    csv_out: str | None = typer.Option(
        None,
        "--csv",
        help="Optional path to write the per-language table as CSV.",
    ),
):
    """Print a per-language × per-service WER table.

    Joins `samples.language` with `wer_metrics` so each row is one
    (language, service, wer-label) triple. Lists Mean WER, Pooled WER, and
    sample count.
    """

    async def main() -> None:
        config = get_config()
        db = Database()
        await db.initialize()

        # Pull every WER metric joined with its sample's language.
        # We do the aggregation in Python rather than SQL to keep this readable
        # and to make pooled-WER computation explicit.
        if wer_label:
            cur = await db._conn.execute(  # noqa: SLF001 — direct query is fine here
                """
                SELECT s.language, w.service_name, w.model_name,
                       w.wer, w.substitutions, w.deletions, w.insertions,
                       w.reference_words
                FROM wer_metrics w
                JOIN samples s ON s.sample_id = w.sample_id
                WHERE w.model_name = ?
                """,
                (wer_label,),
            )
        else:
            cur = await db._conn.execute(  # noqa: SLF001
                """
                SELECT s.language, w.service_name, w.model_name,
                       w.wer, w.substitutions, w.deletions, w.insertions,
                       w.reference_words
                FROM wer_metrics w
                JOIN samples s ON s.sample_id = w.sample_id
                """
            )
        rows = await cur.fetchall()
        await db.close()

        if not rows:
            console.print("[yellow]No WER metrics found for the given filter.[/yellow]")
            return

        # Aggregate: {(language, service, model): {n, sum_wer, sub, del, ins, ref}}
        buckets: dict[tuple[str, str, str | None], dict] = {}
        for r in rows:
            key = (r["language"] or "", r["service_name"], r["model_name"])
            b = buckets.setdefault(
                key,
                dict(n=0, sum_wer=0.0, sub=0, dele=0, ins=0, ref=0),
            )
            b["n"] += 1
            b["sum_wer"] += r["wer"]
            b["sub"] += r["substitutions"]
            b["dele"] += r["deletions"]
            b["ins"] += r["insertions"]
            b["ref"] += r["reference_words"]

        # Order: language alpha, then mean WER asc within each language.
        records = []
        for (lang, service, model), b in buckets.items():
            mean_wer = b["sum_wer"] / b["n"] if b["n"] else 0.0
            pooled = (b["sub"] + b["dele"] + b["ins"]) / b["ref"] if b["ref"] else 0.0
            records.append(
                dict(
                    language=lang,
                    label=_label(lang),
                    service=service,
                    model=model or "",
                    n=b["n"],
                    mean_wer=mean_wer,
                    pooled_wer=pooled,
                )
            )
        records.sort(key=lambda r: (r["label"], r["mean_wer"]))

        # Render Rich table
        table = Table(title="Per-language Semantic WER")
        table.add_column("Language", style="bold")
        table.add_column("Service", style="cyan")
        table.add_column("WER label", style="dim")
        table.add_column("N", justify="right")
        table.add_column("Mean WER", justify="right")
        table.add_column("Pooled WER", justify="right")

        current_lang = None
        for r in records:
            lang_cell = r["label"] if r["label"] != current_lang else ""
            current_lang = r["label"]
            table.add_row(
                lang_cell,
                r["service"],
                r["model"],
                str(r["n"]),
                f"{r['mean_wer']:.2%}",
                f"{r['pooled_wer']:.2%}",
            )

        console.print(table)

        # Optional CSV
        if csv_out:
            import csv

            out = Path(csv_out)
            with out.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["language_code", "language", "service", "wer_label",
                            "n_samples", "mean_wer_pct", "pooled_wer_pct"])
                for r in records:
                    w.writerow([
                        r["language"], r["label"], r["service"], r["model"],
                        r["n"], f"{r['mean_wer']*100:.2f}", f"{r['pooled_wer']*100:.2f}",
                    ])
            console.print(f"\nCSV written: [green]{out}[/green]")

    asyncio.run(main())
