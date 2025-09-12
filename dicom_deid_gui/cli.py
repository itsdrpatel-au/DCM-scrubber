from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from .controller import run_batch


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dicom-deid", description="DICOM de-identification utility"
    )
    parser.add_argument("--input", "-i", action="append", required=True, help="Input folder(s)")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument("--rows", type=int, default=200, help="Rows to mask (top)")
    parser.add_argument("--recurse", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--workers", type=int, default=0, help="Concurrency (0=CPU count)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no writes)")
    parser.add_argument(
        "--export-png",
        action="store_true",
        help="Also export PNG previews alongside outputs (in output/png)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    inputs = [Path(p).resolve() for p in args.input]
    output = Path(args.output).resolve()
    workers = args.workers if args.workers and args.workers > 0 else None

    console = Console()
    progress = Progress()
    task_id = None

    def on_progress(done: int, total: int, rate: float, eta_s: float) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("Processing", total=total)
        progress.update(task_id, completed=done)

    def on_log(level: str, msg: str) -> None:
        # Avoid PHI; messages should be generic
        if level == "error":
            console.print(f"[red]{msg}[/red]")
        elif level == "warn":
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            console.print(msg)

    with progress:
        report_path = run_batch(
            inputs=inputs,
            output_dir=output,
            rows=args.rows,
            recurse=args.recurse,
            workers=workers,
            dry_run=args.dry_run,
            on_progress=on_progress,
            on_log=on_log,
            export_pngs=args.export_png,
        )

    console.print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
