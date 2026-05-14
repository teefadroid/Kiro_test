"""One-shot OCR runner.

Reads every supported file in ``./input`` and writes a transcript per file into
``./output`` using the ``arabic_ocr`` package. Designed to be the single entry
point a non-developer runs:

    python agent.py

Knobs are exposed via flags and environment variables, but the defaults are
chosen so the zero-argument invocation "just works" once an API key is set.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Make `src/` importable when the script is run from the repo root without
# needing `pip install -e .` first. If the package is already installed, this
# is a harmless no-op because the installed version wins on sys.path.
REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arabic_ocr.loader import IMAGE_EXTS, PDF_EXTS  # noqa: E402
from arabic_ocr.ocr import OCRResult, PageResult, ocr_file  # noqa: E402
from arabic_ocr.providers import available_providers  # noqa: E402
from arabic_ocr.providers.base import ProviderError  # noqa: E402

SUPPORTED_EXTS = IMAGE_EXTS | PDF_EXTS
DEFAULT_INPUT = REPO_ROOT / "input"
DEFAULT_OUTPUT = REPO_ROOT / "output"

log = logging.getLogger("agent")


def _discover_inputs(input_dir: Path) -> list[Path]:
    """Return sorted supported files directly inside ``input_dir`` (non-recursive)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    files = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return files


_FMT_SUFFIX = {"text": ".txt", "json": ".json", "md": ".md"}


def _output_path(src: Path, output_dir: Path, fmt: str) -> Path:
    return output_dir / (src.stem + _FMT_SUFFIX[fmt])


def _render(result: OCRResult, fmt: str) -> str:
    if fmt == "json":
        return result.as_json()
    if fmt == "md":
        return result.as_markdown()
    return result.as_text(include_separators=len(result.pages) > 1)


def _progress_factory(file_index: int, file_total: int, name: str):
    """Build a per-file progress callback that prefixes the file being processed."""
    def progress(current: int, total: int, page: PageResult) -> None:
        if page.skipped:
            status = "skipped (no Arabic)"
        elif page.error:
            status = f"ERROR: {page.error}"
        else:
            status = "ok"
        print(
            f"  [{file_index}/{file_total}] {name}: "
            f"page {page.page_number} ({current}/{total}) "
            f"{page.latency_s}s -- {status}",
            file=sys.stderr,
        )
    return progress


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent.py",
        description="Batch-OCR every file in ./input and write results to ./output.",
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                   help=f"Folder to scan for PDFs/images (default: {DEFAULT_INPUT.name}/).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Folder to write transcripts into (default: {DEFAULT_OUTPUT.name}/).")
    p.add_argument("--provider", choices=available_providers(),
                   default=os.environ.get("ARABIC_OCR_PROVIDER", "ollama"),
                   help="Vision LLM backend (default: ollama, or $ARABIC_OCR_PROVIDER).")
    p.add_argument("--model", default=os.environ.get("ARABIC_OCR_MODEL"),
                   help="Model override (e.g. qwen2.5vl:7b, gpt-4o-mini, claude-3-5-haiku-latest).")
    p.add_argument("--dpi", type=int, default=200,
                   help="PDF rendering DPI (default: 200). Lower = faster on CPU.")
    p.add_argument("--timeout", type=float, default=None,
                   help="Per-page HTTP timeout in seconds. Default: provider-specific "
                        "(900s for ollama, 120s for hosted providers).")
    p.add_argument("--pages", default=None,
                   help='Page selection like "1,3-5" (applies to every PDF). Default: all pages.')
    p.add_argument("--format", choices=("text", "json", "md"), default="text",
                   help="Output format per file: text (.txt), json (.json), "
                        "or md (.md, Markdown with RTL-aware page headings). "
                        "Default: text.")

    diacritics = p.add_mutually_exclusive_group()
    diacritics.add_argument("--preserve-diacritics", dest="preserve_diacritics",
                            action="store_true", default=True,
                            help="Keep Arabic tashkeel in the output (default).")
    diacritics.add_argument("--no-diacritics", dest="preserve_diacritics",
                            action="store_false",
                            help="Strip Arabic tashkeel from the output.")

    p.add_argument("--overwrite", action="store_true",
                   help="Re-OCR files even if an output already exists.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = _discover_inputs(input_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not files:
        print(
            f"No supported files found in {input_dir}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTS))}",
            file=sys.stderr,
        )
        return 0

    print(
        f"Found {len(files)} file(s) in {input_dir} "
        f"-> {output_dir} (provider={args.provider}, model={args.model or 'default'})",
        file=sys.stderr,
    )

    total = len(files)
    skipped = 0
    ok = 0
    partial = 0
    failed = 0
    t_start = time.monotonic()

    for idx, src in enumerate(files, start=1):
        dest = _output_path(src, output_dir, args.format)
        if dest.exists() and not args.overwrite:
            print(f"  [{idx}/{total}] {src.name}: skipped (exists -- use --overwrite to redo)",
                  file=sys.stderr)
            skipped += 1
            continue

        try:
            result = ocr_file(
                src,
                provider=args.provider,
                model=args.model,
                dpi=args.dpi,
                page_spec=args.pages,
                preserve_diacritics=args.preserve_diacritics,
                timeout=args.timeout,
                progress=_progress_factory(idx, total, src.name),
            )
        except ProviderError as e:
            print(f"  [{idx}/{total}] {src.name}: provider error -- {e}", file=sys.stderr)
            failed += 1
            continue
        except (FileNotFoundError, ValueError) as e:
            print(f"  [{idx}/{total}] {src.name}: error -- {e}", file=sys.stderr)
            failed += 1
            continue

        dest.write_text(_render(result, args.format), encoding="utf-8")

        page_errs = [p for p in result.pages if p.error]
        if page_errs:
            partial += 1
            print(
                f"  [{idx}/{total}] {src.name}: wrote {dest.name} "
                f"({len(result.pages) - len(page_errs)}/{len(result.pages)} pages ok)",
                file=sys.stderr,
            )
        else:
            ok += 1
            print(f"  [{idx}/{total}] {src.name}: wrote {dest.name}", file=sys.stderr)

    elapsed = time.monotonic() - t_start
    print(
        f"Done in {elapsed:.1f}s -- ok={ok} partial={partial} "
        f"failed={failed} skipped={skipped} total={total}",
        file=sys.stderr,
    )

    # Exit codes: 0 clean, 4 at least one page-level error, 3 at least one file-level failure.
    if failed:
        return 3
    if partial:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
