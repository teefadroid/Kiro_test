# arabic-llm-ocr

A small, from-scratch OCR tool for **Arabic** text in PDFs and images.

Instead of using a classical OCR engine (Tesseract, EasyOCR, PaddleOCR), this
tool sends page images to a **vision-capable LLM** and asks it to transcribe
the Arabic text. Modern multimodal LLMs handle Arabic script — including
diacritics (tashkeel), ligatures, and right-to-left layout — noticeably better
than most classical engines, especially on handwriting or low-quality scans.

## Features

- Input: single image (`.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`) or PDF (multi-page).
- Pluggable providers:
  - `openai` — GPT-4o / GPT-4o-mini (default)
  - `anthropic` — Claude 3.5 Sonnet / Haiku
  - `ollama` — any local vision model (e.g. `qwen2.5vl`, `llama3.2-vision`)
- Output: plain text (default), or JSON with per-page results.
- No system dependencies beyond Python (PyMuPDF renders PDFs internally — no Poppler/Tesseract).

## Install

```bash
pip install -e .
```

## Quickest start: drop-in folder workflow

1. Put your PDFs/images in `input/`.
2. Export a key for your chosen provider (see table below).
3. Run:

```bash
python agent.py
```

Each file in `input/` is OCR'd and a transcript is written to `output/` with
the same stem (`input/contract.pdf` -> `output/contract.txt`). Files that
already have output are skipped unless you pass `--overwrite`.

Common flags:

```bash
python agent.py --provider anthropic            # switch backend
python agent.py --model gpt-4o-mini             # cheaper/faster model
python agent.py --format json                   # structured output per page
python agent.py --pages 1-3 --dpi 300           # only first 3 pages, higher DPI
python agent.py --no-diacritics                 # strip tashkeel
python agent.py --input-dir scans --output-dir out   # custom folders
```

## Library / CLI usage

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...
arabic-ocr path/to/document.pdf -o transcript.txt

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
arabic-ocr scan.jpg --provider anthropic

# Local via Ollama (no API key, offline-capable)
arabic-ocr scan.png --provider ollama --model qwen2.5vl:7b

# JSON output, one entry per page
arabic-ocr book.pdf --format json -o out.json
```

## CLI

```
arabic-ocr INPUT [-o OUTPUT]
                 [--provider {openai,anthropic,ollama}]
                 [--model MODEL]
                 [--dpi DPI]
                 [--pages 1,3-5]
                 [--format {text,json}]
                 [--preserve-diacritics / --no-diacritics]
                 [--verbose]
```

- `--dpi` (default 200): rendering resolution for PDF pages. Higher = better OCR, slower.
- `--pages`: comma-separated page list, supports ranges (`1,3-5,8`). 1-indexed.
- `--preserve-diacritics` (default: on): keep tashkeel; turn off to strip them.

## Environment variables

| Variable            | Purpose                                     |
| ------------------- | ------------------------------------------- |
| `OPENAI_API_KEY`    | Required for `--provider openai`            |
| `OPENAI_BASE_URL`   | Optional override (e.g. Azure, proxies)     |
| `ANTHROPIC_API_KEY` | Required for `--provider anthropic`         |
| `OLLAMA_HOST`       | Default `http://localhost:11434`            |

## How it works

1. **Load** the input. If PDF, each requested page is rasterized with PyMuPDF
   at the chosen DPI. If an image, it's loaded and normalized with Pillow.
2. **Encode** each page as a base64 PNG.
3. **Prompt** the vision LLM with a short Arabic-OCR system prompt that:
   - asks for verbatim transcription,
   - preserves right-to-left reading order,
   - keeps or strips diacritics per user flag,
   - returns only the text — no commentary, no translation.
4. **Collect & assemble** outputs, joined with page separators or serialized as JSON.

## Limitations

- Accuracy depends on the underlying model. For best results on printed Arabic,
  GPT-4o or Claude 3.5 Sonnet are strong baselines; for handwriting, try both.
- LLM output is probabilistic — spot-check critical documents.
- No bounding-box output; this tool targets text extraction, not layout analysis.

## License

MIT
