"""Load PDFs and images into a uniform list of page images.

The rest of the pipeline only cares about a sequence of ``PageImage`` objects,
so this module hides the difference between raster images and multi-page PDFs.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_EXTS = {".pdf"}

# Cap any single page edge to this many pixels before sending to an LLM.
# Vision models reject very large images and the extra pixels rarely help OCR.
MAX_EDGE_PX = 2200


@dataclass
class PageImage:
    """A single page rendered as a PIL image, with its 1-indexed page number."""

    page_number: int
    image: Image.Image

    def to_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def to_base64_png(self) -> str:
        return base64.b64encode(self.to_png_bytes()).decode("ascii")


def parse_page_spec(spec: str | None, total_pages: int) -> list[int]:
    """Parse ``"1,3-5,8"`` into ``[1, 3, 4, 5, 8]``. 1-indexed, clamped to ``total_pages``."""
    if not spec:
        return list(range(1, total_pages + 1))

    pages: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(chunk))

    selected = sorted(p for p in pages if 1 <= p <= total_pages)
    if not selected:
        raise ValueError(
            f"Page selection {spec!r} produced no valid pages "
            f"(document has {total_pages} page(s))."
        )
    return selected


def _downscale_if_needed(img: Image.Image, max_edge: int = MAX_EDGE_PX) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _normalize(img: Image.Image) -> Image.Image:
    """Flatten transparency and force RGB so PNG encoding is consistent."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _load_image(path: Path) -> list[PageImage]:
    with Image.open(path) as im:
        im.load()
        img = _normalize(im)
    img = _downscale_if_needed(img)
    return [PageImage(page_number=1, image=img)]


def _load_pdf(path: Path, dpi: int, pages: Sequence[int] | None) -> list[PageImage]:
    # Imported lazily so that image-only users don't pay the PyMuPDF import cost.
    import fitz  # PyMuPDF

    out: list[PageImage] = []
    with fitz.open(path) as doc:
        total = doc.page_count
        page_numbers = list(pages) if pages else list(range(1, total + 1))
        # Render at the requested DPI. 72 is PDF's default user-space DPI.
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for pno in page_numbers:
            if pno < 1 or pno > total:
                raise ValueError(f"Page {pno} out of range (1..{total}).")
            page = doc.load_page(pno - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = _downscale_if_needed(img)
            out.append(PageImage(page_number=pno, image=img))
    return out


def load_pages(
    path: str | Path,
    *,
    dpi: int = 200,
    page_spec: str | None = None,
) -> list[PageImage]:
    """Load a PDF or image file into a list of ``PageImage``.

    Parameters
    ----------
    path:
        Path to a ``.pdf`` or supported image file.
    dpi:
        Rendering DPI for PDF pages. Ignored for images.
    page_spec:
        Optional page selection like ``"1,3-5"``. Ignored for images.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        return _load_image(p)
    if ext in PDF_EXTS:
        # Pre-resolve page spec against the document's real page count.
        import fitz

        with fitz.open(p) as doc:
            selected = parse_page_spec(page_spec, doc.page_count)
        return _load_pdf(p, dpi=dpi, pages=selected)

    raise ValueError(
        f"Unsupported file type: {ext!r}. "
        f"Expected PDF or one of {sorted(IMAGE_EXTS)}."
    )


def iter_pages(pages: Iterable[PageImage]) -> Iterator[PageImage]:
    """Passthrough iterator — a placeholder for future streaming/batching."""
    yield from pages
