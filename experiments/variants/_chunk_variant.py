"""Record size: the main solution with a different way of splitting pages into records.

    text250 / text500 / text1000   paragraph text cut into pieces of about 250, 500 (shipped first)
                                   or 1000 characters (text1500, 1500 characters, is adopted)
    overlap                 500-character pieces that repeat the previous piece's last sentence
    section                 one record per heading section, however long (tables still one record per row)
    table                   each table is one record (all rows together) instead of one record per row
    page                    one record per page (text and tables together): the no-chunking baseline

Records longer than the shipped 900-character display limit are shown to the reader in full (up to 6000
characters), so a bigger record is not cut off before the model sees it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _search_variant as sv  # noqa: E402  (puts harness/ and solution/ on sys.path)
import kb  # noqa: E402
import reader  # noqa: E402
from kb import _SENT, _cells, clean_text, parse_effective  # noqa: E402

CONFIGS = {
    "text500": dict(text_chars=500, overlap=False, table="row"),     # shipped first
    "text250": dict(text_chars=250, overlap=False, table="row"),
    "text1000": dict(text_chars=1000, overlap=False, table="row"),
    "text1500": dict(text_chars=1500, overlap=False, table="row"),
    "text2000": dict(text_chars=2000, overlap=False, table="row"),
    "overlap": dict(text_chars=500, overlap=True, table="row"),
    "overlap1500": dict(text_chars=1500, overlap=True, table="row"),
    "section": dict(text_chars=10**9, overlap=False, table="row"),
    "table": dict(text_chars=500, overlap=False, table="table"),
    "page": dict(text_chars=10**9, overlap=False, table="page"),
}
SHOW_CHARS = 6000


def make_chunk_page(text_chars: int, overlap: bool, table: str):
    def chunk_page(url: str, title: str, md: str):
        md = clean_text(md)
        lines = md.split("\n")
        heads: list[str] = []
        i, n = 0, len(lines)
        para: list[str] = []
        out: list = []
        whole: list[str] = []          # for table == "page": everything on the page, in order

        def _emit(text, ctx):
            text = text.strip()
            if len(text) >= 40 or (len(text) >= 15 and re.search(r"[$%]\s?\d|\d\s?%", text)):
                if table == "page":
                    whole.append(text)
                else:
                    out.append(("text", text, ctx, "", {}, parse_effective(text)))

        def flush_para():
            if not para:
                return
            units = []
            for p in para:
                p = p.strip()
                if not p:
                    continue
                p = re.sub(r"^[-*]\s+", "", p)
                units.extend(x.strip() for x in _SENT.split(p) if x.strip())
            para.clear()
            ctx = " > ".join(heads[-2:])
            cur, last = "", ""
            for u in units:
                if cur and len(cur) + len(u) > text_chars:
                    _emit(cur, ctx)
                    cur = (last + " ") if overlap and last else ""
                cur += u + " "
                last = u
            _emit(cur, ctx)

        while i < n:
            ln = lines[i].rstrip()
            if ln.startswith("#"):
                flush_para()
                level = len(ln) - len(ln.lstrip("#"))
                heads = heads[: level - 1] + [ln.lstrip("# ").strip()]
                if table == "page":
                    whole.append(ln.lstrip("# ").strip())
                i += 1
                continue
            if ln.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
                flush_para()
                header = _cells(ln)
                j = i + 2
                rows = []
                while j < n and lines[j].lstrip().startswith("|"):
                    rows.append(_cells(lines[j]))
                    j += 1
                tail = " ".join(lines[j: j + 6])
                eff = parse_effective(tail) or parse_effective(" ".join(lines[max(0, i - 4): i]))
                ctx = " > ".join(heads[-2:])
                labels = []
                for r in rows:
                    vals = {header[k] if k < len(header) else f"c{k}": r[k] for k in range(len(r))}
                    label = " | ".join(f"{h}: {v}" for h, v in vals.items() if v)
                    labels.append(label)
                    if table == "row":
                        key_cells = [r[k] for k in range(len(r)) if not re.search(r"\d+\.\d+%|\$[\d,]+\.\d\d$", r[k])]
                        out.append(("row", label, ctx + " " + " ".join(header), " ".join(key_cells).lower(), vals, eff))
                if table == "table" and labels:
                    out.append(("text", "\n".join(labels), ctx + " " + " ".join(header), "", {}, eff))
                elif table == "page":
                    whole.extend(labels)
                i = j
                continue
            if ln.strip():
                para.append(ln)
            i += 1
        flush_para()
        if table == "page" and whole:
            text = "\n".join(whole)
            out.append(("text", text, title, "", {}, parse_effective(text)))
        return out
    return chunk_page


def render(records):
    out = []
    for i, c in enumerate(records, 1):
        head = f"[{i}] page: {c.url} | {reader._trust(c)}"
        if c.effective:
            head += f" | effective {c.effective}"
        section = " > ".join(x for x in (c.title, c.context) if x)
        if section:
            head += f" | section: {section[:160]}"
        out.append(f"{head}\n{c.text[:SHOW_CHARS]}")
    return "\n\n".join(out)


def install(name: str):
    kb.chunk_page = make_chunk_page(**CONFIGS[name])
    reader.Reader.render = staticmethod(render)


def build_with(name: str):
    install(name)
    return sv._load_main().build()
