#!/usr/bin/env python3
"""
Build assets/js/search-index.json from the site's HTML.

The site HTML may have been reformatted by prettier (tags broken across
lines). To stay robust we normalize each input file to single-line HTML
before pattern-matching.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ─── helpers ────────────────────────────────────────────────────────────────
def normalize(html: str) -> str:
    """Collapse all whitespace and tighten tags. Prettier may emit `</span >`
    or `<span class="x" >` — we squeeze internal tag whitespace too so the
    parsers see clean `<span class="x">` / `</span>`."""
    html = re.sub(r"\s+", " ", html).strip()
    # remove space before `>` in tags
    html = re.sub(r"\s+>", ">", html)
    # remove space between attribute and `>`
    return html


_ENTS = {
    "&nbsp;": " ",
    "&middot;": "·",
    "&mdash;": "—",
    "&ndash;": "–",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&#39;": "'",
    "&eacute;": "é",
    "&egrave;": "è",
    "&aacute;": "á",
    "&iacute;": "í",
    "&oacute;": "ó",
    "&uacute;": "ú",
    "&auml;": "ä",
    "&ouml;": "ö",
    "&uuml;": "ü",
    "&ccedil;": "ç",
    "&Eacute;": "É",
}


def strip_tags(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    for k, v in _ENTS.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def find_all(pattern: str, text: str, flags: int = 0):
    return re.findall(pattern, text, flags)


# ─── parsers ────────────────────────────────────────────────────────────────
def parse_news(path: Path) -> list[dict]:
    html = normalize(path.read_text(encoding="utf-8"))
    entries = []
    # match each <li> ... </li> inside <ul class="news">
    ul_m = re.search(r'<ul class="news">(.*?)</ul>', html)
    if not ul_m:
        return entries
    body = ul_m.group(1)
    for m in re.finditer(r"<li>(.*?)</li>", body):
        chunk = m.group(1)
        date_m = re.search(r'<span class="date">(.*?)</span\s*>', chunk)
        item_m = re.search(r'<span class="item">(.*?)</span\s*>(?:\s*$|\s*<)', chunk)
        if not (date_m and item_m):
            continue
        date = strip_tags(date_m.group(1))
        text = strip_tags(item_m.group(1))
        entries.append(
            {
                "title": (text[:90] + "…") if len(text) > 90 else text,
                "snippet": text,
                "url": "subpages/news.html",
                "category": "news",
                "date": date,
            }
        )
    return entries


def parse_publications(path: Path) -> list[dict]:
    html = normalize(path.read_text(encoding="utf-8"))
    entries = []
    # each pub is <div class="pub"> wrapping <div class="venue"> and <div> with details
    # we just want title + authors + venue/year; structure:
    #   <div class="pub"><div class="venue">VENUE<span class="sub">SUB</span></div>
    #     <div><span class="title">TITLE</span><span class="authors">…</span>
    #          <span class="periodical">…</span>…
    # depth-1 closing — find non-nested via spans we care about
    for m in re.finditer(r'<div class="pub">(.*?)<div class="bib-panel">', html):
        block = m.group(1)
        title_m = re.search(r'<span class="title">(.*?)</span\s*>', block)
        authors_m = re.search(r'<span class="authors">(.*?)</span\s*>', block)
        venue_m = re.search(r'<span class="periodical">(.*?)</span\s*>', block)
        venue_top_m = re.search(r'<div class="venue">(.*?)</div\s*>', block)
        if not title_m:
            continue
        title = strip_tags(title_m.group(1))
        authors = strip_tags(authors_m.group(1)) if authors_m else ""
        venue = strip_tags(venue_m.group(1)) if venue_m else ""
        venue_top = strip_tags(venue_top_m.group(1)) if venue_top_m else ""
        snippet = " · ".join(filter(None, [authors, venue, venue_top]))
        year_m = re.search(r"\b(20\d{2})\b", venue + " " + venue_top)
        entries.append(
            {
                "title": title,
                "snippet": snippet,
                "url": "subpages/publications.html",
                "category": "publication",
                "date": year_m.group(1) if year_m else "",
            }
        )
    return entries


def parse_writings(path: Path) -> list[dict]:
    html = normalize(path.read_text(encoding="utf-8"))
    entries = []
    ul_m = re.search(r'<ul class="posts">(.*?)</ul>', html)
    if not ul_m:
        return entries
    body = ul_m.group(1)
    for m in re.finditer(r"<li>(.*?)</li>", body):
        chunk = m.group(1)
        date_m = re.search(r'<span class="date">(.*?)</span\s*>', chunk)
        link_m = re.search(
            r'<span class="post-title">\s*<a href="([^"]+)"[^>]*>(.*?)</a\s*>\s*</span\s*>',
            chunk,
        )
        desc_m = re.search(r'<span class="post-desc">(.*?)</span\s*>', chunk)
        if not (date_m and link_m):
            continue
        date = strip_tags(date_m.group(1))
        href = link_m.group(1)
        if href.startswith("../"):
            href = href[3:]
        title = strip_tags(link_m.group(2))
        desc = strip_tags(desc_m.group(1)) if desc_m else ""
        entries.append(
            {
                "title": title,
                "snippet": desc,
                "url": href,
                "category": "writing",
                "date": date,
            }
        )
    return entries


def parse_colophon(path: Path) -> list[dict]:
    if not path.exists():
        return []
    html = normalize(path.read_text(encoding="utf-8"))
    # grab the lede plus first paragraph for a representative snippet
    lede_m = re.search(r'<div class="lede">(.*?)</div\s*>', html)
    para_m = re.search(r'<p>(.*?)</p\s*>', html)
    lede = strip_tags(lede_m.group(1)) if lede_m else ""
    para = strip_tags(para_m.group(1)) if para_m else ""
    snippet = (lede + " " + para).strip()
    return [
        {
            "title": "Colophon",
            "snippet": snippet[:320],
            "url": "subpages/colophon.html",
            "category": "page",
        }
    ]


def parse_about(path: Path) -> list[dict]:
    html = normalize(path.read_text(encoding="utf-8"))
    entries = []
    bio_m = re.search(r'<div class="bio">(.*?)</div>\s*<figure', html)
    if bio_m:
        bio_text = strip_tags(bio_m.group(1))
        entries.append(
            {
                "title": "About — Zander W. Blasingame",
                "snippet": bio_text[:320],
                "url": "index.html#about",
                "category": "page",
            }
        )
    # repos list: <li><a href="…">name<span class="desc">desc</span></a></li>
    for href, name, desc in re.findall(
        r'<a href="([^"]+)"[^>]*>\s*([^<]+?)\s*<span class="desc">(.*?)</span\s*>\s*</a\s*>',
        html,
    ):
        if "github.com" not in href:
            continue
        entries.append(
            {
                "title": strip_tags(name).strip(),
                "snippet": strip_tags(desc).strip(),
                "url": href,
                "category": "code",
            }
        )
    return entries


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    out: list[dict] = []
    out += parse_about(ROOT / "index.html")
    out += parse_writings(ROOT / "subpages" / "writings.html")
    out += parse_publications(ROOT / "subpages" / "publications.html")
    out += parse_news(ROOT / "subpages" / "news.html")
    out += parse_colophon(ROOT / "subpages" / "colophon.html")

    target_json = ROOT / "assets" / "js" / "search-index.json"
    target_js = ROOT / "assets" / "js" / "search-index.js"
    payload = json.dumps(out, ensure_ascii=False)
    target_json.write_text(payload, encoding="utf-8")
    # JS shim so the site works on file:// too (no fetch needed)
    target_js.write_text(
        "/* auto-generated by build-search-index.py — do not edit */\n"
        "window.__searchIndex = " + payload + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(out)} entries → {target_json.relative_to(ROOT)} + {target_js.relative_to(ROOT)}")
    c = Counter(e["category"] for e in out)
    for cat, n in c.most_common():
        print(f"  {cat:12s} {n}")


if __name__ == "__main__":
    main()
