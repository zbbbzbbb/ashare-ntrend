"""
Tiny dependency-free Markdown -> standalone mobile-friendly HTML converter.
Handles exactly the constructs used in this project's reports: #/##/### headings,
pipe tables, > blockquotes, ordered/unordered lists, --- rules, **bold**, `code`.

Tables are wrapped in a horizontal-scroll container with a sticky first column,
and the page sets a mobile viewport — so it reads cleanly on a phone browser.

Usage: python3 -m research.md2html <file1.md> [file2.md ...]
       (writes <file>.html next to each input)
"""
import re, sys, html
from pathlib import Path

CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;padding:16px;max-width:900px;margin:0 auto;
 font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
 color:#1a1a1a;background:#fff;-webkit-text-size-adjust:100%}
@media(prefers-color-scheme:dark){body{color:#e6e6e6;background:#161616}
 th{background:#262626!important}tr:nth-child(even) td{background:#1e1e1e!important}
 td:first-child,th:first-child{background:#202020!important}blockquote{background:#1e1e1e!important;border-color:#3a6ea5!important}
 code{background:#2a2a2a!important}hr{border-color:#333!important}a{color:#6ca8ff}}
h1{font-size:1.5em;line-height:1.3;margin:.6em 0 .4em}
h2{font-size:1.25em;margin:1.4em 0 .4em;padding-top:.4em;border-top:2px solid #e5e5e5}
h3{font-size:1.08em;margin:1em 0 .3em;color:#0a58ca}
@media(prefers-color-scheme:dark){h3{color:#6ca8ff}h2{border-color:#333}}
p{margin:.5em 0}
blockquote{margin:.8em 0;padding:.6em .9em;background:#f5f8ff;border-left:4px solid #3a6ea5;border-radius:4px;font-size:.95em}
ul,ol{margin:.5em 0;padding-left:1.4em}li{margin:.3em 0}
code{background:#f0f0f0;padding:1px 5px;border-radius:4px;font-size:.88em;font-family:ui-monospace,Menlo,Consolas,monospace}
hr{border:none;border-top:1px solid #ddd;margin:1.4em 0}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:.8em 0;border:1px solid #e0e0e0;border-radius:6px}
table{border-collapse:collapse;font-size:13px;min-width:100%}
th,td{border:1px solid #e3e3e3;padding:7px 9px;white-space:nowrap;text-align:left}
th{background:#f2f2f2;position:sticky;top:0}
tr:nth-child(even) td{background:#fafafa}
td:first-child,th:first-child{position:sticky;left:0;background:#f7f7f7;font-weight:600}
strong{font-weight:700}
"""


def _inline(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def _cells(line):
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts


def convert(md, title="report"):
    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        # table: a | row followed by a |---| separator
        if ln.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            head = _cells(ln); i += 2
            body = []
            while i < n and lines[i].lstrip().startswith("|"):
                body.append(_cells(lines[i])); i += 1
            t = ['<div class="tw"><table><thead><tr>']
            t += [f"<th>{_inline(c)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for row in body:
                t.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t)); continue
        if re.match(r"^\s*-{3,}\s*$", ln):
            out.append("<hr>"); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1)); out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>"); i += 1; continue
        if ln.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(_inline(re.sub(r"^\s*>\s?", "", lines[i]))); i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>"); continue
        if re.match(r"^\s*\d+\.\s+", ln):
            buf = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                buf.append(f"<li>{_inline(re.sub(r'^\s*\d+\.\s+', '', lines[i]))}</li>"); i += 1
            out.append("<ol>" + "".join(buf) + "</ol>"); continue
        if re.match(r"^\s*[-*]\s+", ln):
            buf = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                buf.append(f"<li>{_inline(re.sub(r'^\s*[-*]\s+', '', lines[i]))}</li>"); i += 1
            out.append("<ul>" + "".join(buf) + "</ul>"); continue
        if ln.strip() == "":
            i += 1; continue
        out.append(f"<p>{_inline(ln)}</p>"); i += 1

    return (f"<!doctype html><html lang=zh><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{''.join(out)}</body></html>")


def main(argv):
    if not argv:
        print("usage: python3 -m research.md2html <file.md> ..."); return
    for f in argv:
        p = Path(f)
        out = p.with_suffix(".html")
        out.write_text(convert(p.read_text(encoding="utf-8"), title=p.stem), encoding="utf-8")
        print(f"{p.name} -> {out.name}  ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1:])
