#!/usr/bin/env python3
"""Find new state-of-the-art papers for LLM lie detection and weight-space reading.

Uses only free sources that need no account or API key:
  - arXiv API (keyword searches, newest first)
  - OpenAlex (papers that cite the seed papers, plus keyword search)
  - OpenReview (conference submissions, e.g. ICLR 2027)

It removes papers already listed in research-findings/*.md and literature/*.md, extracts reported scores
(AUROC/AUC/accuracy) from abstracts, and flags papers that may be at par with or
better than the two yardsticks:
  - FIBS (arXiv 2610.12445): probes at 0.987 AUROC on SHADE-Arena, 0.980 on BashArena
  - Weight-Space Programming (MIT blog, Nov 2026): LLM reads/edits raw weights

Output: research-findings/script-output/sota-candidates.md (a ranked list to read by hand).

Usage:
  python3 research-findings/tools/find_sota.py              # papers from the last 60 days
  python3 research-findings/tools/find_sota.py --days 365   # wider window
  python3 research-findings/tools/find_sota.py --aspect weights --out research-findings/script-output/sota-candidates-weights.md
"""
import argparse
import datetime as dt
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "llm-lie-detector-literature-script"}
FINDINGS = pathlib.Path(__file__).resolve().parent.parent  # research-findings/
REPO = FINDINGS.parent
OUT = FINDINGS / "script-output" / "sota-candidates.md"

SEEDS = {  # arXiv ID -> (name, aspect); citing papers of these are collected
    "2610.12445": ("FIBS probes", "detection"),
    "2502.03407": ("Apollo linear deception probes", "detection"),
    "2606.12618": ("Did you lie?", "detection"),
    "2511.16035": ("Liars' Bench", "detection"),
    "2512.15674": ("Activation Oracles", "detection"),
    "2610.07334": ("Weight Oracles", "weights"),
    "2611.08070": ("WSP critique", "weights"),
    "2611.05341": ("WSP clinical audit", "weights"),
    "2604.19117": ("Shared sycophancy-lying circuit", "detection"),
}

ARXIV_QUERIES = [  # (query, aspect)
    ('abs:"deception" AND abs:"probe"', "detection"),
    ('abs:"lie detection" AND (abs:"language model" OR abs:"LLM")', "detection"),
    ('abs:"lie detector" AND (abs:"language model" OR abs:"LLM")', "detection"),
    ('abs:"deceptive" AND abs:"activations"', "detection"),
    ('abs:"honesty" AND abs:"probe"', "detection"),
    ('abs:"sandbagging" AND (abs:"probe" OR abs:"detect")', "detection"),
    ('abs:"hidden objective" OR abs:"hidden goal"', "detection"),
    ('abs:"alignment faking" AND abs:"detect"', "detection"),
    ('abs:"activation oracle"', "detection"),
    ('abs:"weight-space programming" OR abs:"weight space programming"', "weights"),
    ('abs:"raw weights" AND abs:"language model"', "weights"),
    ('abs:"weight diff" AND abs:"interpret"', "weights"),
    ('abs:"backdoor" AND abs:"weight space"', "weights"),
    ('abs:"parameter decomposition"', "weights"),
    ('abs:"model diffing"', "weights"),
]

OPENREVIEW_QUERIES = [  # (query, aspect)
    ("deception probe", "detection"), ("lie detection language model", "detection"),
    ("honesty adapter", "detection"), ("deception detection activations", "detection"),
    ("sandbagging detection", "detection"), ("hidden behaviors model organisms", "detection"),
    ("weight space programming", "weights"), ("reading neural network weights", "weights"),
    ("model diffing weights", "weights"),
]

# A paper must mention at least one of these to count as in-field.
TOPIC = re.compile(
    r"decept|deceiv|\blie\b|\blies\b|lying|liar|dishonest|honesty|truthful|sandbag|"
    r"hidden (goal|objective|behavio)|alignment faking|scheming|sycophan|"
    r"weight.space|raw weights|weight diff|activation oracle|model organism", re.I)
# ...and at least one of these, so behavioral-only work is ranked lower.
WHITEBOX = re.compile(
    r"probe|activation|hidden state|residual|internal|representation|weights|"
    r"circuit|attention head|steering|sparse autoencoder|transcoder|interpretab", re.I)
OFFTOPIC = re.compile(
    r"honeypot|phishing|scam|fraud|deepfake|speech|video|image restoration|"
    r"wireless|radar|cyber deception|human deception|polygraph|wi-fi|fusion|plasma|"
    r"segment anything|SAM2|music|DAO|penetration.testing|GUI|web agents? under deceptive", re.I)
BEATS = re.compile(r"outperform|surpass|state.of.the.art|beat|exceed|better than", re.I)
SCORE = re.compile(
    r"(AUROC|AUC|accuracy|recall|F1)[^0-9]{0,25}(0?\.\d{2,3}|\d{2}(?:\.\d)?\s?%)", re.I)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


def norm(title):
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())[:60]


def arxiv(days, aspect):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    since = (dt.date.today() - dt.timedelta(days=days)).strftime("%Y%m%d")
    out = []
    for q, a in ARXIV_QUERIES:
        if aspect != "all" and a != aspect:
            continue
        query = f"({q}) AND submittedDate:[{since}0000 TO 209912312359]"
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": query, "max_results": 100,
             "sortBy": "submittedDate", "sortOrder": "descending"})
        try:
            root = ET.fromstring(get(url, 90))
        except Exception as e:
            print("arXiv error:", q, e)
            continue
        for e in root.findall("a:entry", ns):
            aid = e.find("a:id", ns).text.split("/abs/")[-1].split("v")[0]
            out.append({
                "title": " ".join(e.find("a:title", ns).text.split()),
                "abstract": " ".join(e.find("a:summary", ns).text.split()),
                "date": e.find("a:published", ns).text[:10],
                "id": f"arXiv {aid}", "url": f"https://arxiv.org/abs/{aid}",
                "source": "arXiv search",
            })
        time.sleep(3)  # arXiv asks for at most one request every 3 seconds
    return out


def openalex_abstract(inv):
    if not inv:
        return ""
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return " ".join(w for _, w in pos)


def openalex(days, aspect):
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    out = []
    for aid, (name, a) in SEEDS.items():
        if aspect != "all" and a != aspect:
            continue
        try:
            w = json.loads(get(f"https://api.openalex.org/works/doi:10.48550/arXiv.{aid}"))
        except Exception:
            continue  # seed not indexed yet
        wid = w["id"].rsplit("/", 1)[-1]
        url = ("https://api.openalex.org/works?" + urllib.parse.urlencode({
            "filter": f"cites:{wid},from_publication_date:{since}", "per-page": 200,
            "select": "title,doi,publication_date,abstract_inverted_index"}))
        try:
            res = json.loads(get(url))["results"]
        except Exception as e:
            print("OpenAlex error:", name, e)
            continue
        for x in res:
            doi = (x.get("doi") or "").replace("https://doi.org/", "")
            out.append({
                "title": x.get("title") or "", "date": x.get("publication_date") or "",
                "abstract": openalex_abstract(x.get("abstract_inverted_index")),
                "id": doi or "-", "url": f"https://doi.org/{doi}" if doi else "",
                "source": f"cites {name}",
            })
        time.sleep(1)
    return out


def openreview(days, aspect):
    cutoff = (time.time() - days * 86400) * 1000
    out = []
    for q, a in OPENREVIEW_QUERIES:
        if aspect != "all" and a != aspect:
            continue
        url = "https://api2.openreview.net/notes/search?" + urllib.parse.urlencode(
            {"term": q, "type": "terms", "content": "all", "group": "all",
             "source": "all", "limit": 50})
        try:
            notes = json.loads(get(url)).get("notes", [])
        except Exception as e:
            print("OpenReview error:", q, e)
            continue
        for n in notes:
            if n.get("cdate", 0) < cutoff:
                continue
            c = n.get("content", {})
            val = lambda k: (c.get(k) or {}).get("value", "") if isinstance(c.get(k), dict) else (c.get(k) or "")
            if not val("title") or not val("abstract"):
                continue  # reviews and comments have no abstract
            out.append({
                "title": val("title"), "abstract": " ".join(str(val("abstract")).split()),
                "date": time.strftime("%Y-%m-%d", time.gmtime(n["cdate"] / 1000)),
                "id": f"OpenReview {n['id']}", "url": f"https://openreview.net/forum?id={n['id']}",
                "source": f"OpenReview ({n.get('domain', '')})",
            })
        time.sleep(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--aspect", choices=["all", "detection", "weights"], default="all",
                    help="detection = lie/deception detection (FIBS side); weights = reading/editing weights (WSP side)")
    ap.add_argument("--out", default=str(OUT), help="output markdown file")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    known_files = list(FINDINGS.glob("*.md")) + list((REPO / "literature").glob("*.md"))
    known_text = " ".join(p.read_text() for p in known_files)
    known = norm(known_text) and re.sub(r"[^a-z0-9]", "", known_text.lower())

    papers = arxiv(args.days, args.aspect) + openalex(args.days, args.aspect) + openreview(args.days, args.aspect)
    merged = {}
    for p in papers:
        key = norm(p["title"])
        if not key:
            continue
        if key in merged:
            merged[key]["source"] += "; " + p["source"]
            if len(p["abstract"]) > len(merged[key]["abstract"]):
                merged[key]["abstract"] = p["abstract"]
        else:
            merged[key] = p

    rows = []
    for key, p in merged.items():
        text = p["title"] + " " + p["abstract"]
        if not TOPIC.search(text) or OFFTOPIC.search(text):
            continue
        aid = re.search(r"\d{4}\.\d{5}", p["id"])
        is_known = key[:40] in known or (aid and aid.group(0) in known_text)
        scores = sorted({m.group(0).strip() for m in SCORE.finditer(p["abstract"])})
        high = [s for s in scores if re.search(r"0?\.9[5-9]|9[5-9](\.\d)?\s?%|1\.0", s)]
        score = (3 if WHITEBOX.search(text) else 0) + (2 if BEATS.search(p["abstract"]) else 0) \
            + (2 if high else 0) + (1 if scores else 0)
        rows.append((not is_known, score, p, scores, high))

    rows.sort(key=lambda r: (r[0], r[1], r[2]["date"]), reverse=True)
    new = [r for r in rows if r[0]]

    lines = [
        "# SOTA candidates (generated)",
        "",
        f"Generated {dt.date.today()} by `research-findings/tools/find_sota.py --days {args.days} --aspect {args.aspect}`.",
        "Sources: arXiv API, OpenAlex citations of the seed papers, OpenReview. No API keys.",
        "",
        "Yardsticks: **FIBS** probes 0.987 AUROC (SHADE-Arena) / 0.980 (BashArena); "
        "**Weight-Space Programming** reads and edits raw weights (0.016% edit, MATH-500 7.9% -> 48.1%).",
        "A ⚑ flag means the abstract reports a score of 0.95 or higher, or claims to beat prior work. "
        "Benchmarks differ between papers, so read the paper before calling it better than FIBS.",
        "",
        f"**{len(new)} new in-field papers** (not yet in the literature files), out of {len(rows)} in-field, "
        f"{len(merged)} total.",
        "",
        "| Rank | Paper | Date | Scores in abstract | Flag | Found via |",
        "|---|---|---|---|---|---|",
    ]
    for i, (_, score, p, scores, high) in enumerate(new, 1):
        flag = "⚑" if (high or BEATS.search(p["abstract"])) else ""
        link = f"[{p['title']}]({p['url']})" if p["url"] else p["title"]
        lines.append(f"| {i} | {link} | {p['date']} | {', '.join(scores[:4]) or '-'} | {flag} | {p['source'][:60]} |")
    lines += ["", "## Abstracts of the top 25", ""]
    for _, score, p, scores, high in new[:25]:
        lines += [f"### {p['title']}", f"{p['id']} · {p['date']} · {p['source']}", "", p["abstract"][:1200], ""]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(new)} new candidates)")


if __name__ == "__main__":
    main()
