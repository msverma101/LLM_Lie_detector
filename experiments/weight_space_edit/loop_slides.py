"""Live HTML deck for the WSP research loop (stdlib only, so it can be rebuilt anywhere)."""
import json
import os
import sys
import time


def build_slides(state, out):
    """Self-contained HTML deck, rebuilt after every session (no external scripts)."""
    import html as H
    hist = state.get("history", [])
    s0 = state["theta0_sel"]["sel_score"]
    pts, best = [], s0
    for h in hist:
        if h["sel_score"] is not None:
            pts.append((h["session"], h["sel_score"], h.get("sel_score_sd") or 0, h["kept"]))
            if h["kept"]:
                best = max(best, h["sel_score"])
    W_, Hh, pad = 980, 320, 50
    ys = [s0] + [p[1] for p in pts] + [state["incumbent_sel"]["sel_score"]]
    lo, hi = min(ys) - 0.04, max(ys) + 0.04
    n = max(len(hist), 1)
    X = lambda i: pad + (W_ - 2 * pad) * (i / n)  # noqa: E731
    Y = lambda v: Hh - pad + 10 - (Hh - 2 * pad) * (v - lo) / (hi - lo + 1e-9)  # noqa: E731
    svg = [f'<svg viewBox="0 0 {W_} {Hh}" role="img" aria-label="selection score per session">',
           f'<line x1="{pad}" x2="{W_-pad}" y1="{Y(s0):.1f}" y2="{Y(s0):.1f}" class="base"/>',
           f'<text x="{W_-pad}" y="{Y(s0)-6:.1f}" text-anchor="end" class="lbl">original θ0 {s0:.3f}</text>']
    inc, path = s0, [f"M{X(0):.1f},{Y(s0):.1f}"]
    for (i, v, sd, kept) in pts:
        if kept:
            inc = v
        path.append(f"L{X(i):.1f},{Y(inc):.1f}")
        svg.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(v-sd):.1f}" y2="{Y(v+sd):.1f}" class="err"/>')
        svg.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="7" class="{"kept" if kept else "disc"}"><title>s{i:02d}: {v:.3f} ± {sd:.3f}</title></circle>')
        svg.append(f'<text x="{X(i):.1f}" y="{Hh-12}" text-anchor="middle" class="lbl">s{i}</text>')
    svg.insert(3, f'<path d="{" ".join(path)}" class="step"/>')
    svg.append("</svg>")
    rows = "".join(
        f"<tr class='{'k' if h['kept'] else ''}'><td>s{h['session']:02d}</td><td>{'✔' if h['kept'] else '✗'}</td>"
        f"<td class='num'>{(h['sel_score'] or 0):.3f} ± {(h.get('sel_score_sd') or 0):.3f}</td>"
        f"<td class='num'>{(h.get('sel_tpr_at_5fpr') or 0):.3f}</td><td>{'labels' if h.get('uses_labels') else 'label-free'}</td>"
        f"<td>{H.escape((h['idea'] or '')[:260])}</td></tr>" for h in hist)
    fin = state.get("final")
    fin_html = ""
    if fin:
        fin_html = ("<section><div class='kick'>Sealed test · evaluated once</div><h2>Did it transfer?</h2><table><tr><th>arm</th>"
                    "<th>TPR@5%</th><th>FPR@5%</th><th>macro-AUROC</th></tr>" + "".join(
                        f"<tr class='{'k' if r['arm']=='wsp_loop' else ''}'><td>{r['arm']}</td><td class='num'>{r['tpr5']:.3f}</td>"
                        f"<td class='num'>{r['fpr5']:.3f}</td><td class='num'>{r['test_macro_auroc']:.3f}</td></tr>" for r in fin)
                    + "</table></section>")
    notes = H.escape(state.get("notes", "")[-1500:])
    inc_sel = state["incumbent_sel"]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WSP Loop Live</title><style>
:root{{--bg:#0b0d14;--panel:#131726;--ink:#eef0fa;--muted:#9097b4;--line:#2a3050;--good:#34d399;--bad:#f43f5e;--acc:#a78bfa}}
@media (prefers-color-scheme: light){{:root:not([data-theme="dark"]){{--bg:#f6f5f0;--panel:#fff;--ink:#16182a;--muted:#5c6280;--line:#d8d6cc;--good:#059669;--bad:#e11d48;--acc:#6d28d9}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.5 system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:24px 16px 60px}}section{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0}}
h1{{font-size:clamp(28px,5vw,48px);margin:.2em 0}}h2{{margin:.1em 0 .5em}}.kick{{font:12px ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--acc)}}
.big{{display:flex;gap:28px;flex-wrap:wrap}}.big div{{min-width:150px}}.big b{{display:block;font-size:34px}}
svg{{width:100%;height:auto}}.base{{stroke:var(--muted);stroke-dasharray:6 5}}.step{{fill:none;stroke:var(--good);stroke-width:3}}
.kept{{fill:var(--good)}}.disc{{fill:var(--bad);opacity:.75}}.err{{stroke:var(--muted);stroke-width:2}}.lbl{{fill:var(--muted);font:12px ui-monospace,monospace}}
table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}}
.num{{font-family:ui-monospace,monospace;white-space:nowrap}}tr.k td{{color:var(--good)}}.wrap{{overflow-x:auto}}pre{{white-space:pre-wrap;color:var(--muted);font-size:13px}}
</style></head><body><main>
<section><div class="kick">Claude Impact Lab · llm-lying-detection · live</div><h1>Can Claude improve a lie detector by editing its weights?</h1>
<p>Claude Opus 5.5 gets the trained probe's weights in a Python kernel and rewrites them directly: no optimizer, no gradient step. Each session starts from the best weights so far; a candidate is kept only if it beats the incumbent on the selection families. The test families stay sealed until the end.</p>
<div class="big"><div><span class="muted">original θ0</span><b>{s0:.3f}</b></div><div><span class="muted">best so far ({H.escape(str(state['incumbent']))})</span><b style="color:var(--good)">{inc_sel['sel_score']:.3f}</b></div>
<div><span class="muted">sessions</span><b>{len(hist)}</b></div><div><span class="muted">kept</span><b>{sum(h['kept'] for h in hist)}</b></div></div>
<p class="muted">sel_score = mean share of lies caught at 1–5% false alarms, selection families (b1-bare, b1-followup, caving, negotiation). Error bars: bootstrap sd.</p></section>
<section><div class="kick">Progress</div><h2>Selection score per session</h2>{''.join(svg)}</section>
<section><div class="kick">What Claude tried</div><h2>Sessions</h2><div class="wrap"><table><tr><th>#</th><th>kept</th><th>sel_score</th><th>TPR@5%</th><th>data use</th><th>idea</th></tr>{rows}</table></div></section>
{fin_html}
<section><div class="kick">Claude's latest notes</div><pre>{notes}</pre></section>
<p class="muted">Updated {time.strftime('%Y-%m-%d %H:%M')} · model {state.get('model')} · tokens in/out {state.get('usage', [0, 0])[0]:,}/{state.get('usage', [0, 0])[1]:,}</p>
</main></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write(doc)


if __name__ == "__main__":
    # python loop_slides.py STATE.json OUT.html
    build_slides(json.load(open(sys.argv[1])), sys.argv[2])
