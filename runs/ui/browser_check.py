import sys, json, time
time.sleep(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
from playwright.sync_api import sync_playwright
BASE, TOK = sys.argv[1], sys.argv[2]
out = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1100, "height": 900}); pg = ctx.new_page()
    errs = []; pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"{BASE}/?t={TOK}"); pg.wait_for_selector(".verdict")
    out["verdict"] = pg.inner_text(".verdict h2"); out["status"] = pg.inner_text("#status")
    out["arm_rows"] = pg.locator("#tab-result table tr").count()
    pg.screenshot(path="runs/ui/shots/1_result.png", full_page=True)
    for i, name in enumerate(["why", "methods"], 2):
        pg.click(f"#tabs button:nth-child({ {'why':2,'methods':3}[name] })"); pg.wait_for_timeout(200)
        pg.screenshot(path=f"runs/ui/shots/{i}_{name}.png", full_page=True)
    pg.click("#tabs button:nth-child(4)")
    pg.click("button.preset >> nth=3"); pg.click("button.primary")
    try:
        pg.wait_for_selector("#try-out table", timeout=60000)
    except Exception:
        print("TRY OUTPUT:", pg.inner_text("#try-out")); raise
    out["try_rows"] = pg.locator("#try-out table tr").count(); out["try_flags"] = pg.locator("#try-out .flag").all_inner_texts()
    pg.screenshot(path="runs/ui/shots/4_try.png", full_page=True)
    # XSS probe: markup in the answer must be shown as text, never executed or injected
    pg.fill("#f-a", "<img src=x onerror=window.__pwned=1><b>bold</b>"); pg.click("button.primary"); pg.wait_for_timeout(4000)
    out["xss_executed"] = pg.evaluate("window.__pwned === 1"); out["injected_nodes"] = pg.locator("#try-out img, #tab-try img").count()
    pg.click("#lang"); out["lang_after_toggle"] = pg.inner_text("#t-title")
    # mobile width
    pg.set_viewport_size({"width": 390, "height": 800}); pg.screenshot(path="runs/ui/shots/5_mobile.png", full_page=True)
    out["horizontal_overflow_mobile"] = pg.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
    out["console_errors"] = len(errs); out["first_console_error"] = errs[0][:160] if errs else None
    out["bar_widths_px"] = pg.evaluate("[...document.querySelectorAll('.bar>i')].slice(0,6).map(e=>Math.round(e.getBoundingClientRect().width))")
    b.close()
print(json.dumps(out, indent=1))
