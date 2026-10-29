"""`adakit report`: a matched stock/unlocked comparison, ready to post.

Writes report.md (an issue comment), chart.html + chart.png, and entry.json
(one row for the Windows compatibility tracker). fairness() flags the things
that made earlier A/B posts misleading: different drivers or power limits,
and an FP32 control row that moved.
"""

import json
import shutil
import subprocess
from html import escape
from pathlib import Path

CASES = [("FP16", 8192), ("BF16", 8192), ("FP8 e4m3", 8192), ("TF32", 4096), ("FP32 (control)", 4096)]
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


def pct(gain):
    return f"{round(gain * 100):+d}%"


def load(path):
    r = json.loads(Path(path).read_text())
    best = {}
    for row in r["rows"]:
        key = (row["case"], row["n"])
        if row["median"] > best.get(key, {"median": 0})["median"]:
            best[key] = row
    return r, best


def fairness(stock, unlocked, sb, ub):
    notes = []
    if stock["device"] != unlocked["device"]:
        notes.append(f"Different GPUs: {stock['device']} vs {unlocked['device']}.")
    if stock["driver"] != unlocked["driver"]:
        notes.append(f"Driver changed between runs ({stock['driver']} vs {unlocked['driver']}).")
    if stock["power_limit_w"] != unlocked["power_limit_w"]:
        notes.append(f"Power limit changed ({stock['power_limit_w']} W vs {unlocked['power_limit_w']} W). Gains will mix the two effects.")
    if stock.get("torch") != unlocked.get("torch"):
        notes.append(f"PyTorch changed ({stock.get('torch')} vs {unlocked.get('torch')}).")
    ctl = ("FP32 (control)", 4096)
    if ctl in sb and ctl in ub:
        drift = ub[ctl]["median"] / sb[ctl]["median"] - 1
        if abs(drift) > 0.03:
            notes.append(f"The FP32 control moved {drift * 100:+.1f}%. It should stay flat; clocks or power differed between runs.")
    return notes


def table(sb, ub):
    rows = []
    for key in CASES:
        if key in sb and key in ub:
            s, u = sb[key], ub[key]
            rows.append({"case": key[0], "n": key[1], "stock": s["median"], "unlocked": u["median"],
                         "gain": u["median"] / s["median"] - 1,
                         "per_ghz": (s["tflops_per_ghz"], u["tflops_per_ghz"])})
    return rows


def chart_html(rows, title, foot):
    """Grouped horizontal bars, one scale for every row."""
    w, label_w, bar_h, gap, top = 980, 210, 22, 26, 70
    peak = max(max(r["stock"], r["unlocked"]) for r in rows)
    scale = (w - label_w - 150) / peak
    h = top + len(rows) * (2 * bar_h + gap) + 40
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Segoe UI, system-ui, sans-serif">',
             f'<rect width="{w}" height="{h}" fill="#10161b"/>',
             f'<text x="24" y="34" fill="#e6ecef" font-size="22" font-weight="700">{escape(title)}</text>',
             f'<rect x="{w - 300}" y="22" width="14" height="14" fill="#6c7a84"/><text x="{w - 280}" y="34" fill="#c9d3d8" font-size="14">stock</text>',
             f'<rect x="{w - 210}" y="22" width="14" height="14" fill="#4aa3df"/><text x="{w - 190}" y="34" fill="#c9d3d8" font-size="14">unlocked</text>']
    y = top
    for r in rows:
        name = f"{r['case'].replace(' (control)', '')} {r['n']}³" + (" (control)" if "control" in r["case"] else "")
        parts.append(f'<text x="24" y="{y + bar_h + 6}" fill="#e6ecef" font-size="15">{escape(name)}</text>')
        for i, (val, color) in enumerate(((r["stock"], "#6c7a84"), (r["unlocked"], "#4aa3df"))):
            by = y + i * bar_h
            bw = max(2, val * scale)
            parts.append(f'<rect x="{label_w}" y="{by + 2}" width="{bw:.1f}" height="{bar_h - 4}" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{label_w + bw + 8:.1f}" y="{by + bar_h - 6}" fill="#c9d3d8" font-size="13">{val:.1f}</text>')
        parts.append(f'<text x="{w - 24}" y="{y + bar_h + 6}" fill="{"#7fd19b" if r["gain"] > 0.03 else "#c9d3d8"}" font-size="15" font-weight="700" text-anchor="end">{pct(r["gain"])}</text>')
        y += 2 * bar_h + gap
    parts.append(f'<text x="24" y="{h - 16}" fill="#8a98a2" font-size="12">{escape(foot)}</text></svg>')
    svg = "".join(parts)
    return f"<!doctype html><meta charset='utf-8'><style>body{{margin:0;background:#10161b}}</style>{svg}", w, h


def render_png(html_path, png_path, w, h):
    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if not browser:
        return False
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                    f"--screenshot={Path(png_path).resolve()}", Path(html_path).resolve().as_uri()], capture_output=True, timeout=60)
    return Path(png_path).exists()


def entry(facts, gpu, stock, rows, loader):
    by = {(r["case"], r["n"]): r for r in rows}
    g = lambda k: round(by[k]["gain"] * 100) if k in by else None
    return {"card": gpu.get("name"), "device_id": f"10de:{gpu.get('device')}", "subsystem": gpu.get("subsystem"),
            "vbios": gpu.get("vbios"), "board": facts.get("board"), "bios": facts.get("bios"),
            "laptop": facts.get("laptop"), "os": facts.get("os"), "driver": stock.get("driver"),
            "loader": loader, "power_limit_w": stock.get("power_limit_w"),
            "fp16_gain_pct": g(("FP16", 8192)), "bf16_gain_pct": g(("BF16", 8192)),
            "fp8_gain_pct": g(("FP8 e4m3", 8192)), "tf32_gain_pct": g(("TF32", 4096)),
            "result": "unlocked" if (g(("FP16", 8192)) or 0) >= 30 else "no gain"}


def build(stock_path, unlocked_path, out_dir, facts=None, loader=None):
    stock, sb = load(stock_path)
    unlocked, ub = load(unlocked_path)
    rows = table(sb, ub)
    notes = fairness(stock, unlocked, sb, ub)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md = [f"### {stock['device']}: matched stock / unlocked on Windows", ""]
    if facts:
        gpu = facts["nvidia_gpus"][0] if facts["nvidia_gpus"] else {}
        md += [f"- Card: `10de:{gpu.get('device')}`, subsystem `{gpu.get('subsystem')}`, VBIOS `{gpu.get('vbios')}`",
               f"- Board: {facts.get('board')} (BIOS {facts.get('bios')}), {facts.get('os')}"]
    md += [f"- Driver {stock['driver']}, PyTorch {stock['torch']}, power limit {stock['power_limit_w']} W",
           f"- Loader: {loader or 'not recorded'}", "",
           "| TFLOPS, cuBLAS | Stock | Unlocked | Gain | Per GHz stock → unlocked |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['case']} {r['n']}³ | {r['stock']:.2f} | {r['unlocked']:.2f} | {pct(r['gain'])} | "
                  f"{r['per_ghz'][0]:.1f} → {r['per_ghz'][1]:.1f} |")
    err = unlocked.get("max_rel_error", {})
    md += ["", "Median of 20 GEMMs, best of 2 passes. Max relative error vs FP64 (unlocked): "
           + ", ".join(f"{k} {v:.1e}" for k, v in err.items()) + "."]
    if notes:
        md += ["", "**Comparison notes**", ""] + [f"- {n}" for n in notes]
    md += ["", "Made with ada-unlock-kit."]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")

    html, w, h = chart_html(rows, f"{stock['device']} · stock vs unlocked",
                            f"driver {stock['driver']} · PyTorch {stock['torch'].split('+')[0]} · {stock['power_limit_w']} W · median of 20, best of 2")
    (out / "chart.html").write_text(html, encoding="utf-8")
    png = render_png(out / "chart.html", out / "chart.png", w, h)

    if facts and facts["nvidia_gpus"]:
        (out / "entry.json").write_text(json.dumps(entry(facts, facts["nvidia_gpus"][0], stock, rows, loader), indent=2))
    return rows, notes, png
