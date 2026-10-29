"""ada-unlock-kit: check, stick, mode, verify, bench, report.

  python -m adakit check                      is this PC a candidate? (read-only)
  python -m adakit stick F: --source usbroot  copy a verified stick (replay mode)
  python -m adakit mode F: unlock|replay|disarm|status
  python -m adakit verify                     am I unlocked right now?
  python -m adakit bench stock                matched benchmark, saved to results/
  python -m adakit report results/result-stock.json results/result-unlocked.json
"""

import argparse
import json
import sys
from pathlib import Path

from . import bench, check, report, stick, winsys

RESULTS = Path.cwd() / "results"


def cmd_check(a):
    facts = winsys.gather()
    gpus, sysf = check.assess(facts)
    if a.json:
        print(json.dumps({"facts": facts, "gpus": gpus, "system": sysf}, indent=2, default=str))
    else:
        print(check.render(facts, gpus, sysf))
    return 1 if any(level == check.STOP for level, _ in sysf) else 0


def cmd_stick(a):
    letter = stick.normalize(a.drive)
    info = stick.drive_info(letter)
    stick.check_drive(info, letter)
    print(f"{letter} {info.get('label') or ''} | {info.get('model')} | FAT32 | {int(info['free']) // 2**20} MiB free")
    if not a.yes:
        ok = input(f"Copy the unlock files to {letter}? Existing kit files are replaced, BOOTX64.EFI is backed up. [y/N] ")
        if ok.strip().lower() != "y":
            print("Nothing written.")
            return 1
    manifest, (mode, armed) = stick.build(a.source, letter, unlock=a.unlock)
    for rel, meta in manifest["files"].items():
        print(f"  {rel:<36}{meta['bytes']:>10} bytes  {meta['sha256'][:16]}...")
    print(f"Written and verified. Mode: {mode}{', ARMED (auto-runs)' if armed else ''}.")
    print("Next: power off fully, boot the stick from the firmware boot menu, and check \\loader.log on the stick afterwards.")
    return 0


def cmd_mode(a):
    letter = stick.normalize(a.drive)
    stick.check_drive(stick.drive_info(letter), letter)
    root = Path(letter + "\\")
    if a.mode == "status":
        mode, armed = stick.stick_mode(root)
        problems = stick.verify(root)
    else:
        mode, armed = stick.set_mode(root, a.mode)
        problems = []
    print(f"{letter} mode: {mode}{' | ARMED: auto-runs after 5 s' if armed else ''}")
    for p in problems:
        print(f"  [warn] {p}")
    return 0


def cmd_verify(a):
    r = bench.verify()
    if a.json:
        print(json.dumps(r, indent=2))
        return 0
    print(f"{r['device']} | {r['sms']} SMs | {r['clock_mhz']:.0f} MHz under load{' (power-capped)' if r['power_capped'] else ''}")
    if r["efficiency"] is not None:
        print(f"FP16 GEMM, FP32 accumulate: {r['fp16_fp32acc_tflops']} TFLOPS of {r['full_rate_peak_tflops']} full-rate peak at this clock "
              f"= {r['efficiency']:.2f} (stock is about 0.5)")
    print(f"{r['state'].upper()}: {r['text']}")
    return 0 if r["state"] == "unlocked" else 2


def cmd_bench(a):
    out = bench.bench(a.label, a.out)
    facts = winsys.gather()
    (Path(a.out) / f"facts-{a.label}.json").write_text(json.dumps(facts, indent=2))
    print(f"Saved {out}")
    return 0


def cmd_report(a):
    facts_path = Path(a.unlocked).with_name(Path(a.unlocked).name.replace("result-", "facts-"))
    facts = json.loads(facts_path.read_text()) if facts_path.exists() else None
    rows, notes, png = report.build(a.stock, a.unlocked, a.out, facts, a.loader)
    for r in rows:
        print(f"  {r['case']:<15}{r['n']:>6}  {r['stock']:7.2f} -> {r['unlocked']:7.2f}  {report.pct(r['gain'])}")
    for n in notes:
        print(f"  [note] {n}")
    print(f"Wrote {a.out}\\report.md{', chart.png' if png else ', chart.html (no Edge/Chrome for PNG)'}"
          f"{', entry.json' if facts else ''}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="adakit", description="Windows helper for nv-uefi-unlocker on RTX 40-series cards.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="read-only: is this PC a candidate?")
    c.add_argument("--json", action="store_true")
    s = sub.add_parser("stick", help="copy and verify the unlock files on a FAT32 USB stick")
    s.add_argument("drive")
    s.add_argument("--source", default="usbroot", help="folder made by upstream make-usbroot.ps1")
    s.add_argument("--unlock", action="store_true", help="write in unlock mode (default: replay)")
    s.add_argument("--yes", action="store_true")
    m = sub.add_parser("mode", help="switch a kit stick between replay and unlock")
    m.add_argument("drive")
    m.add_argument("mode", choices=["status", "replay", "unlock", "disarm"])
    v = sub.add_parser("verify", help="benchmark check: is FP32-accumulate at full rate?")
    v.add_argument("--json", action="store_true")
    b = sub.add_parser("bench", help="matched benchmark set")
    b.add_argument("label", help="e.g. stock or unlocked")
    b.add_argument("--out", default=str(RESULTS))
    r = sub.add_parser("report", help="compare two bench results")
    r.add_argument("stock")
    r.add_argument("unlocked")
    r.add_argument("--out", default=str(RESULTS / "report"))
    r.add_argument("--loader", help="loader version, e.g. 'nv-uefi-unlocker v2.1.5'")
    a = p.parse_args(argv)
    try:
        return {"check": cmd_check, "stick": cmd_stick, "mode": cmd_mode, "verify": cmd_verify,
                "bench": cmd_bench, "report": cmd_report}[a.cmd](a)
    except stick.StickError as e:
        print(f"[STOP] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
