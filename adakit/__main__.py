"""ada-unlock-kit: check, stick, mode, verify, bench, report.

  python -m adakit check                      is this PC a candidate? (read-only)
  python -m adakit stick F: --source usbroot  copy a verified stick (replay mode)
  python -m adakit status [F:]                did this boot come from the stick? last loader run
  python -m adakit mode F: unlock|replay|disarm|status|reconnect-safe|reconnect-auto
  python -m adakit verify                     am I unlocked right now?
  python -m adakit bench stock                matched benchmark, saved to results/
  python -m adakit report results/result-stock.json results/result-unlocked.json
"""

import argparse
import json
import sys
import time
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
    stick.check_source(a.source)  # fail before asking, not after
    if not a.other_pc:
        problem = stick.gpu_family_problem(winsys.gather()["nvidia_gpus"])
        if problem:
            raise stick.StickError(problem)
    warnings = stick.asset_warnings({n: Path(a.source) / n for n in stick.KNOWN_ASSETS})
    for w in warnings:
        print(f"  [warn] {w}")
    print(f"{letter} {info.get('label') or ''} | {info.get('model')} | FAT32 | {int(info['free']) // 2**20} MiB free")
    if not a.yes:
        try:
            ok = input(f"Copy the unlock files to {letter}?{' The files above differ from the known-good build.' if warnings else ''} "
                       "Existing kit files are replaced, BOOTX64.EFI is backed up. [y/N] ")
        except EOFError:
            ok = ""
        if ok.strip().lower() != "y":
            print("Nothing written.")
            return 1
    manifest, (mode, armed) = stick.build(a.source, letter, unlock=a.unlock)
    for rel, meta in manifest["files"].items():
        print(f"  {rel:<36}{meta['bytes']:>10} bytes  {meta['sha256'][:16]}...")
    print(f"Written and verified. Mode: {mode}{', ARMED (auto-runs)' if armed else ''}.")
    known = stick.KNOWN_LOADERS.get(manifest["loader_sha256"])
    print(f"Loader SHA-256 {manifest['loader_sha256']}: " + (f"matches the official {known}." if known else
          "not an official release binary the kit knows; compare it with the build you trust."))
    print("Next: power off fully, boot the stick from the firmware boot menu, and check \\loader.log on the stick afterwards.")
    return 0


def cmd_mode(a):
    letter = stick.normalize(a.drive)
    stick.check_drive(stick.drive_info(letter), letter)
    root = Path(letter + "\\")
    if a.mode == "status":
        return show_status(letter)
    mode, armed = stick.set_mode(root, a.mode)
    print(f"{letter} mode: {mode}{' | ARMED: auto-runs after 5 s' if armed else ''} | reconnect: {stick.reconnect_mode(root)}")
    run = stick.last_run(root)
    if stick.marker_ignored(root, run["version"] if run else None):
        print("  [warn] This loader is older than v2.1.6 and ignores the reconnect-safe marker. Put the v2.1.6 release on the stick.")
    return 0


def clock(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def stick_session():
    """(drive, last_run, came_from_stick) for the first stick found."""
    drives = stick.find_sticks()
    if not drives:
        return None, None, None
    run = stick.last_run(drives[0] + "\\")
    via = stick.came_from_stick(run["time"] if run else None, winsys.last_boot())
    return drives[0], run, via


def show_status(letter=None):
    boot = winsys.last_boot()
    drives = [letter] if letter else stick.find_sticks()
    if not drives:
        print(r"No unlock stick found (a FAT32 drive with \EFI\BOOT\BOOTX64.EFI and the payload or loader.log).")
        print(f"This boot  {clock(boot) if boot else '?'}  (run `adakit verify` to measure the current state)")
        return 1
    d = drives[0]
    root = Path(d + "\\")
    mode, armed = stick.stick_mode(root)
    run = stick.last_run(root)
    ver = f"nv-uefi-unlocker {run['version']}" + (f" (built {run['built']})" if run and run.get("built") else "") if run and run.get("version") else "loader version unknown"
    print(f"Stick      {d}  {ver} | mode {mode}{' | ARMED: auto-runs after 5 s' if armed else ''}")
    lh = stick.loader_hash(root)
    known = stick.KNOWN_LOADERS.get(lh) if lh else None
    print((f"Loader     BOOTX64.EFI sha256 {lh[:16]}..." + (f" = official {known}" if known else " (own build or unknown)"))
          if lh else r"Loader     no \EFI\BOOT\BOOTX64.EFI")
    print(f"Reconnect  {stick.reconnect_mode(root)}")
    if stick.marker_ignored(root, run["version"] if run else None):
        print("  [warn] This loader is older than v2.1.6 and ignores tx-reconnect-nonrecursive.txt. Put the v2.1.6 release on the stick.")
    if run:
        if run["aborted"]:
            what = "aborted at the confirm screen, nothing unlocked"
        elif run["unlocked"]:
            what = "unlocked" + (", handed off to Windows" if run["chained"] else ", did NOT reach Windows handoff")
        elif run["unlocked"] is False:
            what = "ran without unlocking (replay mode?)"
        else:
            what = "no transaction result in the log"
        print(f"Last run   {clock(run['time'])}  {what}")
        for e in run["errors"] + run.get("reasons", []):
            print(f"           {e}")
        if run.get("hung_at_reconnect"):
            print("           log stops inside the recursive GOP reconnect: the board hangs there (nvuefi #52/#61).")
            print(f"           Fix (v2.1.6+): `adakit mode {d} reconnect-safe`, then power off fully and boot the stick again.")
    else:
        print("Last run   no loader.log on the stick yet")
    via = stick.came_from_stick(run["time"] if run else None, boot)
    if boot:
        verdict = {True: "came from the stick", False: "did NOT come from the stick: expect stock rate", None: ""}[via]
        if via and run and run["unlocked"] is False:
            verdict = "came from the stick, but that run didn't unlock"
        print(f"This boot  {clock(boot)}  {verdict}")
    for p in stick.verify(root):
        print(f"  [warn] {p}")
    order = stick.boot_order_warning()
    if order:
        print(f"  [warn] {order}")
    print("Run `adakit verify` to measure the current state.")
    return 0


def cmd_status(a):
    return show_status(stick.normalize(a.drive) if a.drive else None)


def cmd_verify(a):
    r = bench.verify()
    if a.json:
        print(json.dumps(r, indent=2))
        return 0
    print(f"{r['device']} | {r['sms']} SMs | {r['clock_mhz']:.0f} MHz under load{' (power-capped)' if r['power_capped'] else ''}")
    if r["efficiency"] is not None:
        print(f"FP16 GEMM, FP32 accumulate: {r['fp16_fp32acc_tflops']} TFLOPS of {r['full_rate_peak_tflops']} full-rate peak at this clock "
              f"= {r['efficiency']:.2f} (stock is about 0.5)")
    m = r["mma"] or {}
    if "efficiency" in m:
        print(f"mma.sync loop (tensor cores only, no memory): {m['tflops']} TFLOPS at {m['clock_mhz']:.0f} MHz "
              f"= {m['efficiency']:.2f} of full rate (stock is about 0.5)")
        if r["efficiency"] is not None and bench.verdict(r["efficiency"])[0] != r["state"]:
            print("The GEMM and the mma loop disagree; the loop decides. A low GEMM with a full-rate loop usually means a power or memory limit, not a lost unlock.")
    elif "error" in m:
        print(f"mma.sync loop skipped ({m['error']}); the verdict uses the GEMM alone.")
    print(f"{r['state'].upper()}: {r['text']}")
    return 0 if r["state"] == "unlocked" else 2


def cmd_bench(a):
    out = bench.bench(a.label, a.out)
    facts = winsys.gather()
    drive, run, via = stick_session()
    facts["boot_via_stick"] = via
    if via and run and run.get("version"):
        facts["loader"] = f"nv-uefi-unlocker {run['version']}"
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
    p = argparse.ArgumentParser(prog="adakit", description="Windows helper for nv-uefi-unlocker (RTX 40 and 30 series).")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="read-only: is this PC a candidate?")
    c.add_argument("--json", action="store_true")
    s = sub.add_parser("stick", help="copy and verify the unlock files on a FAT32 USB stick")
    s.add_argument("drive")
    s.add_argument("--source", default="usbroot", help="folder made by upstream make-usbroot.ps1")
    s.add_argument("--unlock", action="store_true", help="write in unlock mode (default: replay)")
    s.add_argument("--yes", action="store_true")
    s.add_argument("--other-pc", action="store_true", help="skip the RTX 40 check (stick for a different PC)")
    st = sub.add_parser("status", help="did this boot come from the stick? last loader run, boot order")
    st.add_argument("drive", nargs="?")
    m = sub.add_parser("mode", help="switch a kit stick between replay and unlock; reconnect-safe for boards that hang after the unlock")
    m.add_argument("drive")
    m.add_argument("mode", choices=["status", "replay", "unlock", "disarm", "reconnect-safe", "reconnect-auto"])
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
        return {"check": cmd_check, "stick": cmd_stick, "mode": cmd_mode, "status": cmd_status, "verify": cmd_verify,
                "bench": cmd_bench, "report": cmd_report}[a.cmd](a)
    except stick.StickError as e:
        print(f"[STOP] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
