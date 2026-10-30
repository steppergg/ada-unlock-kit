"""`adakit check`: is this PC a reasonable candidate, and what to fix first.

assess() is a pure function of the gathered facts so it can be tested
without a GPU.
"""

from . import compat

OK, WARN, STOP, INFO = "ok", "warn", "stop", "info"

WINDOWS_ADVICE = {
    "working": (OK, "Unlocked on Windows before with the generic loader. Still run replay mode first."),
    "port-only": (WARN, "Only per-board ports have worked on Windows. The generic loader is untested on this card: replay mode first, and report what happens."),
    "untested": (WARN, "No Windows report yet. Run replay mode first, and only try unlock mode if replay is clean."),
    "partial": (WARN, "Some boards lose the speed-up or never get it. Expect to verify carefully."),
    "not-needed": (STOP, "This card already runs at full rate. There is nothing to unlock."),
    "no": (STOP, "This card can't be unlocked with this method."),
    "unknown": (WARN, "This device ID isn't in the table. Treat it as untested."),
}


def assess(facts):
    """Return (gpu_reports, system_findings). Each finding is (level, text)."""
    gpus = []
    for g in facts["nvidia_gpus"]:
        entry = compat.lookup(g["device"])
        level, advice = WINDOWS_ADVICE[entry["windows"]]
        gpus.append({"gpu": g, "entry": entry, "level": level, "advice": advice})

    sysf = []
    if not facts["nvidia_gpus"]:
        sysf.append((STOP, "No NVIDIA GPU found."))
    if len(facts["nvidia_gpus"]) > 1:
        sysf.append((WARN, "More than one NVIDIA GPU. Reports with several cards are rare; note which one drives the display."))
    if facts["uefi"] is False:
        sysf.append((STOP, "Windows booted in legacy BIOS mode. The loader needs UEFI boot."))
    if facts["secure_boot"]:
        sysf.append((WARN, "Secure Boot is on. The loader is unsigned, so Secure Boot must be off for the unlock boot."))
    elif facts["secure_boot"] is False:
        sysf.append((OK, "Secure Boot is off."))
    if facts["bitlocker"] in ("on", "encrypting", "suspended"):
        sysf.append((WARN, "BitLocker is on for the system drive. Back up the recovery key before changing Secure Boot or boot order."))
    if facts["fast_startup"]:
        sysf.append((WARN, "Fast Startup is on. Turn it off: a shutdown must be a real power-off for a cold boot."))
    if facts["laptop"]:
        sysf.append((WARN, "This looks like a laptop. Reported laptop successes ran with the dGPU in MUX / dGPU-only mode; hybrid mode can lose the unlock. On MUXless laptops Windows idle power-down (RTD3) drops it; keep a CUDA process running (win #154)."))
    if facts["other_gpus"] and facts["nvidia_gpus"]:
        sysf.append((INFO, "Other display adapters: " + ", ".join(facts["other_gpus"]) + "."))
    sysf.append((INFO, "The unlock is volatile: it lasts until the next restart or power-off, so the stick runs on every boot."))
    sysf.append((INFO, "It speeds up FP16/BF16/TF32/FP8 tensor math with FP32 accumulate (AI work). Games and DLSS don't change."))
    return gpus, sysf


MARK = {OK: "[ ok ]", WARN: "[warn]", STOP: "[STOP]", INFO: "[info]"}


def render(facts, gpus, sysf):
    lines = [f"{facts['os']} | board {facts['board'] or '?'} | BIOS {facts['bios'] or '?'}", ""]
    for r in gpus:
        g, e = r["gpu"], r["entry"]
        lines.append(f"GPU  {g['name']}  [10de:{g['device']}] subsystem {g['subsystem'] or '?'}")
        lines.append(f"     VBIOS {g.get('vbios', '?')} | driver {g.get('driver', '?')} | power {g.get('power_limit_w', '?')} W (max {g.get('power_max_w', '?')} W)")
        lines.append(f"     {e['name']} | {e['chip']} | Windows: {e['windows']} | Linux: {e['linux']}")
        if e["notes"]:
            lines.append(f"     {e['notes']}")
        lines.append(f"{MARK[r['level']]} {r['advice']}")
        for ref in e["refs"]:
            lines.append(f"       {compat.ref_url(ref)}")
        lines.append("")
    for level, text in sysf:
        lines.append(f"{MARK[level]} {text}")
    return "\n".join(lines)
