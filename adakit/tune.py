"""`adakit tune`: tensor TFLOPS and watts at a series of locked clock caps.

An unlocked card runs into its power limit, and at the limit the last few
hundred MHz cost a lot of watts for little speed. This locks the GPU clock
cap (nvidia-smi -lgc) step by step, runs the FP16 GEMM at each step, and
reports the fastest cap and the most efficient one within KEEP of the best
speed. Clocks are reset (nvidia-smi -rgc) when it ends, even on an error;
they also reset on reboot. Needs an elevated terminal.

nvidia-smi can only cap the clock. A real undervolt (a lower voltage at the
same clock) needs a V/F curve editor such as MSI Afterburner; the efficient
cap found here is a good target clock for that curve.
"""

import subprocess

from . import bench, sysbin

N = 8192
KEEP = 0.97  # "efficient" = best TFLOPS per watt among caps within 3% of the fastest
MIN_MHZ = 210


def caps(max_mhz, low=1800, step=150):
    return list(range(low, int(max_mhz), step)) + [int(max_mhz)]


def pick(rows):
    """rows: dicts with cap, tflops, watts. Returns (fastest, efficient)."""
    fastest = max(rows, key=lambda r: r["tflops"])
    near = [r for r in rows if r["tflops"] >= KEEP * fastest["tflops"]]
    return fastest, max(near, key=lambda r: r["tflops"] / r["watts"])


def _smi(target, *args):
    out = subprocess.run([sysbin.nvidia_smi(), *target, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit((out.stdout + out.stderr).strip() or f"nvidia-smi {' '.join(args)} failed")


def tune(log=print):
    torch = bench._torch()
    target = bench.smi_target(torch)
    max_mhz = float(bench.smi("clocks.max.sm", target))
    op = bench.make_op(torch, N, torch.float16)
    rows = []
    try:
        for cap in caps(max_mhz):
            _smi(target, f"--lock-gpu-clocks={MIN_MHZ},{cap}")
            tflops, _, _ = bench.timed(torch, op, N, warmup=10, iters=20)
            clock, power, capped = bench.sample_under_load(torch, op)
            row = {"cap": cap, "tflops": round(tflops, 2), "clock_mhz": clock, "watts": float(power), "power_capped": capped}
            rows.append(row)
            log(f"  cap {cap:>5} MHz: {tflops:6.1f} TFLOPS at {clock:.0f} MHz, {float(power):5.1f} W, "
                f"{tflops / float(power):.3f} TFLOPS/W{'  power-capped' if capped else ''}")
    finally:
        _smi(target, "--reset-gpu-clocks")
    return rows
