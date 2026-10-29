"""`adakit verify` and `adakit bench`.

verify: one FP16 GEMM with FP32 accumulate, timed while the SM clock is
sampled, divided by the card's peak at that clock. Peak is SMs x clock x
dense FP16 flops per SM per clock (1024 on Ada sm_89, 512 on Ampere sm_86).
Stock cards run FP32 accumulate at half that peak, so the efficiency reads
about 0.45-0.5 stock and 0.8-1.0 unlocked (0.83 measured at the power limit) whatever the clock or power cap.
No kernel driver, no admin rights.

bench: the matched A/B set used in gpu-unlock-ada #80/#82: torch.mm on
contiguous randn inputs, FP32 accumulate forced, warm-up then timed, median
TFLOPS, FP32 as the control row, plus a correctness check against FP64.
"""

import json
import statistics
import subprocess
import time
from pathlib import Path

FLOPS_PER_SM_CLK = {(8, 9): 1024, (8, 6): 512, (8, 7): 512}
UNLOCKED_AT, STOCK_BELOW = 0.70, 0.60
LOAD_S = 1.5


def verdict(efficiency):
    if efficiency is None:
        return "unknown", "This GPU generation has no reference peak in the kit."
    if efficiency >= UNLOCKED_AT:
        return "unlocked", "FP32-accumulate tensor math is running at full rate."
    if efficiency <= STOCK_BELOW:
        return "stock", "FP32-accumulate tensor math is at the stock half rate."
    return "unclear", "In between. Close other GPU work and run verify again."


def _torch():
    import warnings
    warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
    import torch  # deferred so `adakit check` works without PyTorch
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch can't see a CUDA GPU. Install a CUDA build of PyTorch (pip install torch --index-url https://download.pytorch.org/whl/cu128).")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch


def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""


def make_op(torch, n, dtype, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if dtype == "fp8":
        f8 = torch.float8_e4m3fn
        a = torch.randn(n, n, device="cuda").to(f8)
        b = torch.randn(n, n, device="cuda").to(f8).t()
        one = torch.tensor(1.0, device="cuda")
        return lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    return lambda: torch.mm(a, b)


def timed(torch, op, n, warmup=5, iters=20):
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); op(); e.record()
        torch.cuda.synchronize()
        vals.append(2.0 * n ** 3 / (s.elapsed_time(e) / 1000.0) / 1e12)
    return statistics.median(vals), min(vals), max(vals)


def sample_under_load(torch, op):
    """Keep the GEMM running until nvidia-smi (slow to start) answers, so the
    clock and power it reports are the loaded values. The driver refreshes
    these readings about once a second, so load the GPU for LOAD_S first;
    otherwise a short case reports the previous case's sample."""
    end = time.perf_counter() + LOAD_S
    while time.perf_counter() < end:
        op()
        torch.cuda.synchronize()
    proc = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,clocks_event_reasons.sw_power_cap",
                             "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True)
    while proc.poll() is None:
        op()
        torch.cuda.synchronize()
    clock, power, cap = (proc.stdout.read().strip().splitlines()[0]).split(", ")
    return float(clock), power, cap == "Active"


def verify():
    torch = _torch()
    props = torch.cuda.get_device_properties(0)
    per_clk = FLOPS_PER_SM_CLK.get((props.major, props.minor))
    op = make_op(torch, 4096, torch.float16)
    tflops, _, _ = timed(torch, op, 4096, warmup=10, iters=30)
    clock, power, capped = sample_under_load(torch, op)
    peak = props.multi_processor_count * per_clk * clock * 1e6 / 1e12 if per_clk else None
    eff = tflops / peak if peak else None
    state, text = verdict(eff)
    return {"device": props.name, "sms": props.multi_processor_count, "cc": f"{props.major}.{props.minor}",
            "fp16_fp32acc_tflops": round(tflops, 2), "clock_mhz": clock, "power_w": power, "power_capped": capped,
            "full_rate_peak_tflops": round(peak, 2) if peak else None,
            "efficiency": round(eff, 3) if eff else None, "state": state, "text": text}


CASES = [("FP32 (control)", 4096, "fp32", False), ("TF32", 4096, "fp32", True),
         ("FP16", 4096, "fp16", False), ("FP16", 8192, "fp16", False),
         ("BF16", 4096, "bf16", False), ("BF16", 8192, "bf16", False), ("FP8 e4m3", 8192, "fp8", False)]


def correctness(torch):
    worst = {}
    for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        a = torch.randn(2048, 2048, device="cuda", dtype=dtype)
        b = torch.randn(2048, 2048, device="cuda", dtype=dtype)
        got, ref = torch.mm(a, b).double(), torch.mm(a.double(), b.double())
        worst[name] = ((got - ref).abs().max() / ref.abs().max()).item()
    f8 = torch.float8_e4m3fn
    a8, b8 = torch.randn(2048, 2048, device="cuda").to(f8), torch.randn(2048, 2048, device="cuda").to(f8)
    one = torch.tensor(1.0, device="cuda")
    got = torch._scaled_mm(a8, b8.t().contiguous().t(), scale_a=one, scale_b=one, out_dtype=torch.float32).double()
    ref = torch.mm(a8.double(), b8.double())
    worst["fp8"] = ((got - ref).abs().max() / ref.abs().max()).item()
    return worst


def bench(label, out_dir, passes=2, log=print):
    torch = _torch()
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "fp8": "fp8"}
    result = {"label": label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "driver": smi("driver_version"),
              "power_limit_w": smi("power.limit"), "rows": []}
    log(f"{result['device']}  torch {result['torch']}  driver {result['driver']}  power limit {result['power_limit_w']} W")
    for p in range(1, passes + 1):
        for name, n, dt, tf32 in CASES:
            op = make_op(torch, n, dtypes[dt], tf32)
            med, lo, hi = timed(torch, op, n)
            clock, power, capped = sample_under_load(torch, op)
            row = {"case": name, "n": n, "pass": p, "median": round(med, 2), "min": round(lo, 2), "max": round(hi, 2),
                   "clock_mhz": clock, "power_w": power, "power_capped": capped,
                   "tflops_per_ghz": round(med / (clock / 1000.0), 2)}
            result["rows"].append(row)
            log(f"  {name:<15}{n:>6}  {med:8.2f} TFLOPS  {clock:.0f} MHz {power} W{'  power-capped' if capped else ''}  pass {p}")
    result["max_rel_error"] = correctness(torch)
    log(f"  max relative error vs FP64: " + ", ".join(f"{k} {v:.1e}" for k, v in result["max_rel_error"].items()))
    out = Path(out_dir) / f"result-{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out
