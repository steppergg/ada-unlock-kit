# FP8 on an unlocked card

Measured on an RTX 4060 Ti 8 GB (AD106, 128-bit bus), ASUS Z790, PyTorch `torch._scaled_mm` (e4m3 inputs), both states at the same locked clock (`nvidia-smi -lgc 2100`, no power cap in either run), so the ratio is the unlock alone.

| FP8 GEMM | Stock TFLOPS | Unlocked TFLOPS | Gain |
|---|---:|---:|---:|
| 8192, FP32 output | 70.2 | 131.5 | 1.87× |
| 8192, BF16 output | 70.4 | 133.4 | 1.89× |
| 16384, FP32 output | 71.0 | 132.1 | 1.86× |
| 16384, BF16 output | 70.9 | 136.3 | **1.92×** |
| FP16 8192 (reference) | 36.0 | 70.7 | 1.97× |

Stock is half rate whatever the output type. Unlocked, the tensor cores run twice as fast and the output writes start to show: an FP32 result matrix is twice the bytes of a BF16 one, and on a 128-bit bus that traffic is no longer hidden. Numerical results are unchanged (max relative error vs FP64 about 2e-4 in both states).

## Getting the most out of it

- **Use BF16 output** for FP8 GEMMs when your framework lets you (`out_dtype=torch.bfloat16` in `torch._scaled_mm`). It costs nothing at stock and gives back part of the gap once unlocked.
- **Bigger GEMMs** amortise the output writes: 16384 with BF16 output reached 1.92×.
- **Power is the real ceiling.** At normal boost clocks the unlocked card sits on its power limit (about 180 W on this card even with a 216 W limit), so the everyday FP8 gain was +59% (93.5 → 148.7 TFLOPS) rather than 2×. Raise the power limit, and a lower-voltage curve gives more clock per watt.
- It only helps work that runs FP8 matrix multiplies with FP32 accumulation on the tensor cores, such as `torch._scaled_mm` and libraries built on it. Games, DLSS and FP32 CUDA code don't change.

`adakit bench` includes the `FP8 e4m3 bf16-out` case, so your own report shows both output types.
