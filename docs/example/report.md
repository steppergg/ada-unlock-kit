### NVIDIA GeForce RTX 4060 Ti: matched stock / unlocked on Windows

- Card: `10de:2803`, subsystem `1043:88fb`, VBIOS `95.06.26.00.c0`
- Board: ASUSTeK COMPUTER INC. ROG MAXIMUS Z790 APEX (BIOS American Megatrends Inc. 3107), Microsoft Windows 10 Pro build 19045
- Driver 581.42, PyTorch 2.11.0+cu128, power limit 216.00 W
- Loader: nv-uefi-unlocker v2.1.5 + PR #26

| TFLOPS, cuBLAS | Stock | Unlocked | Gain | Per GHz stock → unlocked |
|---|---:|---:|---:|---:|
| FP16 8192³ | 47.86 | 80.78 | +69% | 17.1 → 33.2 |
| BF16 8192³ | 47.77 | 82.47 | +73% | 17.1 → 33.5 |
| FP8 e4m3 8192³ | 93.15 | 148.17 | +59% | 33.4 → 62.5 |
| TF32 4096³ | 22.99 | 38.24 | +66% | 8.2 → 15.4 |
| FP32 (control) 4096³ | 12.81 | 12.73 | -1% | 5.4 → 5.4 |

Median of 20 GEMMs, best of 2 passes. Max relative error vs FP64 (unlocked): fp16 2.6e-04, bf16 2.2e-03, fp8 1.9e-04.

Made with ada-unlock-kit.