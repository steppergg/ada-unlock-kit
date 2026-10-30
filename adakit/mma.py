"""Raw tensor-core throughput: a loop of mma.sync m16n8k16 FP16 x FP16 -> FP32.

cuBLAS GEMM mixes in memory traffic, tile heuristics and the power limit, so a
power-capped unlocked card can read "unclear" (gpu-unlock-ada #108: a 4090 is
flat in cuBLAS at 450 W but 2.00x here). This loop touches no memory, so it
reads the FP32-accumulate gate alone: about 0.5 of full rate stock, about 1.0
unlocked (nv-uefi-unlocker #63: 1.98-2.01x; Windows-GPU-UNLOCK #154 used the
same check).

Compiled at run time with the NVRTC that ships inside PyTorch and launched on
PyTorch's CUDA context, so it needs nothing beyond `adakit verify`'s PyTorch.
"""

import ctypes
import os
import time
from pathlib import Path

from . import sysbin

CHAINS = 8  # independent accumulators per warp, enough to hide mma latency
WARPS_PER_BLOCK = 8
BLOCKS_PER_SM = 4
FLOPS_PER_MMA = 2 * 16 * 8 * 16
TARGET_S = 0.2

SRC = r"""
extern "C" __global__ void mma_loop(float* out, int iters) {
    unsigned a = 0x3c00u * (threadIdx.x & 1u);  // 1.0 or 0.0 halves; keeps sums finite
    float c[%(chains)d][4] = {};
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < %(chains)d; ++j)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%%0,%%1,%%2,%%3}, {%%4,%%5,%%6,%%7}, {%%8,%%9}, {%%0,%%1,%%2,%%3};"
                         : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                         : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
    }
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < %(chains)d; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
""" % {"chains": CHAINS}


def flops(blocks, iters):
    return blocks * WARPS_PER_BLOCK * iters * CHAINS * FLOPS_PER_MMA


def _check(lib, rc, what):
    if rc != 0:
        raise RuntimeError(f"{lib} {what} failed with error {rc}")


def _nvrtc(torch):
    lib = Path(torch.__file__).parent / "lib"
    found = sorted(p for p in lib.glob("nvrtc64_*.dll") if ".alt." not in p.name)
    if not found:
        raise RuntimeError("PyTorch has no NVRTC DLL")
    os.add_dll_directory(str(lib))  # nvrtc loads nvrtc-builtins from the same folder
    return ctypes.WinDLL(str(found[-1]))


def _ptx(torch, cc):
    nv = _nvrtc(torch)
    prog = ctypes.c_void_p()
    _check("nvrtc", nv.nvrtcCreateProgram(ctypes.byref(prog), SRC.encode(), b"mma.cu", 0, None, None), "create")
    opts = (ctypes.c_char_p * 1)(f"--gpu-architecture=compute_{cc[0]}{cc[1]}".encode())
    rc = nv.nvrtcCompileProgram(prog, 1, opts)
    if rc != 0:
        size = ctypes.c_size_t()
        nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        nv.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("nvrtc compile failed: " + log.value.decode(errors="replace"))
    size = ctypes.c_size_t()
    _check("nvrtc", nv.nvrtcGetPTXSize(prog, ctypes.byref(size)), "ptx size")
    ptx = ctypes.create_string_buffer(size.value)
    _check("nvrtc", nv.nvrtcGetPTX(prog, ptx), "ptx")
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.raw


def make_op(torch):
    """Compile the loop and return (launch, blocks): launch(iters) queues one
    run on PyTorch's current stream."""
    props = torch.cuda.get_device_properties(0)
    ptx = _ptx(torch, (props.major, props.minor))
    torch.zeros(1, device="cuda")  # make PyTorch's context current on this thread
    cu = ctypes.WinDLL(str(sysbin.SYSTEM32 / "nvcuda.dll"))
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    _check("cuda", cu.cuModuleLoadData(ctypes.byref(mod), ptx), "module load")
    _check("cuda", cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"mma_loop"), "get function")
    blocks = props.multi_processor_count * BLOCKS_PER_SM
    threads = WARPS_PER_BLOCK * 32
    out = torch.empty(blocks * threads, device="cuda")
    ptr = ctypes.c_uint64(out.data_ptr())
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3

    def launch(iters):
        n = ctypes.c_int(iters)
        args = (ctypes.c_void_p * 2)(ctypes.addressof(ptr), ctypes.addressof(n))
        _check("cuda", cu.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream, args, None), "launch")

    launch.keep = (out, mod)  # the buffer and module must outlive every launch
    return launch, blocks


def measure(torch):
    """Median TFLOPS of the loop, sized to run about TARGET_S per launch.
    Returns (tflops, op) where op() runs one launch to completion."""
    launch, blocks = make_op(torch)
    iters = 256
    while True:  # grow until a launch is long enough to time cleanly
        t = time.perf_counter(); launch(iters); torch.cuda.synchronize()
        dt = time.perf_counter() - t
        if dt >= TARGET_S / 4 or iters >= 1 << 24:
            break
        iters *= 4
    iters = max(1, int(iters * TARGET_S / dt))
    for _ in range(3):  # let clocks settle after the preceding GEMM
        launch(iters)
    torch.cuda.synchronize()
    vals = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); launch(iters); e.record()
        torch.cuda.synchronize()
        vals.append(flops(blocks, iters) / (s.elapsed_time(e) / 1000.0) / 1e12)
    vals.sort()

    def op():
        launch(iters)
    return vals[len(vals) // 2], op
