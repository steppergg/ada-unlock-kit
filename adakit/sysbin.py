"""Full paths for every program the kit starts.

Windows looks in the current folder before PATH when a program is started by
bare name, so `nvidia-smi` run from Downloads could start a planted
nvidia-smi.exe there. Every launch goes through these absolute paths instead.
"""

import os
from pathlib import Path

SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
POWERSHELL = str(SYSTEM32 / "WindowsPowerShell" / "v1.0" / "powershell.exe")
BCDEDIT = str(SYSTEM32 / "bcdedit.exe")
_SMI = [SYSTEM32 / "nvidia-smi.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"]
BROWSERS = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Google" / "Chrome" / "Application" / "chrome.exe"]


def nvidia_smi():
    """nvidia-smi from the driver's install locations, never the current folder."""
    found = next((p for p in _SMI if p.is_file()), None)
    if not found:
        raise FileNotFoundError("nvidia-smi.exe not found in System32 or NVSMI. Is the NVIDIA driver installed?")
    return str(found)


def browser():
    return next((str(p) for p in BROWSERS if p.is_file()), None)
