"""Read-only Windows system facts, gathered with one PowerShell call.

Nothing here writes to the system or needs elevation. Every field degrades to
None when Windows won't tell us, and the checks treat None as "unknown".
"""

import json
import re
import subprocess

from . import sysbin

PS_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
function Reg($path, $name) { (Get-ItemProperty -Path $path -Name $name).$name }
$gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object {
    [pscustomobject]@{ name = $_.Name; pnp = $_.PNPDeviceID; driver = $_.DriverVersion } })
$board = Get-CimInstance Win32_BaseBoard | Select-Object -First 1
$bios = Get-CimInstance Win32_BIOS | Select-Object -First 1
$encl = Get-CimInstance Win32_SystemEnclosure | Select-Object -First 1
$bat = @(Get-CimInstance Win32_Battery).Count
$bl = $null
try { $bl = (New-Object -ComObject Shell.Application).NameSpace($env:SystemDrive + '\').Self.ExtendedProperty('System.Volume.BitLockerProtection') } catch {}
[pscustomobject]@{
    gpus = $gpus
    board_vendor = $board.Manufacturer
    board_model = $board.Product
    bios_vendor = $bios.Manufacturer
    bios_version = $bios.SMBIOSBIOSVersion
    chassis = @($encl.ChassisTypes)
    batteries = $bat
    secure_boot = Reg 'HKLM:\SYSTEM\CurrentControlSet\Control\SecureBoot\State' 'UEFISecureBootEnabled'
    firmware_type = Reg 'HKLM:\SYSTEM\CurrentControlSet\Control' 'PEFirmwareType'
    fast_startup = Reg 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power' 'HiberbootEnabled'
    bitlocker = $bl
    os = (Get-CimInstance Win32_OperatingSystem).Caption
    os_build = (Get-CimInstance Win32_OperatingSystem).BuildNumber
} | ConvertTo-Json -Depth 4 -Compress
"""

PNP_RE = re.compile(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})(?:&SUBSYS_([0-9A-F]{4})([0-9A-F]{4}))?(?:&REV_([0-9A-F]{2}))?", re.I)
LAPTOP_CHASSIS = {8, 9, 10, 11, 14, 30, 31, 32}


def powershell(script):
    out = subprocess.run([sysbin.POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    return out.stdout.strip()


def parse_pnp(pnp):
    """PCI\\VEN_10DE&DEV_2803&SUBSYS_88FB1043&REV_A1 -> ids. SUBSYS is
    <subsystem device><subsystem vendor>."""
    m = PNP_RE.search(pnp or "")
    if not m:
        return None
    ven, dev, sdev, sven, rev = m.groups()
    return {"vendor": ven.lower(), "device": dev.lower(),
            "subsystem": f"{sven.lower()}:{sdev.lower()}" if sven else None,
            "rev": rev.lower() if rev else None}


def nvidia_smi():
    """Per-GPU facts only nvidia-smi knows (VBIOS, power limits, bus id)."""
    fields = "name,pci.bus_id,pci.device_id,vbios_version,driver_version,power.limit,power.max_limit,memory.total"
    try:
        out = subprocess.run([sysbin.nvidia_smi(), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 8:
            continue
        name, bus, pci_id, vbios, driver, pl, plmax, mem = parts
        rows.append({"name": name, "bus_id": bus, "device": pci_id[2:6].lower() if pci_id.startswith("0x") else None,
                     "vbios": vbios, "driver": driver, "power_limit_w": pl, "power_max_w": plmax, "memory_mib": mem})
    return rows


def bitlocker_state(value):
    # System.Volume.BitLockerProtection: 1 on, 2 off, 3 encrypting,
    # 4 decrypting, 5 suspended, 6 locked; anything else unknown.
    return {1: "on", 2: "off", 3: "encrypting", 4: "decrypting", 5: "suspended", 6: "on"}.get(value)


def gather():
    raw = powershell(PS_SCRIPT)
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        info = {}
    gpus = info.get("gpus") or []
    if isinstance(gpus, dict):
        gpus = [gpus]
    smi = nvidia_smi()
    nv = []
    for g in gpus:
        ids = parse_pnp(g.get("pnp"))
        if not ids or ids["vendor"] != "10de":
            continue
        extra = next((s for s in smi if s["device"] == ids["device"]), {})
        nv.append({**ids, "name": g.get("name"), "driver_wddm": g.get("driver"), **{k: v for k, v in extra.items() if k not in ("name", "device")}})
    chassis = set(info.get("chassis") or [])
    return {
        "nvidia_gpus": nv,
        "other_gpus": [g.get("name") for g in gpus if (parse_pnp(g.get("pnp")) or {}).get("vendor") != "10de"],
        "board": " ".join(x for x in (info.get("board_vendor"), info.get("board_model")) if x) or None,
        "bios": " ".join(x for x in (info.get("bios_vendor"), info.get("bios_version")) if x) or None,
        "laptop": bool(chassis & LAPTOP_CHASSIS) or (info.get("batteries") or 0) > 0,
        "secure_boot": None if info.get("secure_boot") is None else bool(info["secure_boot"]),
        "uefi": None if info.get("firmware_type") is None else info["firmware_type"] == 2,
        "fast_startup": None if info.get("fast_startup") is None else bool(info["fast_startup"]),
        "bitlocker": bitlocker_state(info.get("bitlocker")),
        "os": f"{info.get('os') or 'Windows'} build {info.get('os_build') or '?'}",
    }


def last_boot():
    """Windows' last boot as a Unix timestamp, or None."""
    out = powershell("[DateTimeOffset]::new((Get-CimInstance Win32_OperatingSystem).LastBootUpTime).ToUnixTimeSeconds()")
    try:
        return float(out)
    except ValueError:
        return None
