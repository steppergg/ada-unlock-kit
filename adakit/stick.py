"""`adakit stick` / `adakit mode`: lay out a USB stick for nv-uefi-unlocker.

The kit never formats a disk and never touches a drive that isn't a USB disk
with a FAT32 volume. Upstream's make-usbroot.ps1 produces the assets (from
NVIDIA's pinned public packages); this copies only the files the loader opens
(loader/src/main.rs read_boot_file_fmt), puts BOOTX64.EFI where firmware boots
it, verifies every copy by SHA-256 and writes a manifest.

A new stick starts in replay mode: the unlock payload is stored as
<name>.off, so the loader runs the signed firmware transaction and resets
without writing the unlock. `adakit mode F: unlock` renames it back.
"""

import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

FAMILY = "ad102"  # every Ada card loads the AD102-family files
PAYLOAD = f"v67-{FAMILY}-four-plm.bin"
PAYLOAD_SHA256 = "44c36b2d6b3424e53e18f3e55db85c833655de071ce66ce42749588a4a7dc8cb"  # assets/src/manifest.rs
PAYLOAD_BYTES = 0xFA00
ROOT_FILES = [
    f"booter_load_{FAMILY}.bin", f"gsp_boot_desc_{FAMILY}.bin", f"gsp_boot_{FAMILY}.bin",
    f"fwimage_{FAMILY}.bin", f"fwsignature_{FAMILY}.bin", "scrubber_ad10x.bin", PAYLOAD,
]
EFI_REL = "EFI/BOOT/BOOTX64.EFI"
ARMED = "i_know_what_i_am_doing.txt"
MANIFEST = "ADAKIT-MANIFEST.json"


class StickError(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drive_info(letter):
    """Facts about a drive letter from CIM: filesystem, size, and the bus of
    the physical disk behind it (works even when the Storage module is off)."""
    script = f"""
$ErrorActionPreference = 'SilentlyContinue'
$ld = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='{letter}'"
$part = $ld | Get-CimAssociatedInstance -ResultClassName Win32_DiskPartition
$disk = $part | Get-CimAssociatedInstance -ResultClassName Win32_DiskDrive | Select-Object -First 1
[pscustomobject]@{{
  exists = [bool]$ld; fs = $ld.FileSystem; label = $ld.VolumeName; size = $ld.Size; free = $ld.FreeSpace
  drive_type = $ld.DriveType; bus = $disk.InterfaceType; pnp = $disk.PNPDeviceID; media = $disk.MediaType; model = $disk.Model
  system = ($env:SystemDrive -eq '{letter}')
}} | ConvertTo-Json -Compress
"""
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"exists": False}


def check_drive(info, letter):
    """Raise StickError unless this is a FAT32 volume on a USB disk."""
    if not info.get("exists"):
        raise StickError(f"{letter} doesn't exist.")
    if info.get("system"):
        raise StickError(f"{letter} is the Windows system drive.")
    if (info.get("bus") or "").upper() != "USB" and "USB" not in (info.get("pnp") or "").upper():
        raise StickError(f"{letter} isn't on a USB disk (bus: {info.get('bus') or 'unknown'}). The kit only writes to USB sticks.")
    if (info.get("fs") or "").upper() != "FAT32":
        raise StickError(f"{letter} is {info.get('fs') or 'unformatted'}, not FAT32. Firmware can only boot FAT32. "
                         "Format a partition as FAT32 yourself (128 MB or more) and run this again.")


def normalize(letter):
    letter = letter.strip().rstrip("\\/").upper()
    if len(letter) == 1:
        letter += ":"
    if len(letter) != 2 or letter[1] != ":" or not letter[0].isalpha():
        raise StickError(f"'{letter}' isn't a drive letter like F:")
    return letter


def check_source(src):
    """Every needed file present, payload exact. Returns {stick path: source}."""
    src = Path(src)
    plan = {}
    for name in ROOT_FILES:
        p = src / name
        if not p.is_file():
            raise StickError(f"{p} is missing. Run upstream's make-usbroot.ps1 first (it writes .\\usbroot).")
        plan[name] = p
    efi = next((p for p in (src / "BOOTX64.EFI", src / EFI_REL) if p.is_file()), None)
    if not efi:
        raise StickError(f"No BOOTX64.EFI in {src}. Build it from source (cargo efi-tx) or take it from a release you checked.")
    plan[EFI_REL] = efi
    pay = src / PAYLOAD
    if pay.stat().st_size != PAYLOAD_BYTES or sha256(pay) != PAYLOAD_SHA256:
        raise StickError(f"{pay} doesn't match the pinned payload (SHA-256 {PAYLOAD_SHA256[:12]}…). Regenerate it.")
    return plan


def stick_mode(root):
    root = Path(root)
    if (root / PAYLOAD).exists():
        mode = "unlock"
    elif (root / (PAYLOAD + ".off")).exists():
        mode = "replay"
    else:
        mode = "replay (no payload on stick)"
    return mode, (root / ARMED).exists()


def build(src, letter, unlock=False, root_override=None):
    """Copy, verify and record. root_override is for tests (a temp dir)."""
    plan = check_source(src)
    root = Path(root_override) if root_override else Path(letter + "\\")
    need = sum(p.stat().st_size for p in plan.values())
    have = shutil.disk_usage(root).free + sum((root / r).stat().st_size for r in plan if (root / r).exists())
    if have < need + (1 << 20):
        raise StickError(f"{letter} needs {need // 2**20 + 1} MiB free for the unlock files and has {have // 2**20} MiB. "
                         "Use a FAT32 partition of at least 128 MB.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    efi_dst = root / EFI_REL
    if efi_dst.exists():
        backup = efi_dst.with_name(f"BOOTX64.EFI.bak-{stamp}")
        shutil.copy2(efi_dst, backup)
    files = {}
    for rel, src_path in plan.items():
        dst = root / rel
        if rel == PAYLOAD:
            stale = root / (PAYLOAD if not unlock else PAYLOAD + ".off")
            stale.unlink(missing_ok=True)
            if not unlock:
                dst = root / (PAYLOAD + ".off")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst)
        want, got = sha256(src_path), sha256(dst)
        if want != got:
            raise StickError(f"Copy of {rel} doesn't match its source (stick may be failing).")
        files[dst.relative_to(root).as_posix()] = {"sha256": got, "bytes": dst.stat().st_size}
    manifest = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(Path(src).resolve()),
                "loader_sha256": files[EFI_REL]["sha256"], "files": files}
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest, stick_mode(root)


def set_mode(root, mode):
    root = Path(root)
    on, off = root / PAYLOAD, root / (PAYLOAD + ".off")
    if mode == "unlock":
        cand = on if on.exists() else off
        if not cand.exists():
            raise StickError("No payload on the stick. Rebuild it with `adakit stick`.")
        if cand.stat().st_size != PAYLOAD_BYTES or sha256(cand) != PAYLOAD_SHA256:
            raise StickError("The payload on the stick doesn't match the pinned hash. Nothing changed; rebuild the stick.")
        if cand == off:
            off.replace(on)
        else:
            off.unlink(missing_ok=True)
    elif mode == "replay":
        if on.exists():
            on.replace(off)
    elif mode == "disarm":
        armed = root / ARMED
        if armed.exists():
            armed.replace(root / (ARMED + ".off"))
    return stick_mode(root)


def first_boot_entry(bcd_text):
    """Description of the first firmware boot entry in `bcdedit /enum
    firmware` output, or None if it can't be read."""
    m = re.search(r"displayorder\s+(\{[^}]+\})", bcd_text)
    if not m:
        return None
    first = m.group(1)
    for block in re.split(r"\r?\n\s*\r?\n", bcd_text):
        if re.search(r"identifier\s+" + re.escape(first), block):
            d = re.search(r"description\s+(.+)", block)
            return d.group(1).strip() if d else first
    return first


def boot_order_warning():
    """Warn when the firmware will boot Windows before the stick. Needs
    bcdedit, which only answers from an elevated prompt; silent otherwise."""
    try:
        out = subprocess.run(["bcdedit", "/enum", "firmware"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = first_boot_entry(out)
    if first and "windows boot manager" in first.lower():
        return ("Windows Boot Manager is first in the boot order, so the stick won't run on its own. "
                "Pick the stick from the firmware boot menu (F8 on ASUS, F11 on MSI), or set it as Boot Option #1 again "
                "(boards often drop it after the stick is unplugged).")
    return None


def check_unmanaged(root):
    """A stick laid out by hand: no manifest to compare against, so check
    what can be checked without one."""
    problems = []
    for name in ROOT_FILES:
        if name == PAYLOAD:
            continue
        if not (root / name).exists():
            problems.append(f"missing {name}")
    if not (root / EFI_REL).exists():
        problems.append(f"missing {EFI_REL}")
    pay = next((p for p in (root / PAYLOAD, root / (PAYLOAD + ".off")) if p.exists()), None)
    if pay and (pay.stat().st_size != PAYLOAD_BYTES or sha256(pay) != PAYLOAD_SHA256):
        problems.append(f"{pay.name} doesn't match the pinned payload hash")
    return problems


def verify(root):
    """Re-hash the stick against its manifest. Returns list of problems."""
    root = Path(root)
    mf = root / MANIFEST
    if not mf.exists():
        return check_unmanaged(root)
    manifest = json.loads(mf.read_text())
    problems = []
    for rel, meta in manifest["files"].items():
        p = root / rel
        if rel.startswith(PAYLOAD) and not p.exists():
            alt = root / (PAYLOAD if rel.endswith(".off") else PAYLOAD + ".off")
            p = alt if alt.exists() else p
        if not p.exists():
            problems.append(f"missing {rel}")
        elif sha256(p) != meta["sha256"]:
            problems.append(f"changed {rel}")
    return problems


def find_sticks():
    """Drive letters of FAT32 volumes that hold a loader: candidates for
    `adakit status` when no drive is given."""
    script = ("Get-CimInstance Win32_LogicalDisk | Where-Object { $_.FileSystem -eq 'FAT32' } | "
              "ForEach-Object { $_.DeviceID }")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True)
    return [d.strip() for d in out.stdout.split() if d.strip()
            and (Path(d.strip() + "\\") / EFI_REL).exists()
            and any((Path(d.strip() + "\\") / n).exists() for n in (PAYLOAD, PAYLOAD + ".off", "loader.log"))]


def last_run(root):
    """What the loader logged on its last run, from loader.log on the stick."""
    log = Path(root) / "loader.log"
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    ver = re.search(r"nv-uefi-unlocker (v[\w.+-]+)(?: \(built ([^)]+)\))?", text)
    hand = re.search(r"HANDOFF-OUT: ss0=(0x[0-9a-f]+) ss1=(0x[0-9a-f]+)", text)
    return {
        "time": log.stat().st_mtime,
        "version": ver.group(1) if ver else None,
        "built": ver.group(2) if ver and ver.group(2) else None,
        "unlocked": None if not hand else (hand.group(1) == "0x88888888" and hand.group(2) == "0x8"),
        "chained": "CHAIN: LoadImage" in text,
        "aborted": bool(re.search(r"CONFIRM: .* - ABORT", text)),
        "errors": [l.strip() for l in text.splitlines() if "[ERROR]" in l][-3:],
    }


def came_from_stick(log_time, boot_time, window_s=300):
    """True when Windows started within window_s after the loader wrote its
    log, i.e. this session was chain-loaded by the stick."""
    if log_time is None or boot_time is None:
        return None
    return 0 <= boot_time - log_time <= window_s
