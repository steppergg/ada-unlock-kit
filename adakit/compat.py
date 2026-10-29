"""Known-results table: what the issue trackers say about each device ID."""

import json
from pathlib import Path

DATA = json.loads((Path(__file__).with_name("compat.json")).read_text(encoding="utf-8"))

# Rough family ranges for IDs the table doesn't list. Only used to tell the
# user which family an unknown card belongs to, never to admit it.
FAMILIES = [
    (0x1E00, 0x21FF, "Turing", "Experimental Turing path; needs GSP firmware mode on Windows."),
    (0x2200, 0x25FF, "Ampere", "Ampere family; the EFI path has few silicon reports."),
    (0x2600, 0x28FF, "Ada", "Ada family; uses the AD102-family payload."),
    (0x2900, 0x2FFF, "Blackwell", "Blackwell: not possible with this method."),
]


def ref_url(ref):
    repo, num = ref.split("#")
    return DATA["sources"][repo] + num


def family(dev_id):
    n = int(dev_id, 16)
    for lo, hi, name, note in FAMILIES:
        if lo <= n <= hi:
            return name, note
    return None, "Not a GPU family this project covers."


def lookup(dev_id):
    """Return the table entry for a 4-hex-digit device ID, or a synthesized
    'unknown' entry with the family guess."""
    dev_id = dev_id.lower()
    entry = DATA["devices"].get(dev_id)
    if entry:
        return dict(entry, id=dev_id, known=True)
    fam, note = family(dev_id)
    status = "no" if fam == "Blackwell" else "unknown"
    return {"id": dev_id, "known": False, "name": f"Unknown NVIDIA device {dev_id}", "chip": fam or "?",
            "form": "?", "linux": status, "windows": status, "notes": note, "refs": []}
