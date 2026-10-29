"""Build the tracker page from entries.json, optionally adding a kit entry.

  python tracker/build.py                       rebuild tracker/index.html
  python tracker/build.py --add results/report/entry.json --refs ada#82 --notes "..."

--add converts an `adakit report` entry.json into a tracker row (device ID
without the vendor prefix, gains under short names) and appends it.
"""

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
ENTRIES = HERE / "entries.json"
FIELDS = ["card", "device_id", "subsystem", "vbios", "board", "laptop", "tool", "generic",
          "fp16", "bf16", "fp8", "tf32", "result", "notes", "refs", "date"]


def from_kit(e, refs, notes):
    loader = e.get("loader") or "not recorded"
    return {"card": e["card"].replace("NVIDIA GeForce ", ""), "device_id": e["device_id"].split(":")[-1],
            "subsystem": e.get("subsystem"), "vbios": e.get("vbios"), "board": e.get("board"),
            "laptop": bool(e.get("laptop")), "tool": loader, "generic": "nv-uefi-unlocker" in loader,
            "fp16": e.get("fp16_gain_pct"), "bf16": e.get("bf16_gain_pct"), "fp8": e.get("fp8_gain_pct"),
            "tf32": e.get("tf32_gain_pct"), "result": e.get("result"), "notes": notes or "",
            "refs": refs, "date": time.strftime("%Y-%m-%d")}


def validate(data):
    for i, row in enumerate(data["entries"]):
        missing = [f for f in FIELDS if f not in row]
        if missing:
            raise SystemExit(f"entry {i} ({row.get('card')}) is missing {missing}")
        for ref in row["refs"]:
            repo, _, num = ref.partition("#")
            if repo not in ("ada", "nvuefi") or not num.isdigit():
                raise SystemExit(f"entry {i}: bad ref {ref!r} (use ada#82 or nvuefi#13)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--add", help="entry.json written by adakit report")
    p.add_argument("--refs", nargs="*", default=[])
    p.add_argument("--notes", default="")
    a = p.parse_args()
    data = json.loads(ENTRIES.read_text(encoding="utf-8"))
    if a.add:
        data["entries"].append(from_kit(json.loads(Path(a.add).read_text(encoding="utf-8")), a.refs, a.notes))
        data["updated"] = time.strftime("%Y-%m-%d")
        ENTRIES.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    validate(data)
    # Escape every "<" so submitted text can't close the <script> or open a
    # comment inside it; JSON.parse turns < back into "<".
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    html = (HERE / "template.html").read_text(encoding="utf-8").replace("/*ENTRIES*/", payload)
    (HERE / "index.html").write_text(html, encoding="utf-8")
    print(f"{len(data['entries'])} entries -> {HERE / 'index.html'}")


if __name__ == "__main__":
    main()
