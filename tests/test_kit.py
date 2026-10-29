import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adakit import bench, check, compat, report, stick, winsys


class Pnp(unittest.TestCase):
    def test_subsystem_order(self):
        ids = winsys.parse_pnp(r"PCI\VEN_10DE&DEV_2803&SUBSYS_88FB1043&REV_A1\4&1234")
        self.assertEqual(ids, {"vendor": "10de", "device": "2803", "subsystem": "1043:88fb", "rev": "a1"})

    def test_garbage(self):
        self.assertIsNone(winsys.parse_pnp("ROOT\\BasicDisplay\\0000"))


class Compat(unittest.TestCase):
    def test_known(self):
        e = compat.lookup("2803")
        self.assertTrue(e["known"])
        self.assertEqual(e["windows"], "working")

    def test_unknown_ada_is_untested_family(self):
        e = compat.lookup("27ff")
        self.assertFalse(e["known"])
        self.assertEqual((e["chip"], e["windows"]), ("Ada", "unknown"))

    def test_unknown_blackwell_is_no(self):
        self.assertEqual(compat.lookup("2b99")["windows"], "no")

    def test_every_status_has_advice_and_refs_resolve(self):
        for dev, e in compat.DATA["devices"].items():
            self.assertIn(e["windows"], check.WINDOWS_ADVICE, dev)
            for ref in e["refs"]:
                self.assertTrue(compat.ref_url(ref).startswith("https://github.com/"), ref)


def facts(**over):
    base = {"nvidia_gpus": [{"device": "2803", "subsystem": "1043:88fb", "name": "NVIDIA GeForce RTX 4060 Ti"}],
            "other_gpus": [], "board": "ASUSTeK ROG MAXIMUS Z790 APEX", "bios": "AMI 2801", "laptop": False,
            "secure_boot": False, "uefi": True, "fast_startup": False, "bitlocker": "off", "os": "Windows 11"}
    base.update(over)
    return base


class Assess(unittest.TestCase):
    def levels(self, f):
        return {lvl for lvl, _ in check.assess(f)[1]}

    def test_clean_desktop_has_no_warnings(self):
        gpus, sysf = check.assess(facts())
        self.assertEqual(gpus[0]["level"], check.OK)
        self.assertNotIn(check.WARN, self.levels(facts()))

    def test_legacy_bios_stops(self):
        self.assertIn(check.STOP, self.levels(facts(uefi=False)))

    def test_warnings(self):
        f = facts(secure_boot=True, bitlocker="on", fast_startup=True, laptop=True)
        texts = " ".join(t for _, t in check.assess(f)[1])
        for word in ("Secure Boot is on", "BitLocker", "Fast Startup", "laptop"):
            self.assertIn(word, texts)

    def test_no_gpu_stops(self):
        self.assertIn(check.STOP, self.levels(facts(nvidia_gpus=[])))

    def test_pro_card_not_needed(self):
        gpus, _ = check.assess(facts(nvidia_gpus=[{"device": "26b1", "subsystem": None, "name": "RTX 6000 Ada"}]))
        self.assertEqual(gpus[0]["level"], check.STOP)


class Verdict(unittest.TestCase):
    # 4060 Ti 8 GB, 34 SMs, clocks locked at 2100 MHz (bench/result-*-lgc2100.json)
    def eff(self, tflops):
        return tflops / (34 * 1024 * 2100e6 / 1e12)

    def test_stock_and_unlocked_measurements(self):
        self.assertEqual(bench.verdict(self.eff(35.96))[0], "stock")      # 0.49
        self.assertEqual(bench.verdict(self.eff(70.72))[0], "unlocked")   # 0.97

    def test_middle_is_unclear(self):
        self.assertEqual(bench.verdict(0.65)[0], "unclear")

    def test_unknown_generation(self):
        self.assertEqual(bench.verdict(None)[0], "unknown")


class Stick(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.src, self.dst = base / "usbroot", base / "stick"
        self.src.mkdir(); self.dst.mkdir()
        for name in stick.ROOT_FILES:
            (self.src / name).write_bytes(name.encode() * 10)
        (self.src / "BOOTX64.EFI").write_bytes(b"MZ-loader")
        (self.src / "gsp_ga10x.bin").write_bytes(b"intermediate, must not be copied")
        payload = self.src / stick.PAYLOAD
        payload.write_bytes(b"\x00" * stick.PAYLOAD_BYTES)
        self.pin = mock.patch.object(stick, "PAYLOAD_SHA256", hashlib.sha256(payload.read_bytes()).hexdigest())
        self.pin.start()

    def tearDown(self):
        self.pin.stop()
        self.tmp.cleanup()

    def test_build_defaults_to_replay_and_copies_only_needed_files(self):
        manifest, (mode, armed) = stick.build(self.src, "F:", root_override=self.dst)
        self.assertEqual((mode, armed), ("replay", False))
        self.assertTrue((self.dst / "EFI/BOOT/BOOTX64.EFI").exists())
        self.assertFalse((self.dst / stick.PAYLOAD).exists())
        self.assertFalse((self.dst / "gsp_ga10x.bin").exists())
        self.assertEqual(stick.verify(self.dst), [])

    def test_mode_toggle_and_disarm(self):
        stick.build(self.src, "F:", root_override=self.dst)
        self.assertEqual(stick.set_mode(self.dst, "unlock")[0], "unlock")
        self.assertEqual(stick.verify(self.dst), [])
        (self.dst / stick.ARMED).write_text("")
        self.assertTrue(stick.stick_mode(self.dst)[1])
        self.assertFalse(stick.set_mode(self.dst, "disarm")[1])
        self.assertEqual(stick.set_mode(self.dst, "replay")[0], "replay")

    def test_rebuild_backs_up_loader(self):
        stick.build(self.src, "F:", root_override=self.dst)
        stick.build(self.src, "F:", root_override=self.dst)
        self.assertEqual(len(list((self.dst / "EFI/BOOT").glob("BOOTX64.EFI.bak-*"))), 1)

    def test_wrong_payload_refused(self):
        (self.src / stick.PAYLOAD).write_bytes(b"\x01" * stick.PAYLOAD_BYTES)
        with self.assertRaises(stick.StickError):
            stick.check_source(self.src)

    def test_tampered_stick_detected(self):
        stick.build(self.src, "F:", root_override=self.dst)
        (self.dst / "scrubber_ad10x.bin").write_bytes(b"changed")
        self.assertIn("changed scrubber_ad10x.bin", stick.verify(self.dst))

    def test_drive_rules(self):
        good = {"exists": True, "system": False, "bus": "USB", "fs": "FAT32"}
        stick.check_drive(good, "F:")
        for bad in ({"exists": False}, dict(good, system=True), dict(good, bus="SCSI"), dict(good, fs="NTFS")):
            with self.assertRaises(stick.StickError):
                stick.check_drive(bad, "F:")
        with self.assertRaises(stick.StickError):
            stick.normalize("C:\\Windows")
        self.assertEqual(stick.normalize("f"), "F:")


def result(label, fp16, fp32=12.3, driver="581.42", pl="216.00"):
    rows = [{"case": c, "n": n, "pass": 1, "median": v, "min": v, "max": v, "clock_mhz": 2100.0,
             "power_w": "150", "power_capped": False, "tflops_per_ghz": round(v / 2.1, 2)}
            for c, n, v in (("FP16", 8192, fp16), ("BF16", 8192, fp16), ("FP8 e4m3", 8192, 2 * fp16),
                            ("TF32", 4096, fp16 / 2), ("FP32 (control)", 4096, fp32))]
    return {"label": label, "device": "NVIDIA GeForce RTX 4060 Ti", "driver": driver, "power_limit_w": pl,
            "torch": "2.11.0+cu128", "rows": rows, "max_rel_error": {"fp16": 2.8e-4}}


class Report(unittest.TestCase):
    def test_matched_report(self):
        with tempfile.TemporaryDirectory() as d:
            s, u = Path(d, "result-stock.json"), Path(d, "result-unlocked.json")
            s.write_text(json.dumps(result("stock", 48.06)))
            u.write_text(json.dumps(result("unlocked", 81.94)))
            with mock.patch.object(report, "render_png", return_value=False):
                rows, notes, _ = report.build(s, u, Path(d, "out"), facts(), "v2.1.5")
            self.assertEqual(notes, [])
            self.assertAlmostEqual(rows[0]["gain"], 81.94 / 48.06 - 1)
            e = json.loads(Path(d, "out", "entry.json").read_text())
            self.assertEqual((e["fp16_gain_pct"], e["result"]), (70, "unlocked"))
            self.assertIn("| FP16 8192³ | 48.06 | 81.94 | +70% |", Path(d, "out", "report.md").read_text(encoding="utf-8"))

    def test_unfair_comparison_flagged(self):
        sr = result("stock", 46.7, fp32=12.0, pl="160.00")
        ur = result("unlocked", 81.0, fp32=12.8)
        sb = {(r["case"], r["n"]): r for r in sr["rows"]}
        ub = {(r["case"], r["n"]): r for r in ur["rows"]}
        notes = report.fairness(sr, ur, sb, ub)
        self.assertTrue(any("Power limit" in n for n in notes))
        self.assertTrue(any("FP32 control" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
