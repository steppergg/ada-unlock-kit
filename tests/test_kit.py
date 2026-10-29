import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adakit import bench, check, compat, report, stick, winsys


class Security(unittest.TestCase):
    def test_no_bare_program_names(self):
        # Bare names are searched in the current folder first on Windows.
        src = "".join(Path(f).read_text(encoding="utf-8") for f in Path("adakit").glob("*.py"))
        for bare in ('["powershell.exe"', '["nvidia-smi"', '["bcdedit"', "shutil.which("):
            self.assertNotIn(bare, src)
        from adakit import sysbin
        for exe in (sysbin.POWERSHELL, sysbin.BCDEDIT):
            self.assertTrue(Path(exe).is_absolute(), exe)

    def test_bench_label_is_a_plain_name(self):
        for bad in ("../../x", "a\\b", "", ".."):
            with self.assertRaises(SystemExit):
                bench.bench(bad, "results")

    def test_tracker_escapes_script_breakouts(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("tbuild", "tracker/build.py")
        tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
        with tempfile.TemporaryDirectory() as d:
            here = Path(d)
            (here / "template.html").write_text("<script id=data>/*ENTRIES*/</script>", encoding="utf-8")
            row = {k: None for k in tb.FIELDS}
            row.update(card="</script><!--<img src=x onerror=alert(1)>", refs=[], laptop=False, generic=True)
            (here / "entries.json").write_text(json.dumps({"updated": "x", "entries": [row]}), encoding="utf-8")
            with mock.patch.object(tb, "HERE", here), mock.patch.object(tb, "ENTRIES", here / "entries.json"), \
                    mock.patch("sys.argv", ["build.py"]):
                tb.main()
            html = (here / "index.html").read_text(encoding="utf-8")
            inner = html[len("<script id=data>"):-len("</script>")]
            self.assertNotIn("<", inner)
            self.assertEqual(json.loads(inner)["entries"][0]["card"], row["card"])


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

    def test_ampere_peak_matches_whitepaper(self):
        # RTX 3070: 81.3 TFLOPS FP16 dense, 40.6 with FP32 accumulate at
        # 46 SMs x 1.725 GHz. A stock 3070 must not read as unlocked.
        row = {"median": 40.6, "clock_mhz": 1725}
        self.assertEqual(bench.state_of(row, 46, (8, 6))[1], "stock")
        self.assertEqual(bench.state_of(dict(row, median=78.0), 46, (8, 6))[1], "unlocked")


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
        self.real_assets = dict(stick.KNOWN_ASSETS)
        fake = {n: stick.sha256(self.src / n) for n in stick.KNOWN_ASSETS}
        self.assets = mock.patch.dict(stick.KNOWN_ASSETS, fake)
        self.assets.start()

    def tearDown(self):
        self.assets.stop()
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

    def test_bad_payload_never_switched_on(self):
        stick.build(self.src, "F:", root_override=self.dst)
        (self.dst / (stick.PAYLOAD + ".off")).write_bytes(b"\x01" * stick.PAYLOAD_BYTES)
        with self.assertRaises(stick.StickError):
            stick.set_mode(self.dst, "unlock")
        self.assertEqual(stick.stick_mode(self.dst)[0], "replay")

    def test_unlock_with_both_files_and_double_disarm(self):
        stick.build(self.src, "F:", root_override=self.dst, unlock=True)
        self.assertFalse((self.dst / (stick.PAYLOAD + ".off")).exists())
        (self.dst / (stick.PAYLOAD + ".off")).write_bytes((self.dst / stick.PAYLOAD).read_bytes())
        self.assertEqual(stick.set_mode(self.dst, "unlock")[0], "unlock")
        self.assertFalse((self.dst / (stick.PAYLOAD + ".off")).exists())
        for _ in range(2):
            (self.dst / stick.ARMED).write_text("")
            self.assertFalse(stick.set_mode(self.dst, "disarm")[1])

    def test_not_enough_space(self):
        with mock.patch.object(stick.shutil, "disk_usage", return_value=mock.Mock(free=1000)):
            with self.assertRaises(stick.StickError):
                stick.build(self.src, "F:", root_override=self.dst)

    def test_uas_usb_ssd_accepted(self):
        stick.check_drive({"exists": True, "system": False, "bus": "SCSI", "pnp": "USBSTOR\\DISK&VEN_SAMSUNG", "fs": "FAT32"}, "F:")

    def test_last_run_and_session(self):
        log = self.dst / "loader.log"
        log.write_text("[INFO ] loader::banner:  nv-uefi-unlocker v2.1.5 (built Wed 2026-10-28 15:37:01 -04:00)\n"
                       "[INFO ] loader: CONFIRM: \\i_know_what_i_am_doing.txt present - ARMED: 5s countdown auto-runs, ANY key aborts\n"
                       "[INFO ] loader::tx: HANDOFF-OUT: ss0=0x88888888 ss1=0x8 readout1=0x00000000\n"
                       "[INFO ] loader::chainload: CHAIN: LoadImage (FromDevicePath)\n")
        r = stick.last_run(self.dst)
        self.assertEqual((r["version"], r["unlocked"], r["chained"], r["aborted"]), ("v2.1.5", True, True, False))
        log.write_text("CONFIRM: 30s timeout - ABORT to Windows, NOTHING was unlocked this boot (zero GPU writes)\n")
        self.assertTrue(stick.last_run(self.dst)["aborted"])
        self.assertTrue(stick.came_from_stick(1000, 1012))
        self.assertFalse(stick.came_from_stick(1000, 1000 + 7 * 60))
        self.assertFalse(stick.came_from_stick(1000, 900))
        self.assertIsNone(stick.came_from_stick(None, 1000))

    def test_refusal_reasons_surface(self):
        (self.dst / "loader.log").write_text(
            "[INFO ] loader: CONFIRM: \\i_know_what_i_am_doing.txt present - ARMED: 5s countdown auto-runs, ANY key aborts\n"
            "[INFO ] loader::tx_fwsec: layout: MMU_PLM PLM-locked on Turing (0xbadf5040) - skipping the lock-bit sanity check\n"
            "[INFO ] loader: TX: assets missing for family 'ga102' (booter_load_*/gsp_boot_*/fwimage_*/fwsignature_*) - skipping\n")
        r = stick.last_run(self.dst)
        self.assertEqual(len(r["reasons"]), 1)
        self.assertIn("assets missing for family 'ga102'", r["reasons"][0])
        self.assertIsNone(r["unlocked"])

    def test_gpu_family_check(self):
        ada = {"device": "2803", "name": "RTX 4060 Ti"}
        ampere = {"device": "2484", "name": "RTX 3070"}
        self.assertIsNone(stick.gpu_family_problem([ada]))
        self.assertIsNone(stick.gpu_family_problem([ampere, ada]))
        self.assertIsNone(stick.gpu_family_problem([]))  # no driver / other PC
        self.assertIn("10de:2484", stick.gpu_family_problem([ampere]))

    def test_manifest_cannot_point_outside_stick(self):
        stick.build(self.src, "F:", root_override=self.dst)
        mf = self.dst / stick.MANIFEST
        m = json.loads(mf.read_text())
        m["files"]["../../outside.bin"] = {"sha256": "0" * 64, "bytes": 1}
        mf.write_text(json.dumps(m))
        self.assertTrue(any("unexpected path" in p for p in stick.verify(self.dst)))
        self.assertNotIn(":", m["source"])  # folder name only, no local path

    def test_asset_pins(self):
        self.assertEqual(len(self.real_assets), 6)
        self.assertTrue(all(len(h) == 64 for h in self.real_assets.values()))
        files = {n: self.src / n for n in stick.KNOWN_ASSETS}
        self.assertEqual(stick.asset_warnings(files), [])
        (self.src / "scrubber_ad10x.bin").write_bytes(b"tampered")
        self.assertEqual(len(stick.asset_warnings(files)), 1)

    def test_ascii_drive_letters_only(self):
        with self.assertRaises(stick.StickError):
            stick.normalize("é:")

    def test_first_boot_entry(self):
        bcd = ("Firmware Boot Manager\n---------------------\nidentifier              {fwbootmgr}\n"
               "displayorder            {bootmgr}\n                        {f8d10c35}\ntimeout                 1\n\n"
               "Windows Boot Manager\n--------------------\nidentifier              {bootmgr}\n"
               "description             Windows Boot Manager\n\n"
               "Firmware Application (101fffff)\n-------------------------------\nidentifier              {f8d10c35}\n"
               "description             UEFI: SanDisk, Partition 2\n")
        self.assertEqual(stick.first_boot_entry(bcd), "Windows Boot Manager")
        self.assertEqual(stick.first_boot_entry(bcd.replace("displayorder            {bootmgr}\n                        {f8d10c35}",
                                                            "displayorder            {f8d10c35}\n                        {bootmgr}")),
                         "UEFI: SanDisk, Partition 2")
        self.assertIsNone(stick.first_boot_entry("Access is denied."))

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

    def test_no_shared_cases_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            s, u = Path(d, "result-stock.json"), Path(d, "result-unlocked.json")
            s.write_text(json.dumps(dict(result("stock", 48.0), rows=[])))
            u.write_text(json.dumps(result("unlocked", 81.0)))
            with self.assertRaises(SystemExit):
                report.build(s, u, Path(d, "out"))

    def test_mislabelled_run_flagged(self):
        sr, ur = result("stock", 81.0), result("unlocked", 81.0)
        sr.update(state="unlocked", efficiency=0.83)
        ur.update(state="unlocked", efficiency=0.83)
        sb = {(r["case"], r["n"]): r for r in sr["rows"]}
        notes = report.fairness(sr, ur, sb, sb)
        self.assertTrue(any("labelled stock measured as unlocked" in n for n in notes))

    def test_pick_gpu_and_state(self):
        f = facts(nvidia_gpus=[{"device": "2b85", "name": "NVIDIA GeForce RTX 5090"},
                               {"device": "2803", "name": "NVIDIA GeForce RTX 4060 Ti"}])
        self.assertEqual(report.pick_gpu(f, "NVIDIA GeForce RTX 4060 Ti")["device"], "2803")
        row = {"median": 76.45, "clock_mhz": 2340.0}
        eff, state = bench.state_of(row, 34, (8, 9))
        self.assertEqual(state, "unlocked")
        self.assertEqual(bench.state_of({"median": 46.72, "clock_mhz": 2775.0}, 34, (8, 9))[1], "stock")
        self.assertEqual(bench.state_of(row, 34, (7, 5)), (None, None))

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
