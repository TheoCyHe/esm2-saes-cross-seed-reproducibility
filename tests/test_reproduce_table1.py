"""§CT18 (b) and (c): `src/analysis/release_derived.py` and
`scripts/reproduce_table1.py`.

The carrier-excess fixture's answer is WORKED BY HAND and written down (gotcha
32b), and it is worked in two pair orders because the order changes the number:
the release has to carry the order, and the reproduction has to use it.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.release_derived import (fmt, indicator_from_sim,
                                          partner_surrogate, write_csv_gz)

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "reproduce_table1", REPO / "scripts" / "reproduce_table1.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


# ---------------------------------------------------------------------------
# the carrier excess, by hand
# ---------------------------------------------------------------------------

def _hand_fixture():
    carrier = {1: np.array([0, 3]), 0: np.array([1, 2])}
    detected = {1: np.array([True, True]), 0: np.array([True, False])}
    has = {(1, 0): np.array([True, False, True, False]),
           (0, 1): np.array([True, False, False, True])}
    fbin = {1: np.array([0, 0, 0, 1]), 0: np.array([0, 0, 1, 1])}
    return carrier, detected, has, fbin


def test_carrier_excess_matches_the_hand_worked_value():
    """Order [1, 0]: concept 0 only (seed 0 does not detect concept 1). Seed 1's
    carrier, feature 0, has a partner: A = 1. Its bin holds features 0-2, of
    which 0 and 2 have partners; excluding the carrier, 1 of 2: matched 0.5.
    Excess (1 - 0.5) x 100 = +50.0."""
    ex = rt.carrier_excess([1, 0], *_hand_fixture())
    assert ex["A"] == 1.0 and ex["matched_ambient"] == 0.5
    assert ex["excess_pp"] == 50.0 and ex["n_cells"] == 1


def test_the_pair_order_changes_the_number():
    """Order [0, 1]: seed 0's carrier, feature 1, has no partner in seed 1: A = 0.
    Its bin holds features 0-1; excluding the carrier, 1 of 1 has a partner:
    matched 1.0. Excess -100.0 -- so the order the release carries matters."""
    ex = rt.carrier_excess([0, 1], *_hand_fixture())
    assert ex["excess_pp"] == -100.0


def test_carrier_excess_agrees_with_the_record_implementation():
    """The plain-numpy excess equals `carrier_rates_by_cell`'s on a random
    fixture, through the surrogate partner array the exporter builds."""
    from src.analysis.carrier_support import carrier_rates_by_cell
    rng = np.random.default_rng(3)
    order, n_f, n_c = [2, 0, 3, 1], 60, 9
    anchors, cols = ["0", "1", "2", "3"], [0, 1, 2, 3, 4]
    sim = rng.uniform(0.4, 1.0, (4, 5, n_f)).astype(np.float32)
    ind = indicator_from_sim(sim, anchors, cols, 0.70)
    partner = partner_surrogate(ind, order, anchors, cols)
    carrier = rng.integers(0, n_f, (4, n_c))
    detected = rng.random((4, n_c)) < 0.8
    fb = rng.integers(0, 4, (4, n_f))
    ref = carrier_rates_by_cell(carrier, detected, partner, fb, 4)
    a_idx = {s: anchors.index(str(s)) for s in order}
    mine = rt.carrier_excess(
        order, {s: carrier[i] for i, s in enumerate(order)},
        {s: detected[i] for i, s in enumerate(order)},
        {(s, t): ind[a_idx[s], cols.index(t)] for s in order for t in order if s != t},
        {s: fb[i] for i, s in enumerate(order)})
    assert mine["excess_pp"] == ref["excess_A_minus_matched_pp"]
    assert mine["n_cells"] == ref["n_concept_seedpair_cells_evaluated"]


# ---------------------------------------------------------------------------
# the exporter's pure functions
# ---------------------------------------------------------------------------

def test_indicator_excludes_the_own_column_by_label_not_position():
    """Anchor '1' sits in row 0 here and its own column is column 2; a
    positional rule would blank column 0 instead."""
    sim = np.ones((1, 3, 2), dtype=np.float32)
    ind = indicator_from_sim(sim, ["1"], [5, 7, 1], 0.70)
    assert ind[0, :, 0].tolist() == [True, True, False]


def test_partner_surrogate_follows_the_named_order():
    ind = np.zeros((2, 2, 3), dtype=bool)
    ind[0, 1, 2] = True            # anchor "0" has a partner in column 1, feature 2
    p = partner_surrogate(ind, [1, 0], ["0", "1"], [0, 1])
    assert p[1, 0, 2] == 0 and p[1, 0, 0] == -1 and p[0, 1].max() == -1


def test_fmt_round_trips_a_float_exactly():
    x = 0.1 + 0.2
    assert float(fmt(x)) == x and fmt(True) == "1" and fmt(np.int64(7)) == "7"


def test_csv_gz_is_byte_deterministic(tmp_path):
    for name in ("a.csv.gz", "b.csv.gz"):
        write_csv_gz(tmp_path / name, ["x"], [[1], [2]])
    assert (tmp_path / "a.csv.gz").read_bytes() == (tmp_path / "b.csv.gz").read_bytes()
    with gzip.open(tmp_path / "a.csv.gz", "rt", encoding="utf-8") as f:
        assert f.read() == "x\n1\n2\n"


# ---------------------------------------------------------------------------
# the reproduction script's instruments can fail
# ---------------------------------------------------------------------------

def test_check_fails_off_the_record_rounding():
    assert rt.check("headline_mean_rf_0.70_n39", 0.4955134)["verdict"] == "PASS"
    assert rt.check("headline_mean_rf_0.70_n39", 0.4955144)["verdict"] == "FAIL"
    assert rt.check("carrier_excess_single_pp", 7.22186)["verdict"] == "PASS"
    assert rt.check("carrier_excess_single_pp", 7.22196)["verdict"] == "FAIL"


def _mini_bundle(tmp_path: Path) -> Path:
    b = tmp_path / "derived"
    b.mkdir()
    (b / "meta.json").write_text("{}", encoding="utf-8")
    h = hashlib.sha256((b / "meta.json").read_bytes()).hexdigest()
    with open(b / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256", "producing_commit", "owning_nor_row"])
        w.writerow(["release/derived/meta.json", h, "abc", "—"])
    return b


def test_manifest_verification_passes_then_refuses_a_changed_file(tmp_path):
    b = _mini_bundle(tmp_path)
    assert len(rt.verify_manifest(b)) == 1
    (b / "meta.json").write_text("{ }", encoding="utf-8")
    with pytest.raises(rt.BundleError, match="sha256"):
        rt.verify_manifest(b)


def test_manifest_verification_refuses_a_missing_file(tmp_path):
    b = _mini_bundle(tmp_path)
    (b / "meta.json").unlink()
    with pytest.raises(rt.BundleError, match="absent"):
        rt.verify_manifest(b)


def _served(tmp_path: Path) -> tuple[Path, Path]:
    """A 'release' directory to download from, and a manifest naming its files."""
    src = tmp_path / "assets"
    src.mkdir()
    for name, text in (("a.csv", "x,1\n"), ("b.json", "{}")):
        (src / name).write_text(text, encoding="utf-8")
    man = tmp_path / "MANIFEST.csv"
    with open(man, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256", "producing_commit", "owning_nor_row"])
        for name in ("a.csv", "b.json"):
            w.writerow([f"release/derived/{name}",
                        hashlib.sha256((src / name).read_bytes()).hexdigest(), "abc", "-"])
    return src, man


def test_download_fetches_and_verifies_against_the_repository_manifest(tmp_path):
    src, man = _served(tmp_path)
    dst = tmp_path / "bundle"
    got = rt.download(src.as_uri(), dst, man)
    assert sorted(got) == ["a.csv", "b.json"]
    assert (dst / "a.csv").read_bytes() == (src / "a.csv").read_bytes()
    assert rt.download(src.as_uri(), dst, man) == []        # present and verified: not refetched


def test_download_refuses_a_file_whose_hash_differs(tmp_path):
    src, man = _served(tmp_path)
    (src / "b.json").write_text("{ }", encoding="utf-8")    # the served copy is not the manifest's
    with pytest.raises(rt.BundleError, match="sha256"):
        rt.download(src.as_uri(), tmp_path / "bundle", man)
    assert not (tmp_path / "bundle" / "b.json").exists()
    assert not (tmp_path / "bundle" / "b.json.part").exists()


def test_download_fails_on_an_absent_asset(tmp_path):
    src, man = _served(tmp_path)
    (src / "a.csv").unlink()
    with pytest.raises(Exception):
        rt.download(src.as_uri(), tmp_path / "bundle", man)


def test_check_value_compares_names_and_rounded_numbers():
    assert rt.check_value("x", "§", "jackknife", 0, "jackknife")["verdict"] == "PASS"
    assert rt.check_value("x", "§", "jackknife", 0, "fixed_effect")["verdict"] == "FAIL"
    assert rt.check_value("x", "§", 0.2524, 4, 0.25242286279805487)["verdict"] == "PASS"
    assert rt.check_value("x", "§", 0.2524, 4, 0.25252)["verdict"] == "FAIL"


def test_nine_arm_expected_values_are_written_not_computed():
    """gotcha 32 (b): every expected cell is a literal in the script."""
    assert set(rt.NINE_ARM) == {"w2560", "w10240", "w20480", "w40960", "k64_w2560",
                                "k64n9", "k64_w40960", "k16_w10240", "k128_w10240"}
    for arm, r in rt.NINE_ARM.items():
        assert r["slope"][3] in ("fixed_effect", "across_seed", "jackknife"), arm
    assert rt.NINE_ARM["k64n9"]["slope"][:3] == (1.7016, 1.4288, 1.9745)


def test_an_incomplete_extended_bundle_is_refused_by_name(tmp_path):
    with pytest.raises(rt.BundleError, match="extended bundle"):
        rt.require_extended(tmp_path)


def test_cli_exits_one_on_a_refused_bundle(tmp_path):
    b = _mini_bundle(tmp_path)
    (b / "meta.json").write_text("changed", encoding="utf-8")
    assert rt.main(["--bundle", str(b)]) == 1


# ---------------------------------------------------------------------------
# the real bundle
# ---------------------------------------------------------------------------

BUNDLE = REPO / "release" / "derived"


def _bundle_present() -> bool:
    """The manifest lists the extended bundle (§CT18-F) and every file it lists
    is here. Until the extended manifest is committed, the five-file bundle of
    §CT18 (b) is not enough for the regeneration, and the run is skipped."""
    man = BUNDLE / "MANIFEST.csv"
    if not man.exists():
        return False
    rows = list(csv.DictReader(open(man, encoding="utf-8", newline="")))
    names = {Path(r["path"]).name for r in rows}
    return (set(rt.EXTENDED) <= names
            and all((BUNDLE / n).exists() for n in names))


@pytest.mark.skipif(not _bundle_present(),
                    reason="the extended bundle (§CT18-F) is not in this checkout: its "
                           "manifest is not committed yet, or the data files (release "
                           "assets, fetched with --download) are absent; the run on the "
                           "bundle is skipped, not passed")
def test_reproduce_table1_on_the_bundle(tmp_path):
    out_json = tmp_path / "out.json"
    assert rt.main(["--bundle", str(BUNDLE), "--json", str(out_json)]) == 0
    out = json.loads(out_json.read_text(encoding="utf-8"))
    # 14 (§CT18 (c)) + Table 1's other two intervals at 0.60 and 0.70 (8) + the
    # nine-arm table, 8 cells per arm (72) -- §CT18-F item 6
    assert out["n_fail"] == 0 and len(out["checks"]) == 94


def test_the_launcher_command_parses():
    """Every flag `slurm/release_derived.sbatch` passes is one the module
    accepts, and every required flag is passed (read from the file, so a flag
    renamed on one side fails here rather than on the cluster)."""
    import re
    import shlex
    from src.analysis.release_derived import build_parser
    text = (REPO / "slurm" / "release_derived.sbatch").read_text(encoding="utf-8")
    cmd = text.split("python -m src.analysis.release_derived", 1)[1].split("\nif ", 1)[0]
    cmd = re.sub(r'"\$[A-Z]+[^"]*"', "/x", cmd.replace("\\" + "\n", " "))
    ns = build_parser().parse_args(shlex.split(cmd))
    assert str(ns.out_dir) == str(Path("/x")) and ns.degenerate_flag is not None


def test_import_pins_blas_threads_to_one_before_numpy():
    """A caller's OMP_NUM_THREADS=8 is overridden at import (§CT18-D3 results):
    the owners' fits ran single-threaded, and on Linux four threads moved fitted
    values off the record."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    code = ("import os, sys; sys.path.insert(0, '.'); import scripts.reproduce_table1; "
            "print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'], "
            "os.environ['OPENBLAS_NUM_THREADS'])")
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                       env=env, capture_output=True, text=True, check=True)
    assert r.stdout.split() == ["1", "1", "1"]


def test_single_threaded_limits_every_pool_inside_the_block():
    pytest.importorskip("threadpoolctl")
    with rt.single_threaded() as pools:
        from threadpoolctl import threadpool_info
        live = threadpool_info()
    assert pools and all(p["num_threads"] == 1 for p in pools)
    assert all(p["num_threads"] == 1 for p in live)


def test_torch_is_imported_before_any_thread_pool_is_touched():
    """On Windows a PyTorch imported after threadpoolctl has loaded MKL's OpenMP
    runtime fails to load (WinError 127, measured 2026-11-13 in an environment
    built from env/environment.yml). `run` therefore imports it first, on every
    platform. Read in a fresh process, with the pool-limiting block replaced by
    one that records what was imported when it was entered and then stops."""
    pytest.importorskip("torch")
    import subprocess
    import sys
    code = (
        "import sys, contextlib; sys.path.insert(0, '.')\n"
        "import scripts.reproduce_table1 as rt\n"
        "assert 'torch' not in sys.modules and 'threadpoolctl' not in sys.modules\n"
        "class Stop(Exception): pass\n"
        "@contextlib.contextmanager\n"
        "def probe():\n"
        "    print('torch' in sys.modules, 'threadpoolctl' in sys.modules)\n"
        "    raise Stop()\n"
        "    yield\n"
        "rt.single_threaded = probe\n"
        "try:\n"
        "    rt.run('no-such-bundle')\n"
        "except Stop:\n"
        "    pass\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["True", "False"]


def test_progress_writes_one_line_per_fit_and_restores_the_function(monkeypatch):
    """The progress lines wrap `design3_join.fit_arm` for the block: arguments
    and the result pass through untouched, each call writes one line naming the
    stage, and the original function is back afterwards."""
    import io
    import src.analysis.design3_join as d3
    seen = []

    def stub(*a, **k):
        seen.append((a, k))
        return {"the": "result"}
    monkeypatch.setattr(d3, "fit_arm", stub)
    buf = io.StringIO()
    with rt.Progress(buf) as pr:
        assert d3.fit_arm is not stub
        pr.set("stage one")
        assert d3.fit_arm(1, 2, x=3) == {"the": "result"}
        pr.set("stage two")
        assert d3.fit_arm(4) == {"the": "result"}
    assert d3.fit_arm is stub
    assert seen == [((1, 2), {"x": 3}), ((4,), {})]
    lines = buf.getvalue().splitlines()
    assert all(ln.startswith("[reproduce_table1 ") for ln in lines)
    assert [ln.split("] ", 1)[1] for ln in lines] == [
        "stage: stage one", "fit 1 (stage one)", "stage: stage two", "fit 2 (stage two)",
        "done: 2 fits"]


def test_a_refused_bundle_writes_progress_to_standard_error_only(tmp_path, capsys):
    """Standard output carries the tier, the checks and the last line; the
    progress lines go to standard error (a refusal has no standard output)."""
    pytest.importorskip("torch")
    assert rt.main(["--bundle", str(tmp_path)]) == 1
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "[reproduce_table1 " in cap.err and "REFUSED: no manifest at" in cap.err


# §CT18-D4 item 3: the two declared tiers.
REF = {"system": "Linux", "python": "3.11.15", "numpy": "2.4.6", "scipy": "1.17.1",
       "h5py": "3.16.0", "blas": ["openblas 0.3.33"], "threads": [1]}


def test_tier_is_reference_only_when_every_fact_matches():
    assert rt.tier(dict(REF)) == ("REFERENCE", [])
    for k, v in [("system", "Windows"), ("python", "3.13.9"), ("numpy", "2.3.5"),
                 ("scipy", "1.16.3"), ("h5py", "3.15.1"), ("blas", ["mkl 2025.0"]),
                 ("threads", [4]), ("threads", []), ("blas", [])]:
        name, why = rt.tier(dict(REF, **{k: v}))
        assert name == "PORTABLE" and len(why) == 1, (k, v, why)


def _chk(name, record, recomputed):
    return {"name": name, "record": record, "recomputed": recomputed, "verdict":
            "PASS" if round(recomputed, 4) == round(record, 4) else "FAIL"}


def test_portable_widens_only_fitted_interval_endpoints():
    cases = [_chk("betabinomial_0.70_jackknife_ci_hi", 0.4557, 0.45564),      # endpoint, in tol
             _chk("k64_w40960_slope_widest_lo", 0.2362, 0.23627),             # endpoint, in tol
             _chk("w2560_slope_widest_hi", 0.6591, 0.6610),                   # endpoint, 1.9e-3 out
             _chk("betabinomial_0.70_fe_pp", 0.2889, 0.28896),                # fixed effect
             _chk("headline_mean_rf_0.70_n39", 0.4955, 0.49556)]              # counted value
    assert rt.apply_tier([dict(c) for c in cases], "REFERENCE") == 5
    got = [dict(c) for c in cases]
    assert rt.apply_tier(got, "PORTABLE") == 3
    assert [c["verdict"] for c in got] == ["PASS_BY_TOLERANCE", "PASS_BY_TOLERANCE",
                                           "FAIL", "FAIL", "FAIL"]


def test_the_windows_run_of_2026_10_29_passes_portable_and_fails_reference():
    """Report 1's Windows run (91 of 94): its three failures are the endpoints
    named here, and each lies within the PORTABLE tolerance."""
    p = (Path(__file__).resolve().parents[1] / "analysis" / "adhoc" / "probes" / "2026-10-29"
         / "release" / "reproduce_windows.json")
    checks = json.loads(p.read_text(encoding="utf-8"))["checks"]
    failed = sorted(c["name"] for c in checks if c["verdict"] == "FAIL")
    assert failed == ["betabinomial_0.70_jackknife_ci_hi", "k64_w40960_slope_widest_lo",
                      "w10240_slope_widest_hi"]
    assert rt.apply_tier([dict(c) for c in checks], "REFERENCE") == 3
    assert rt.apply_tier([dict(c) for c in checks], "PORTABLE") == 0


def test_the_reference_environment_file_pins_the_reference_tier():
    """env/reference_environment.yml and REFERENCE_ENV must not drift apart
    (§CT18-D4 item 3)."""
    yaml = pytest.importorskip("yaml")
    p = Path(__file__).resolve().parents[1] / "env" / "reference_environment.yml"
    env = yaml.safe_load(p.read_text(encoding="utf-8"))
    pins = dict(d.split("=", 1) for d in env["dependencies"] if isinstance(d, str) and "=" in d)
    ref = rt.REFERENCE_ENV
    assert pins["python"] == ref["python"] and pins["numpy"] == ref["numpy"]
    assert pins["scipy"] == ref["scipy"] and pins["h5py"] == ref["h5py"]
    assert ref["blas"] == f"openblas {pins['libopenblas']}"
    assert "threadpoolctl" in pins and "prefix" not in env
