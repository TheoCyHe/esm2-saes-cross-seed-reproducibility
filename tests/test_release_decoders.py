"""src/analysis/release_decoders.py (§CT18-D4 item 2): toy checkpoints, the
gates on the written asset, numeric seed order, and byte determinism."""
import csv
import json

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.analysis import release_decoders as rd  # noqa: E402

W = 8          # toy width


def _ckpts(root, labels, seed=0):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    for lab in labels:
        d = root / f"seed_{lab}"
        d.mkdir()
        torch.save({"W_dec": torch.from_numpy(rng.standard_normal((rd.D_MODEL, W)).astype("float32")),
                    "W_enc": torch.zeros(W, rd.D_MODEL)}, d / "model.pt")
        (d / "meta.json").write_text(json.dumps({"git_sha": f"sha{lab}"}), encoding="utf-8")
    return root


def test_build_writes_numeric_order_and_passes_its_gates(tmp_path):
    a = _ckpts(tmp_path / "a", [0, 1, 10])          # as strings, 10 would sort before 2 (gotcha 29)
    b = _ckpts(tmp_path / "b", [2], seed=1)
    sets = {"toy": ((a, b), W, 4)}
    res = rd.build(tmp_path / "out", ["toy"], "c" * 40, sets=sets)
    assert res["all_pass"], [g for g in res["gates"] if g["verdict"] != "PASS"]
    with h5py.File(tmp_path / "out" / "decoders_toy.h5", "r") as hf:
        assert json.loads(hf.attrs["seed_labels"]) == [0, 1, 2, 10]
        assert json.loads(hf.attrs["checkpoint_git_sha"]) == ["sha0", "sha1", "sha2", "sha10"]
        want = torch.load(a / "seed_10" / "model.pt", weights_only=True)["W_dec"].numpy()
        assert np.array_equal(hf["W_dec"][3], want)
    rows = list(csv.DictReader(open(tmp_path / "out" / "MANIFEST.csv", encoding="utf-8")))
    assert rows[0]["asset"] == "decoders_toy.h5" and rows[0]["n_dictionaries"] == "4"


def test_a_changed_slice_fails_the_bit_identity_gate(tmp_path):
    a = _ckpts(tmp_path / "a", [0, 1])
    seeds = rd.seed_dirs([a])
    p = tmp_path / "x.h5"
    rd.write_asset(p, seeds, W, "c" * 40)
    with h5py.File(p, "r+") as hf:
        hf["W_dec"][1, 0, 0] = np.nextafter(hf["W_dec"][1, 0, 0], np.float32(np.inf))
    g = {x["what"].split(": ", 1)[1]: x for x in rd.gate_asset(p, seeds, W, 2)}
    assert g["every slice equals its checkpoint's W_dec bit for bit"]["verdict"] == "FAIL"
    assert g["every slice equals its checkpoint's W_dec bit for bit"]["differing_seeds"] == [1]


def test_wrong_count_and_size_limit_fail(tmp_path):
    a = _ckpts(tmp_path / "a", [0, 1])
    seeds = rd.seed_dirs([a])
    p = tmp_path / "x.h5"
    rd.write_asset(p, seeds, W, "c" * 40)
    g = rd.gate_asset(p, seeds, W, n_expected=3, limit=100)
    assert [x["verdict"] for x in g] == ["FAIL", "FAIL", "FAIL", "FAIL"]


def test_writing_twice_gives_identical_bytes(tmp_path):
    a = _ckpts(tmp_path / "a", [0, 1, 2])
    seeds = rd.seed_dirs([a])
    rd.write_asset(tmp_path / "one.h5", seeds, W, "c" * 40)
    rd.write_asset(tmp_path / "two.h5", seeds, W, "c" * 40)
    assert rd.sha256_file(tmp_path / "one.h5") == rd.sha256_file(tmp_path / "two.h5")


def test_a_duplicate_seed_label_is_refused(tmp_path):
    a = _ckpts(tmp_path / "a", [0])
    b = _ckpts(tmp_path / "b", [0], seed=1)
    with pytest.raises(SystemExit, match="seed 0 is in both"):
        rd.seed_dirs([a, b])


def test_verify_checks_existing_files_against_the_checkpoints_and_a_table(tmp_path):
    a = _ckpts(tmp_path / "a", [0, 1, 2])
    sets = {"toy": ((a,), W, 3)}
    out = tmp_path / "out"
    rd.build(out, ["toy"], "c" * 40, sets=sets)
    p = out / "decoders_toy.h5"
    sums = tmp_path / "sums.tsv"

    def table(sha=None, md5=None, size=None, extra=""):
        sums.write_text("asset\toriginal_sha256\tsha256\tbytes\tmd5\n"
                        f"{p.name}\t{'0' * 64}\t{sha or rd.sha256_file(p)}\t{size or p.stat().st_size}\t"
                        f"{md5 or rd.md5_file(p)}\n" + extra, encoding="utf-8")

    table()
    res = rd.verify(out, ["toy"], sums, "d" * 40, sets=sets)
    assert res["all_pass"] and len(res["gates"]) == 6           # four of the file, the table's row, no extra
    assert rd.verify(out, ["toy"], None, "d" * 40, sets=sets)["all_pass"]
    # a table that states another hash, another md5 or another size fails that gate and only it
    for kw in ({"sha": "1" * 64}, {"md5": "1" * 32}, {"size": 1}):
        table(**kw)
        bad = [g["what"] for g in rd.verify(out, ["toy"], sums, "d" * 40, sets=sets)["gates"] if g["verdict"] != "PASS"]
        assert bad == [f"{p.name}: sha256, md5 and size as the table's row"], (kw, bad)
    # a table naming a file that was not verified, a missing file and no set at all each fail
    table(extra="decoders_other.h5\t0\t0\t1\t0\n")
    assert not rd.verify(out, ["toy"], sums, "d" * 40, sets=sets)["all_pass"]
    assert not rd.verify(tmp_path / "nowhere", ["toy"], None, "d" * 40, sets=sets)["all_pass"]
    assert not rd.verify(out, [], None, "d" * 40, sets=sets)["all_pass"]
    # a slice that differs from its checkpoint fails the bit-identity gate although the table is rewritten to match
    with h5py.File(p, "r+") as hf:
        hf["W_dec"][2, 0, 0] = np.nextafter(hf["W_dec"][2, 0, 0], np.float32(np.inf))
    table()
    res = rd.verify(out, ["toy"], sums, "d" * 40, sets=sets)
    bad = [g for g in res["gates"] if g["verdict"] != "PASS"]
    assert not res["all_pass"] and len(bad) == 1 and bad[0]["differing_seeds"] == [2]


def test_the_sets_cover_140_dictionaries_in_ten_assets():
    assert len(rd.SETS) == 10 and sum(n for _, _, n in rd.SETS.values()) == 140
