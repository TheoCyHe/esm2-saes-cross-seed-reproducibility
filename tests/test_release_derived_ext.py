"""Amendment CT §CT18-F: the extended bundle's flat-file I/O and its gates.

The gates' real test is the build on Bouchet, against the owner artifacts;
these tests show each piece can fail: a changed value, a mislabelled row, a
finite own column, and a one-ulp difference are each caught."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import release_derived_ext as rx

REPO = Path(__file__).resolve().parents[1]


def _sim(n_a=3, n_s=4, d=50, seed=0):
    rng = np.random.default_rng(seed)
    sim = rng.uniform(0.2, 1.0, (n_a, n_s, d)).astype(np.float32)
    sim[rng.random(sim.shape) < 0.3] = -np.inf          # no mutual partner
    for a in range(n_a):
        sim[a, a] = -np.inf                              # own column
    return sim


def test_sim_round_trips_exactly_with_labels(tmp_path):
    sim = _sim()
    anchors, cols = ["3", "1", "7"], ["3", "1", "7", "9"]
    p = tmp_path / "s.csv.gz"
    rx.write_sim(p, sim, anchors, cols)
    back, a, c = rx.read_sim(p, np.float32)
    assert a == anchors and c == cols
    assert np.array_equal(back, sim)


def test_float64_similarity_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    sim = rng.normal(size=(2, 3, 20))
    sim[0, 0] = -np.inf
    sim[1, 1] = -np.inf
    p = tmp_path / "s64.csv.gz"
    rx.write_sim(p, sim, ["0", "1"], ["0", "1", "2"])
    back, _, _ = rx.read_sim(p, np.float64)
    assert np.array_equal(back, sim)


def test_a_finite_own_column_is_refused(tmp_path):
    sim = _sim()
    sim[1, 1, 5] = 0.9
    with pytest.raises(ValueError, match="own column"):
        rx.write_sim(tmp_path / "x.csv.gz", sim, ["0", "1", "2"], ["0", "1", "2", "3"])


def test_rows_out_of_anchor_order_are_refused(tmp_path):
    import gzip
    sim = _sim()
    p = tmp_path / "s.csv.gz"
    rx.write_sim(p, sim, ["0", "1", "2"], ["0", "1", "2", "3"])
    lines = gzip.open(p, "rt", encoding="utf-8").read().splitlines()
    lines[1], lines[51] = lines[51], lines[1]            # swap a row of anchor 0 with anchor 1
    q = tmp_path / "t.csv.gz"
    rx.write_csv_gz(q, lines[0].split(","), [ln.split(",") for ln in lines[1:]])
    with pytest.raises(ValueError):
        rx.read_sim(q, np.float32)


def test_table_round_trips_bool_int_float_and_nan(tmp_path):
    rng = np.random.default_rng(2)
    cols = {"k_0.70": rng.integers(0, 9, (2, 30)), "M": rng.random((2, 30)),
            "keep_a4a": rng.random((2, 30)) < 0.5,
            "s_f": rng.random((2, 30)).astype(np.float32)}
    cols["s_f"][1, 3] = np.nan
    p = tmp_path / "t.csv.gz"
    rx.write_table(p, ["4", "2"], cols)
    back, a = rx.read_table(p, {"k_0.70": np.int64, "M": np.float64, "keep_a4a": bool,
                                "s_f": np.float32})
    assert a == ["4", "2"]
    for k, v in cols.items():
        dt = {"k_0.70": np.int64, "M": np.float64, "keep_a4a": bool, "s_f": np.float32}[k]
        assert rx.arrays_equal(back[k], v, dt), k


def test_gzip_is_deterministic(tmp_path):
    sim = _sim()
    p1, p2 = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    rx.write_sim(p1, sim, ["0", "1", "2"], ["0", "1", "2", "3"])
    rx.write_sim(p2, sim, ["0", "1", "2"], ["0", "1", "2", "3"])
    assert rx.sha256_file(p1) == rx.sha256_file(p2)


def test_first_difference_finds_one_ulp():
    a = {"x": [1.0, {"y": 0.28893591021007203}]}
    b = {"x": [1.0, {"y": float(np.nextafter(0.28893591021007203, 1.0))}]}
    assert rx.first_difference(a, a) is None
    assert rx.verdict(a, a, "t")["verdict"] == "BIT_IDENTICAL"
    v = rx.verdict(a, b, "t")
    assert v["verdict"] == "DIFFERS" and "/x[1]/y" in v["first_difference"]
    assert rx.first_difference({"a": 1}, {"a": 1, "b": 2}) is not None


def _fit_inputs(n_a=4, d=300, seed=0):
    rng = np.random.default_rng(seed)
    sds = np.array([1.5, 2.0, 2.5, 3.0])[:n_a]
    logM = np.stack([rng.normal(0.0, s, d) for s in sds])
    p = 1 / (1 + np.exp(-(0.2 + 0.3 * logM / sds[:, None])))
    sim = np.where(rng.random((n_a, n_a, d)) < p[:, None, :],
                   rng.uniform(0.7, 1.0, (n_a, n_a, d)),
                   rng.uniform(0.0, 0.69, (n_a, n_a, d))).astype(np.float32)
    for a in range(n_a):
        sim[a, a] = -np.inf
    return (sim, logM, rng.uniform(-4, -1, (n_a, d)), rng.uniform(0, 1, (n_a, d)),
            np.ones((n_a, d), dtype=bool))


@pytest.fixture(scope="module")
def block():
    sim, logM, log_phi, d_fs, keep = _fit_inputs()
    return rx.slope_block(sim, logM, log_phi, d_fs, keep, [0, 1, 2, 3])


def test_slope_block_has_the_owners_structure(block):
    """The same keys as a committed slope owner's blocks (structure only; the
    values are the build gate's business)."""
    o = json.loads((REPO / "analysis/adhoc/probes/2026-10-24/ct24_ct25/slope_w2560_ct.json")
                   .read_text(encoding="utf-8"))
    assert set(block["per_theta"]) == set(o["per_theta"]["0.70"])
    assert set(block["full_fit"]) == set(o["full_fit"]["0.70"])
    assert set(block["per_theta"]["loo"][0]) == set(o["per_theta"]["0.70"]["loo"][0])
    assert block["per_theta"]["jackknife"]["n"] == 4


def test_one_changed_similarity_cell_changes_the_block(block):
    sim, logM, log_phi, d_fs, keep = _fit_inputs()
    sim2 = sim.copy()
    i = np.argwhere(np.isfinite(sim2[0, 1]) & (sim2[0, 1] >= 0.70))[0, 0]
    sim2[0, 1, i] = 0.5                                   # one match removed
    b2 = rx.slope_block(sim2, logM, log_phi, d_fs, keep, [0, 1, 2, 3])
    assert rx.verdict(b2["per_theta"], block["per_theta"], "t")["verdict"] == "DIFFERS"
    b3 = rx.slope_block(sim, logM, log_phi, d_fs, keep, [0, 1, 2, 3])
    assert rx.verdict(b3["per_theta"], block["per_theta"], "t")["verdict"] == "BIT_IDENTICAL"


def test_arm_paths_name_the_launchers_files():
    p = rx.arm_paths("w2560", Path("/A"))
    assert p["rf_counts_all_theta"] == Path("/A/cs/rf_counts_all_theta_w2560_n9.h5")
    assert p["slope_owner"] == Path("/A/ct/slope_w2560_ct.json")
    t = rx.arm_paths("k16_w10240", Path("/A"))
    assert t["gtilde"] == Path("/A/ct/gtilde_k16_w10240_b_ratio_topmerged_ct.h5")
    assert t["slope_owner"] == Path("/A/ct/slope_k16_w10240_topmerged_ct.json")
    assert t["sffit_owner"] is None
    k = rx.arm_paths("k64", Path("/A"))
    assert k["k64n9_arm_owner"] == Path("/A/ct/ct28_arm_k64n9_ct.json")
    for name, a in rx.ARMS.items():
        assert (REPO / a["config"]).exists(), name


def test_drop_paths_exempts_only_the_named_paths():
    o = {"identity": {"git_sha": "a"}, "strata": {"exp": {"gtilde": "/x", "fit": {"fe": 1.0}},
                                                  "prop": {"gtilde": "/y", "fit": {"fe": 2.0}}}}
    d = rx.drop_paths(o, ["/identity", "/strata/*/gtilde"])
    assert d == {"strata": {"exp": {"fit": {"fe": 1.0}}, "prop": {"fit": {"fe": 2.0}}}}
    assert "identity" in o                                 # the input is not modified


def test_replay_verdict_ignores_exempt_and_catches_the_rest(tmp_path):
    a = {"identity": {"host": "x"}, "half_A_gtilde": {"n": 1}, "arms": {"dec_B": {"fe": 0.2852}}}
    b = {"identity": {"host": "y"}, "half_A_gtilde": {"n": 2}, "arms": {"dec_B": {"fe": 0.2852}}}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")
    assert rx.replay_verdict(pa, pb, ["/half_A_gtilde"], "t")["verdict"] == "BIT_IDENTICAL"
    assert rx.replay_verdict(pa, pb, [], "t")["verdict"] == "DIFFERS"
    b["arms"]["dec_B"]["fe"] = 0.28520000000000006
    pb.write_text(json.dumps(b), encoding="utf-8")
    assert rx.replay_verdict(pa, pb, ["/half_A_gtilde"], "t")["verdict"] == "DIFFERS"


def test_patched_loader_returns_a_copy_and_restores():
    from src.analysis import jackknife_dictionaries as jk
    orig = jk.load_join_inputs
    inp = {"sim": np.zeros(1), "logM": np.ones(1)}
    with rx.patched_loader(inp):
        got = jk.load_join_inputs(None)
        got["sim"] = "replaced"
        assert jk.load_join_inputs(None)["sim"] is inp["sim"]
    assert jk.load_join_inputs is orig


def test_record_hash_check_has_four_outcomes():
    sha = "a17472e8d8911f0515d238a0c67c79f66f3e230f20aaca0aafb311c60035c084"
    t = {"NoR": "owner `x_ct.json` (sha256 `a17472e8d8911f05…`)\n"
                "`y_ct.json` sha256 `ffffffffffffffff`\n"
                "`z_ct.json` and `w_ct.json`: `1111aaaa1111aaaa`, `2222bbbb2222bbbb`\n"
                "`v_ct.json` git_sha 94ca8b98a5f40bc61def883f9d00b7d9479b8f06"}
    assert rx.record_hash_check("x_ct.json", sha, t)["verdict"] == "MATCH"
    assert rx.record_hash_check("y_ct.json", sha, t)["verdict"] == "MISMATCH"
    assert rx.record_hash_check("z_ct.json", sha, t)["verdict"] == "UNRESOLVED"
    assert rx.record_hash_check("v_ct.json", sha, t)["verdict"] == "NOT_STATED"   # a 40-hex git sha is not a hash
    assert rx.record_hash_check("u_ct.json", sha, t)["verdict"] == "NOT_STATED"
    # the job-31211370 false positive: 16 digits of a decimal are not a hash
    dec = {"NoR": "| rate | 0.4594441105769231 | `q_ct.json` `mean_match_rate` |"}
    assert rx.record_hash_check("q_ct.json", sha, dec)["verdict"] == "NOT_STATED"
    # a hash stated on the line after the file's name
    nxt = {"NoR": "**Owner:** `r_ct.json` (sha256\n"
                  "`a17472e8d8911f0515d238a0c67c79f66f3e230f20aaca0aafb311c60035c084`, job 1)"}
    assert rx.record_hash_check("r_ct.json", sha, nxt)["verdict"] == "MATCH"


def test_subspace_recount_drops_only_the_unstored_fields():
    rng = np.random.default_rng(3)
    n = 60
    t = {"c": rng.integers(0, 5, n), "feasible": rng.random(n) < 0.9, "rank_ok": rng.random(n) < 0.95,
         "alignment_ok": rng.random(n) < 0.9, "tie": rng.random(n) < 0.05,
         "match_fraction": rng.random(n), "omega": rng.random(n),
         "omega_null_mean": rng.random(n) * 0.1, "omega_null_sd": rng.random(n) * 0.05 + 0.01,
         "omega_separation": rng.normal(5, 1, n), "omega_exceedance": rng.random(n)}
    s = rx.subspace_recount(t, [f"C{i}" for i in range(5)])
    assert "n_null_draws_rank_deficient" not in s["accounting"]
    assert "fraction_beyond_95" not in s["distributions"]
    assert all("fraction_beyond_95" not in r for r in s["per_concept"])
    assert s["accounting"]["n_cells"] == n and "crosstab" in s and "CT22_P1" in s


def _gates(d: Path, fail: str | None = None, skip: str | None = None):
    for fam in rx.FAMILIES:
        if fam == skip:
            continue
        (d / f"gates_{fam}.json").write_text(json.dumps(
            {"all_pass": fam != fail, "gates": [{"what": "x", "verdict": "PASS"}]}), encoding="utf-8")


def test_manifest_refuses_a_missing_family_and_a_failed_gate(tmp_path):
    out, proj = tmp_path / "out", tmp_path / "proj"
    out.mkdir()
    proj.mkdir()
    _gates(out, skip="replay")
    with pytest.raises(SystemExit, match="gates_replay.json is absent"):
        rx.build_manifest(out, proj, copy=False)
    _gates(out, fail="arm_w2560")
    with pytest.raises(SystemExit, match="gates_arm_w2560.json has a failed gate"):
        rx.build_manifest(out, proj, copy=False)


def test_every_planned_file_has_an_owning_row():
    names = ["sim_decoder_n39.csv.gz", "sim_activation_n39.csv.gz", "sim_activation_halfA_n39.csv.gz",
             "sim_oneway_n39.csv.gz", "sim_rule1_n19.csv.gz", "sim_rule3_rawdot_n19.csv.gz",
             "features_logM_variants_n39.csv.gz", "features_carrier_n39.csv.gz",
             "features_disorder_n39.csv.gz", "features_classification_n39.csv.gz",
             "subspace_ct26_nonanchor_ct__m3_ap.csv.gz", "owner__jackknife_ct.json",
             "arm_k64_sim_activation.csv.gz"]
    names += [f"arm_{a}_{x}" for a in rx.ARMS
              for x in ("features.csv.gz", "sim_decoder.csv.gz", "fvu_per_seed.csv")]
    for n in names:
        assert rx.owning_row(n)
    assert rx.owning_row("arm_k64_w2560_features.csv.gz") != rx.owning_row("arm_k64_features.csv.gz")
    with pytest.raises(SystemExit):
        rx.owning_row("unplanned_file.csv")


# §CT18-D2's owed items: the input trace, the record-hash check on inputs, and
# the meta-only update of a shipped bundle.

def test_trace_records_reads_through_open_gzip_and_h5py_not_writes(tmp_path):
    import gzip
    import h5py
    h5 = tmp_path / "a.h5"
    with h5py.File(h5, "w") as hf:
        hf["x"] = np.arange(3)
    gz = tmp_path / "b.csv.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write("x\n")
    js = tmp_path / "c.json"
    js.write_text("{}", encoding="utf-8")
    written = tmp_path / "d.txt"
    outside = tmp_path / "e.txt"
    outside.write_text("e", encoding="utf-8")
    with rx.trace_inputs() as seen:
        with h5py.File(h5, "r") as hf:
            hf["x"][:]
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            f.read()
        json.loads(js.read_text(encoding="utf-8"))
        written.write_text("w", encoding="utf-8")
    outside.read_text(encoding="utf-8")            # after the block: not traced
    assert {h5.resolve(), gz.resolve(), js.resolve()} <= seen
    assert written.resolve() not in seen and outside.resolve() not in seen


def test_inputs_block_classifies_and_flags_a_contradicted_record_hash(tmp_path):
    root, out, bundle, repo = (tmp_path / n for n in ("proj", "out", "bundle", "repo"))
    for d in (root, out, bundle, repo):
        d.mkdir()
    (root / "art_ct.h5").write_bytes(b"artifact")
    (root / "ok_ct.h5").write_bytes(b"ok")
    (out / "mine.csv.gz").write_bytes(b"output")
    (bundle / "features_n39.csv.gz").write_bytes(b"bundle")
    (repo / "configs.yaml").write_bytes(b"repo")
    (repo / "mod.py").write_bytes(b"code")
    other = tmp_path / "elsewhere.txt"
    other.write_bytes(b"x")
    seen = {p.resolve() for p in (root / "art_ct.h5", root / "ok_ct.h5", out / "mine.csv.gz",
                                  bundle / "features_n39.csv.gz", repo / "configs.yaml",
                                  repo / "mod.py", other)}
    ok_sha = rx.sha256_file(root / "ok_ct.h5")
    texts = {"NoR": f"`art_ct.h5` sha256 `ffffffffffffffff`\n`ok_ct.h5` sha256 `{ok_sha[:16]}`"}
    rows = rx.inputs_block(seen, out, bundle, repo, texts, roots=(root,))
    kinds = {Path(r["path"]).name: r["kind"] for r in rows}
    assert kinds == {"art_ct.h5": "artifact", "ok_ct.h5": "artifact",
                     "features_n39.csv.gz": "bundle", "configs.yaml": "repo",
                     "elsewhere.txt": "other"}             # the output and the .py are not inputs
    rh = {Path(r["path"]).name: r.get("record_hash") for r in rows}
    assert rh["art_ct.h5"] == "MISMATCH" and rh["ok_ct.h5"] == "MATCH"
    g = rx.input_gates(rows, root / "ok_ct.h5")
    assert g[0]["verdict"] == "PASS" and g[1]["verdict"] == "FAIL"
    assert g[1]["mismatch"] == [str((root / "art_ct.h5").resolve())]
    # the mutant: the primary input was never opened
    assert rx.input_gates(rows, root / "never_opened.h5")[0]["verdict"] == "FAIL"
    clean = [r for r in rows if r.get("record_hash") != "MISMATCH"]
    assert rx.input_gates(clean, root / "ok_ct.h5")[1]["verdict"] == "PASS"


def _shipped(tmp_path, built="b" * 40):
    """A shipped project bundle (one old row, two built rows, meta_ext) and a
    traced rebuild of it with every family's gates and inputs."""
    import csv as _csv
    proj, out = tmp_path / "proj", tmp_path / "out"
    proj.mkdir()
    out.mkdir()
    files = {"old.csv.gz": b"old", "x.csv.gz": b"xx", "y.csv": b"yy"}
    for n, b in files.items():
        (proj / n).write_bytes(b)
    (proj / "meta_ext.json").write_text(json.dumps({"producing_commit": built}), encoding="utf-8")
    rows = [["release/derived/old.csv.gz", rx.sha256_file(proj / "old.csv.gz"), "a" * 40, "§1"],
            ["release/derived/x.csv.gz", rx.sha256_file(proj / "x.csv.gz"), built, "§2"],
            ["release/derived/y.csv", rx.sha256_file(proj / "y.csv"), built, "§3"],
            ["release/derived/meta_ext.json", rx.sha256_file(proj / "meta_ext.json"), built, "—"]]
    with open(proj / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256", "producing_commit", "owning_nor_row"])
        w.writerows(rows)
    for n in ("x.csv.gz", "y.csv"):
        (out / n).write_bytes(files[n])
    for fam in rx.FAMILIES:
        (out / f"gates_{fam}.json").write_text(json.dumps(
            {"all_pass": True, "gates": [{"what": "x", "verdict": "PASS"}],
             "inputs": [{"path": f"/p/{fam}.h5", "kind": "artifact", "sha256": "0" * 64,
                         "record_hash": "NOT_STATED"}]}), encoding="utf-8")
    return proj, out, rows


def test_update_meta_changes_only_meta_ext_and_its_row(tmp_path, monkeypatch):
    import csv as _csv
    from src.sae import train
    monkeypatch.setattr(train, "git_sha", lambda: "c" * 40)
    monkeypatch.setattr(train, "git_dirty", lambda: False)
    proj, out, rows = _shipped(tmp_path)
    before = {n: (proj / n).read_bytes() for n in ("old.csv.gz", "x.csv.gz", "y.csv")}
    g = rx.update_meta(out, proj, copy=True)
    assert g["all_pass"] and g["n_data_files_byte_identical"] == 2
    new = list(_csv.reader(open(proj / "MANIFEST.csv", encoding="utf-8", newline="")))[1:]
    assert new[:3] == rows[:3]                                   # other rows untouched
    assert new[3][2] == "c" * 40 and new[3][1] == rx.sha256_file(proj / "meta_ext.json")
    meta = json.loads((proj / "meta_ext.json").read_text(encoding="utf-8"))
    assert meta["producing_commit"] == "b" * 40                  # the data's build commit kept
    assert set(meta["inputs"]) == set(rx.FAMILIES)
    assert meta["inputs_recorded"]["record_hash_verdicts_artifact_inputs"] == {
        "NOT_STATED": len(rx.FAMILIES)}
    assert before == {n: (proj / n).read_bytes() for n in before}


def test_update_meta_refuses_a_rebuild_that_is_not_byte_identical(tmp_path, monkeypatch):
    from src.sae import train
    monkeypatch.setattr(train, "git_sha", lambda: "c" * 40)
    monkeypatch.setattr(train, "git_dirty", lambda: False)
    proj, out, _ = _shipped(tmp_path)
    (out / "y.csv").write_bytes(b"yz")
    with pytest.raises(SystemExit, match="not byte-identical"):
        rx.update_meta(out, proj, copy=True)
    (out / "y.csv").write_bytes(b"yy")
    (out / "z.csv").write_bytes(b"extra")
    with pytest.raises(SystemExit, match="differ from the manifest"):
        rx.update_meta(out, proj, copy=True)
    (out / "z.csv").unlink()
    g = json.loads((out / "gates_replay.json").read_text(encoding="utf-8"))
    del g["inputs"]
    (out / "gates_replay.json").write_text(json.dumps(g), encoding="utf-8")
    with pytest.raises(SystemExit, match="no input record"):
        rx.update_meta(out, proj, copy=True)
    assert json.loads((proj / "meta_ext.json").read_text(encoding="utf-8")) == {
        "producing_commit": "b" * 40}                            # nothing copied on refusal
