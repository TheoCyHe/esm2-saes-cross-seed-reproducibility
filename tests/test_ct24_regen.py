"""§CT24-F item 2's regeneration gates. Each must FAIL on the condition it
claims to detect (gotcha 13), and a provenance key must not be able to fail
the content gate (gotcha 25a). Expectations are written down, not derived
from the module (gotcha 32b)."""
import json

from src.analysis import ct24_regen as R

OLD = {"git_sha": "5ed4c60", "git_dirty": False, "config_hash": "e0c28c157d8b18b0",
       "source_fasta": "/old/train.fasta", "timestamp_utc": "2026-07-24T15:08:21",
       "n_shards": 24, "max_len": 1022, "n_sequences": 521831,
       "total_residues_capped": 180416896, "shard_residues": [7517371, 7517370],
       "shard_sequences": [21743, 21742], "shard_paths": ["/s/shard_000.fasta"],
       "residue_imbalance": 3.99e-07}


def _new(**kw):
    d = json.loads(json.dumps(OLD))
    d.update(kw)
    return d


def test_shards_gate_passes_when_only_provenance_differs():
    new = _new(git_sha="2b4cd9b", config_hash="ffff", source_fasta="/archive/x",
               timestamp_utc="2026-10-23T00:00:00", git_dirty=True)
    assert R.gate_shards(new, OLD)["verdict"] == "PASS"


def test_shards_gate_fails_on_one_residue_count():
    new = _new(shard_residues=[7517371, 7517371])
    res = R.gate_shards(new, OLD)
    assert res["verdict"] == "FAIL"
    assert {r["key"]: r["status"] for r in res["keys"]}["shard_residues"] == "DIFFERS"


def test_shards_gate_fails_on_a_key_absent_from_the_new_manifest():
    new = _new()
    del new["n_sequences"]
    res = R.gate_shards(new, OLD)
    assert res["verdict"] == "FAIL"
    assert {r["key"]: r["status"] for r in res["keys"]}["n_sequences"] == "ABSENT_ON_NEW"


def test_shards_gate_fails_on_a_shard_path_change():
    assert R.gate_shards(_new(shard_paths=["/t/shard_000.fasta"]), OLD)["verdict"] == "FAIL"


META = {"n_rows": 180416896,
        "activations_attrs": {"dtype": "float16", "final_layer_norm_applied": False,
                              "layer": 24, "max_len": 1022,
                              "model": "facebook/esm2_t33_650M_UR50D", "seed": 0,
                              "subsampled": False}}


def test_cache_gate_passes_with_the_expected_new_device_attr():
    attrs = dict(META["activations_attrs"], device_class="rtx_5000_ada")
    res = R.gate_cache([90208448, 90208448], attrs, META)
    assert res["verdict"] == "PASS"
    st = {r["key"]: r["status"] for r in res["shard0_attrs"]["keys"]}
    assert st["device_class"] == "EXPECTED_NEW_ONLY"


def test_cache_gate_fails_on_one_row_short():
    res = R.gate_cache([90208448, 90208447], dict(META["activations_attrs"]), META)
    assert res["verdict"] == "FAIL" and res["rows_agree"] is False


def test_cache_gate_fails_on_a_different_layer():
    attrs = dict(META["activations_attrs"], layer=23)
    assert R.gate_cache([180416896], attrs, META)["verdict"] == "FAIL"


def test_cache_gate_fails_on_an_unexpected_new_attr():
    attrs = dict(META["activations_attrs"], residue_ids="x")
    assert R.gate_cache([180416896], attrs, META)["verdict"] == "FAIL"


def test_cache_gate_fails_when_meta_lacks_n_rows():
    assert R.gate_cache([1], {}, {"activations_attrs": {}})["verdict"] == "FAIL"


def test_bdec_gate():
    assert R.RECORDED_B_DEC_INIT == "f3185a6c22f91039"
    assert R.gate_bdec("f3185a6c22f91039")["verdict"] == "PASS"
    assert R.gate_bdec("f3185a6c22f91038")["verdict"] == "FAIL"
    assert R.gate_bdec(None)["verdict"] == "FAIL"


def test_cli_exit_codes(tmp_path):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps(OLD), encoding="utf-8")
    new.write_text(json.dumps(_new(git_sha="z")), encoding="utf-8")
    assert R.main(["shards", "--new", str(new), "--old", str(old),
                   "--out", str(tmp_path / "v.json")]) == 0
    new.write_text(json.dumps(_new(n_shards=23)), encoding="utf-8")
    assert R.main(["shards", "--new", str(new), "--old", str(old),
                   "--out", str(tmp_path / "v2.json")]) == 1
