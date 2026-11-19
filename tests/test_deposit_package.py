"""scripts/deposit_package.py: the checksum and asset lists are written only
from a bundle that verifies against its manifest (§CT18-D2)."""
import csv
import hashlib

import pytest

from scripts import deposit_package as dp


def _bundle(d):
    d.mkdir()
    files = {"a.csv.gz": b"aaa", "b.json": b"{}"}
    for n, b in files.items():
        (d / n).write_bytes(b)
    with open(d / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256", "producing_commit", "owning_nor_row"])
        for n, b in files.items():
            w.writerow([f"release/derived/{n}", hashlib.sha256(b).hexdigest(), "c" * 40, "§1 | §2"])
    return files


def test_writes_every_row_and_the_manifest(tmp_path):
    files = _bundle(tmp_path / "b")
    s = dp.build(tmp_path / "b", tmp_path / "z")
    sums = (tmp_path / "z" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert sums[:2] == [f"{hashlib.sha256(b).hexdigest()}  {n}" for n, b in files.items()]
    assert sums[2].endswith("  MANIFEST.csv") and s["n_assets"] == 3
    md = (tmp_path / "z" / "ASSETS.md").read_text(encoding="utf-8")
    assert "`a.csv.gz`" in md and "§1 / §2" in md          # a '|' in a row would break the table


@pytest.mark.parametrize("mutate", ["alter", "remove"])
def test_refuses_and_writes_nothing_on_a_bad_file(tmp_path, mutate):
    _bundle(tmp_path / "b")
    if mutate == "alter":
        (tmp_path / "b" / "b.json").write_bytes(b"{ }")
    else:
        (tmp_path / "b" / "b.json").unlink()
    with pytest.raises(SystemExit, match="REFUSING"):
        dp.build(tmp_path / "b", tmp_path / "z")
    assert not (tmp_path / "z").exists()


def _upload_inputs(tmp_path):
    _bundle(tmp_path / "b")
    dm = tmp_path / "decoders_MANIFEST.csv"
    dm.write_text("asset,sha256,bytes,n_dictionaries,seed_labels,source_dirs,producing_commit\n"
                  f"decoders_x.h5,{'f' * 64},5,1,\"[0]\",/p,{'c' * 40}\n", encoding="utf-8")
    sums = tmp_path / "sums.tsv"
    sums.write_text(f"asset\tbytes\tsha256\tmd5\ndecoders_x.h5\t5\t{'f' * 64}\t{'e' * 32}\n",
                    encoding="utf-8")
    env = tmp_path / "env.yml"
    env.write_text("name: x\n", encoding="utf-8")
    return dm, sums, env


def test_upload_set_tar_is_deterministic_and_round_trips(tmp_path):
    import tarfile
    dm, sums, env = _upload_inputs(tmp_path)
    s = dp.build_upload(tmp_path / "b", dm, sums, env, tmp_path / "u1", tmp_path / "z1")
    import os
    os.utime(tmp_path / "b" / "a.csv.gz", (1_700_000_000, 1_700_000_000))   # a later copy's mtime
    dp.build_upload(tmp_path / "b", dm, sums, env, tmp_path / "u2", tmp_path / "z2")
    assert (tmp_path / "u1" / "release_derived.tar").read_bytes() == \
        (tmp_path / "u2" / "release_derived.tar").read_bytes()
    with tarfile.open(tmp_path / "u1" / "release_derived.tar") as tf:
        names = tf.getnames()
        assert names == sorted(names) and "release_derived/MANIFEST.csv" in names
        assert tf.extractfile("release_derived/a.csv.gz").read() == b"aaa"
        assert {m.mtime for m in tf.getmembers()} == {0}
    rows = (tmp_path / "z1" / "DATA_UPLOAD_CHECKSUMS.tsv").read_text(encoding="utf-8").splitlines()
    assert s["n_files"] == 5 and rows[-1].startswith("decoders_x.h5\t5\t" + "e" * 32)


def test_the_upload_set_is_the_bundle_the_manifests_the_environment_and_the_decoders(tmp_path):
    """The data record's upload set has no other file. The project record is deposited on its own,
    with restricted files, and the packaging takes no archive of it (section CT18-D17)."""
    import inspect
    dm, sums, env = _upload_inputs(tmp_path)
    s = dp.build_upload(tmp_path / "b", dm, sums, env, tmp_path / "u", tmp_path / "z")
    rows = [r.split("\t") for r in
            (tmp_path / "z" / "DATA_UPLOAD_CHECKSUMS.tsv").read_text(encoding="utf-8").splitlines()]
    assert s["n_files"] == 5 and [r[0] for r in rows[1:]] == [
        "release_derived.tar", "derived_MANIFEST.csv", "decoders_MANIFEST.csv", "reference_environment.yml",
        "decoders_x.h5"]
    assert sorted(f.name for f in (tmp_path / "u").iterdir()) == [
        "decoders_MANIFEST.csv", "derived_MANIFEST.csv", "reference_environment.yml", "release_derived.tar"]
    assert list(inspect.signature(dp.build_upload).parameters) == [
        "bundle", "decoders_manifest", "decoder_sums", "env_file", "upload_dir", "out"]


def test_upload_set_refuses_a_decoder_sum_that_disagrees(tmp_path):
    dm, sums, env = _upload_inputs(tmp_path)
    sums.write_text(f"asset\tbytes\tsha256\tmd5\ndecoders_x.h5\t5\t{'0' * 64}\t{'e' * 32}\n",
                    encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the manifest"):
        dp.build_upload(tmp_path / "b", dm, sums, env, tmp_path / "u", tmp_path / "z")
