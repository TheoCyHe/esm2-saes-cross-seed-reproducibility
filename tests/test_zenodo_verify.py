"""scripts/zenodo_verify.py: the draft's file list against the upload table, in
both listing forms Zenodo's docs show; every kind of difference refuses."""
import json

import pytest

from scripts import zenodo_verify as zv

TABLE = "file\tbytes\tmd5\tsha256\na.tar\t10\t" + "a" * 32 + "\t" + "0" * 64 + "\nb.h5\t20\t" + "b" * 32 + "\t" + "1" * 64 + "\n"
LEGACY = [{"filename": "a.tar", "filesize": 10, "checksum": "a" * 32},
          {"filename": "b.h5", "filesize": 20, "checksum": "B" * 32}]
NEW = {"entries": [{"key": "a.tar", "size": 10, "checksum": "md5:" + "a" * 32},
                   {"key": "b.h5", "size": 20, "checksum": "md5:" + "b" * 32}]}


def _run(tmp_path, listing):
    t = tmp_path / "t.tsv"
    t.write_text(TABLE, encoding="utf-8")
    j = tmp_path / "l.json"
    j.write_text(json.dumps(listing), encoding="utf-8")
    return zv.main(["--from-json", str(j), "--table", str(t)])


@pytest.mark.parametrize("listing", [LEGACY, NEW])
def test_both_forms_match(tmp_path, listing):
    assert _run(tmp_path, listing) == 0


@pytest.mark.parametrize("mutate", ["md5", "size", "absent", "extra", "sha1"])
def test_every_difference_refuses(tmp_path, mutate):
    lst = [dict(e) for e in LEGACY]
    if mutate == "md5":
        lst[1]["checksum"] = "c" * 32
    elif mutate == "size":
        lst[0]["filesize"] = 11
    elif mutate == "absent":
        lst = lst[:1]
    elif mutate == "extra":
        lst.append({"filename": "c.txt", "filesize": 1, "checksum": "d" * 32})
    if mutate == "sha1":
        lst[0]["checksum"] = "sha1:" + "e" * 40
        with pytest.raises(SystemExit, match="not md5"):
            _run(tmp_path, lst)
        return
    assert _run(tmp_path, lst) == 1


def test_the_token_is_required_for_a_fetch(monkeypatch):
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="ZENODO_TOKEN is not set"):
        zv.fetch(1, sandbox=True)
