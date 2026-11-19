"""Amendment CT, §CT8 item 4: the `--truncation-policy` help states the
population the code builds -- at most 1,022 residues -- and does not equate it
with InterPLM's < 1,024 (corrections entry 269). Run through the real entry
point, because the parser is built inside `main()`."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _help() -> str:
    env = dict(os.environ, PYTHONUTF8="1", COLUMNS="400")
    r = subprocess.run([sys.executable, "-m", "src.annotations.build_residue_labels",
                        "--help"], cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    return " ".join(r.stdout.split())


def test_truncation_policy_help_says_at_most_1022():
    h = _help()
    assert "'untruncated' means <= 1,022 residues" in h
    assert "not the same set" in h


def test_truncation_policy_help_no_longer_equates_the_two_populations():
    h = _help()
    assert "This is InterPLM's own population (<1024 residues)" not in h
