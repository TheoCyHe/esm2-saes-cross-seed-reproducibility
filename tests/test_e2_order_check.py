"""§CT18-D item 3: scripts/e2_order_check.py on a toy history, every verdict
produced (gotcha 25: the instrument can return more than one outcome)."""
import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e2", REPO / "scripts" / "e2_order_check.py")
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)


def _git(d, *a):
    return subprocess.run(["git", *a], cwd=d, capture_output=True, text=True, check=True).stdout.strip()


def _toy(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@example.invalid")
    _git(d, "config", "user.name", "t")
    p = d / "PREREGISTRATION.md"
    p.write_text("# P\n\n## Amendment AR — the join\n", encoding="utf-8")
    _git(d, "add", "."); _git(d, "commit", "-q", "-m", "AR")
    c_ar = _git(d, "rev-parse", "HEAD")
    p.write_text(p.read_text(encoding="utf-8") + "\n### CT19. CARRIERS\n", encoding="utf-8")
    _git(d, "add", "."); _git(d, "commit", "-q", "-m", "CT19")
    c_ct19 = _git(d, "rev-parse", "HEAD")
    (d / "x.txt").write_text("run", encoding="utf-8")
    _git(d, "add", "."); _git(d, "commit", "-q", "-m", "run")
    c_run = _git(d, "rev-parse", "HEAD")
    return d, c_ar, c_ct19, c_run


def _art(tmp_path, name, sha, dec):
    f = tmp_path / name
    f.write_text(json.dumps({"identity": {"git_sha": sha, "governing_amendment": dec}}), encoding="utf-8")
    return f


def test_every_verdict(tmp_path):
    d, c_ar, c_ct19, c_run = _toy(tmp_path)
    idx = e2.heading_index((d / "PREREGISTRATION.md").read_text(encoding="utf-8"))
    assert set(idx) == {"AR", "CT19"}
    cache = {}
    v = lambda f: e2.check(f, idx, cache, cwd=d)["verdict"]          # noqa: E731
    assert v(_art(tmp_path, "a.json", c_run, "CT19 (the brief's CT11)")) == "PRECEDES"
    assert v(_art(tmp_path, "b.json", c_ct19, "CT19")) == "SAME_COMMIT"
    assert v(_art(tmp_path, "c.json", c_ar, "AR (the join) + CT19")) == "DOES_NOT_PRECEDE"
    assert v(_art(tmp_path, "d.json", c_run, "ZZ9 unknown")) == "UNDETERMINED"
    assert v(_art(tmp_path, "e.json", None, "CT19")) == "UNDETERMINED"
    assert v(_art(tmp_path, "f.json", "0" * 40, "CT19")) == "UNDETERMINED"
    f = tmp_path / "h.json"                       # declared under `specification`
    f.write_text(json.dumps({"identity": {"git_sha": c_run, "specification":
                                          "PREREGISTRATION amendment CT, §CT19"}}), encoding="utf-8")
    assert v(f) == "PRECEDES"
    r = e2.check(_art(tmp_path, "g.json", c_run, "AR + CT19"), idx, cache, cwd=d)
    assert r["declaring_section"] == "CT19"                    # the latest of the declared sections
