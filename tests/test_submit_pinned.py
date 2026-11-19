"""scripts/submit_pinned.sh (§CT18-D4 item 6, gotcha 47): a broken clone check
stops the submission before `sbatch` is called. A stand-in `sbatch` on PATH
records each call; the clone source is a throwaway local repository."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

BASH = shutil.which("bash")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "submit_pinned.sh"
pytestmark = pytest.mark.skipif(BASH is None or shutil.which("git") is None,
                                reason="needs bash and git")


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def world(tmp_path):
    origin = tmp_path / "origin"
    (origin / "src").mkdir(parents=True)
    _git(origin, "init", "-q")
    _git(origin, "config", "user.email", "t@example.org")
    _git(origin, "config", "user.name", "t")
    (origin / "src" / "x.py").write_text("class trace_inputs:\n    pass\n", encoding="utf-8")
    _git(origin, "add", ".")
    _git(origin, "commit", "-q", "-m", "one")
    sha = _git(origin, "rev-parse", "HEAD")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "sbatch.log"
    (bin_ / "sbatch").write_text(f'#!/bin/bash\necho "$*" >> "{log.as_posix()}"\n',
                                 encoding="utf-8", newline="\n")
    (bin_ / "sbatch").chmod(0o755)
    return {"tmp": tmp_path, "origin": origin, "sha": sha, "bin": bin_, "log": log}


def _run(w, sha, checks, clone="clone", extra_env=None):
    env = dict(os.environ, REPO_URL=w["origin"].as_posix(),
               PATH=w["bin"].as_posix() + os.pathsep + os.environ.get("PATH", ""))
    env.update(extra_env or {})
    cmd = [BASH, SCRIPT.as_posix(), sha, (w["tmp"] / clone).as_posix(), *checks,
           "--", "sbatch", "job.sbatch"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    calls = w["log"].read_text(encoding="utf-8").splitlines() if w["log"].exists() else []
    return r.returncode, calls, r.stderr


def test_a_passing_check_submits(world):
    rc, calls, err = _run(world, world["sha"], ["src/x.py:^class trace_inputs"])
    assert rc == 0 and calls == ["job.sbatch"], err


@pytest.mark.parametrize("sha,checks,why", [
    ("0" * 40, ["src/x.py:^class trace_inputs"], "checkout of"),
    (None, ["src/x.py:^def trace_inputs"], "no match"),        # the fourth instance's pattern
    (None, ["src/absent.py:x"], "absent"),
    (None, ["src/x.py"], "not <file>:<ERE>"),
])
def test_a_broken_check_stops_before_sbatch(world, sha, checks, why):
    rc, calls, err = _run(world, sha or world["sha"], checks)
    assert rc != 0 and calls == [] and why in err, err


def test_a_dirty_clone_stops_before_sbatch(world):
    """A clone template whose post-checkout hook leaves an untracked file."""
    tpl = world["tmp"] / "template"
    (tpl / "hooks").mkdir(parents=True)
    hook = tpl / "hooks" / "post-checkout"
    hook.write_text("#!/bin/sh\necho stray > stray.txt\n", encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    rc, calls, err = _run(world, world["sha"], ["src/x.py:^class trace_inputs"],
                          extra_env={"GIT_TEMPLATE_DIR": tpl.as_posix()})
    assert rc != 0 and calls == [] and "dirty" in err, err


def test_an_existing_clone_dir_is_refused(world):
    (world["tmp"] / "clone").mkdir()
    rc, calls, err = _run(world, world["sha"], ["src/x.py:^class trace_inputs"])
    assert rc != 0 and calls == [] and "exists" in err, err
