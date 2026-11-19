"""§CT29-F item 1: each grid config is configs/mvp.yaml with only `k` (line 98)
and, off width 10,240, `dict_size` (line 97) changed. Written-down expectations
(gotcha 32b): the differing lines are spelled out here, not derived."""
from pathlib import Path

import pytest
import yaml

EXPECTED = {
    "k64_w2560": [(97, b"  dict_size: 10240          # 8x expansion",
                   b"  dict_size: 2560          # 2x expansion"),
                  (98, b"  k: 32", b"  k: 64")],
    "k64_w40960": [(97, b"  dict_size: 10240          # 8x expansion",
                    b"  dict_size: 40960          # 32x expansion"),
                   (98, b"  k: 32", b"  k: 64")],
    "k16_w10240": [(98, b"  k: 32", b"  k: 16")],
    "k128_w10240": [(98, b"  k: 32", b"  k: 128")],
}
PARSED = {"k64_w2560": (64, 2560), "k64_w40960": (64, 40960),
          "k16_w10240": (16, 10240), "k128_w10240": (128, 10240)}


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_grid_config_differs_from_mvp_only_on_declared_lines(arm):
    a = Path("configs/mvp.yaml").read_bytes().split(b"\n")
    b = Path(f"configs/grid/mvp_{arm}.yaml").read_bytes().split(b"\n")
    assert len(a) == len(b)
    diff = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert diff == EXPECTED[arm]
    cfg = yaml.safe_load(Path(f"configs/grid/mvp_{arm}.yaml").read_text(encoding="utf-8"))
    assert (cfg["sae"]["k"], cfg["sae"]["dict_size"]) == PARSED[arm]
    assert cfg["reference"]["dict_size"] == 10240


def test_launcher_assertion_matches_the_written_expectation():
    """The launcher builds its expected diff from (k, width); it must produce
    exactly the lines written down above for every arm, and refuse a config
    that changes anything else."""
    import re
    text = Path("slurm/ct29_train_array.sbatch").read_text(encoding="utf-8")
    body = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF", text, re.S).group(1)
    for arm, (k, w) in PARSED.items():
        ns = {"__name__": "x"}
        import sys
        old = sys.argv
        try:
            sys.argv = ["x", f"configs/grid/mvp_{arm}.yaml", str(k), str(w)]
            exec(compile(body, "launcher", "exec"), ns)
        finally:
            sys.argv = old
    # A mutant config with one more differing line must be refused.
    import tempfile, os, sys
    src = Path("configs/grid/mvp_k16_w10240.yaml").read_bytes().split(b"\n")
    src[100] = src[100] + b"  # mutated"
    with tempfile.NamedTemporaryFile("wb", suffix=".yaml", delete=False) as f:
        f.write(b"\n".join(src)); path = f.name
    try:
        old = sys.argv
        sys.argv = ["x", path, "16", "10240"]
        with pytest.raises(AssertionError):
            exec(compile(body, "launcher", "exec"), {"__name__": "x"})
    finally:
        sys.argv = old
        os.unlink(path)
