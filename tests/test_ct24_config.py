"""configs/k64/mvp_k64.yaml differs from configs/mvp.yaml on exactly one line,
sae.k 32 -> 64 (§CT24-F item 1). Written-down expectation (gotcha 32b)."""
from pathlib import Path

import yaml


def test_k64_config_differs_from_mvp_on_exactly_sae_k():
    a = Path("configs/mvp.yaml").read_bytes().split(b"\n")
    b = Path("configs/k64/mvp_k64.yaml").read_bytes().split(b"\n")
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert diff == [(b"  k: 32", b"  k: 64")]
    cfg = yaml.safe_load(Path("configs/k64/mvp_k64.yaml").read_text(encoding="utf-8"))
    assert cfg["sae"]["k"] == 64
    assert cfg["sae"]["dict_size"] == 10240
    assert cfg["reference"]["dict_size"] == 10240
