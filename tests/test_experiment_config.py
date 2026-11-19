"""Amendment CR §CR1 -- the one helper through which the CLI-default rows
resolve. Three outcomes for `resolve`, one for an absent key, and the repo's
own config carries the values the old literals had (written down here, not
read back from the code under test -- gotcha 32b)."""
from __future__ import annotations

import pytest

from src.experiment_config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    lookup,
    resolve,
)

CFG = {"model": {"max_len": 500, "layer": 7, "name": "toy/model"}}


def test_no_cli_value_takes_the_config_value():
    assert resolve(CFG, "model.max_len", None, "--max-len", "cfg.yaml") == 500


def test_an_equal_cli_value_is_accepted():
    assert resolve(CFG, "model.max_len", 500, "--max-len", "cfg.yaml") == 500


def test_a_disagreeing_cli_value_is_refused_naming_both_values():
    with pytest.raises(SystemExit) as e:
        resolve(CFG, "model.max_len", 1022, "--max-len", "cfg.yaml")
    msg = str(e.value)
    assert "--max-len" in msg and "1022" in msg and "500" in msg
    assert "model.max_len" in msg and "cfg.yaml" in msg


def test_an_absent_key_is_a_refusal_not_a_default():
    with pytest.raises(SystemExit) as e:
        lookup(CFG, "model.d_model", "cfg.yaml")
    assert "model.d_model" in str(e.value) and "cfg.yaml" in str(e.value)
    with pytest.raises(SystemExit):
        resolve(CFG, "annotations.subtype_note_types", None, "-", "cfg.yaml")


def test_load_config_refuses_a_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        load_config(tmp_path / "nope.yaml")


def test_the_repository_config_carries_the_old_literals():
    """The values the four scripts hardcoded until CR, so a no-flag run is
    unchanged. Written down, not derived."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert lookup(cfg, "model.max_len", DEFAULT_CONFIG_PATH) == 1022
    assert lookup(cfg, "model.layer", DEFAULT_CONFIG_PATH) == 24
    assert (lookup(cfg, "model.name", DEFAULT_CONFIG_PATH)
            == "facebook/esm2_t33_650M_UR50D")
    assert lookup(cfg, "confound_control.analysis_min_firing_rows",
                  DEFAULT_CONFIG_PATH) == 200
