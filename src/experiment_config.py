"""
Experiment-defining values come from `configs/mvp.yaml`, and a CLI flag may only
AGREE with the config, never override it (PREREGISTRATION amendment CR, §CR1).

Why this module exists. `CLAUDE.md`'s locked decision is that nothing
experiment-defining is hardcoded in `src/`; until CR, four scripts carried
`1022` / `24` / the model name as argparse DEFAULTS, so a config change to
`model.max_len` would have changed nothing they did (`docs/DATAFLOW.md` §4,
stage 2/3 row). The scripts now default the flag to `None` and route it through
`resolve`, which has exactly three outcomes:

  * no CLI value            -> the config's value;
  * CLI value == config     -> accepted (the existing sbatch files pass
                               `--layer 24` explicitly, and keep working);
  * CLI value != config     -> `SystemExit`, printing both values and where
                               each came from. Presence never overrides.

An ABSENT config key is a `SystemExit` naming the key, never a default: the
hazard CR closes for `subtype_note_types` was a fallback that governed silently
when the key was missing (§CR1 Row B), and this helper does not reintroduce that
shape for any other key.

`DEFAULT_CONFIG_PATH` is the repository's own `configs/mvp.yaml`, resolved from
this file's location, for the two entry points (`cache.py`, `design3_join.py`)
that had no `--config` flag and whose sbatch call sites pass none (§CR0 item 4).
A path is plumbing, not an experiment-defining value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "mvp.yaml"


def load_config(path: Path) -> dict:
    """Parse the YAML config. A missing file is a `SystemExit`, not a
    `FileNotFoundError` traceback, so an sbatch log shows one line."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"FATAL: config file {path} does not exist.")
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"FATAL: config file {path} did not parse to a mapping.")
    return cfg


def lookup(cfg: dict, dotted_key: str, config_path: Path | str) -> Any:
    """`cfg["a"]["b"]["c"]` for `"a.b.c"`. An absent key at any level is a
    `SystemExit` that names the key and the file; there is no default."""
    node: Any = cfg
    walked: list[str] = []
    for part in dotted_key.split("."):
        walked.append(part)
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(
                f"FATAL: config key `{dotted_key}` is absent from {config_path} "
                f"(missing at `{'.'.join(walked)}`). Experiment-defining values "
                "are read from the config and are never defaulted in code "
                "(amendment CR).")
        node = node[part]
    return node


def resolve(cfg: dict, dotted_key: str, cli_value: Any, flag: str,
            config_path: Path | str) -> Any:
    """The config's value for `dotted_key`, unless `cli_value` is given, in
    which case it must EQUAL the config's value or the run is refused."""
    cfg_value = lookup(cfg, dotted_key, config_path)
    if cli_value is None:
        return cfg_value
    if cli_value != cfg_value:
        raise SystemExit(
            f"FATAL: {flag} {cli_value!r} disagrees with `{dotted_key}` = "
            f"{cfg_value!r} in {config_path}. The config is the only place an "
            "experiment-defining value may be set; a CLI value may restate it "
            "but not override it (amendment CR). Either drop the flag or change "
            "the config for a declared reason.")
    return cli_value
