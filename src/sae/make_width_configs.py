"""Generate one config per dictionary width for the CN width sweep.

WHY THIS EXISTS RATHER THAN A CLI FLAG. `train.py` has no `--dict-size`
override (amendment CM5), and per-width config FILES are the better fix
because `config_hash` hashes the file -- so the arms become distinguishable
by an identity key rather than being indistinguishable in `meta.json`.

WHY IT IS A SCRIPT RATHER THAN FOUR HAND-EDITS. `configs/mvp.yaml` contains
`dict_size` TWICE. Line ~92 is `sae.dict_size`, which governs training
(`train.py:472` reads `cfg["sae"]["dict_size"]`). Line ~203 is
`reference.dict_size`, InterPLM's released width -- an external fact of
record settled from the released weights (`docs/LITERATURE.md` section 6,
`ARTIFACT` tier). A blind substitution rewrites a comparator's declared
width. Every assertion below exists to make that impossible.

The edit is asserted in BOTH directions (gotcha 13's fourth extension: an
operation that can silently do nothing is not a verified operation).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# The anchor width is included deliberately: at 10240 the generated file must
# be BYTE-IDENTICAL to the source, which makes its config_hash equal to the
# primary ensemble's and is itself a check that the generator is inert when
# asked for a no-op.
DEFAULT_WIDTHS = (2560, 10240, 20480, 40960)


class RefusedEdit(RuntimeError):
    """Raised when the edit cannot be proven to have hit the right key."""


def rewrite_sae_dict_size(text: str, width: int, d_model: int | None = None) -> str:
    """Return `text` with ONLY `sae.dict_size` set to `width`.

    Refuses unless exactly one line is eligible and it lies inside the `sae:`
    block. Never touches `reference.dict_size`.

    If the line carries an `Nx expansion` comment and `d_model` is given, the
    comment is recomputed too. A config reading `dict_size: 40960  # 8x
    expansion` would be a stale assertion sitting in an authoritative file --
    the exact defect shape this repo has shipped before. At the anchor width
    the recomputed comment is unchanged, so byte-identity survives.
    """
    lines = text.splitlines(keepends=True)

    # Locate top-level block boundaries. A top-level key is a line starting at
    # column 0 with an identifier followed by ':'.
    block_of: list[str | None] = []
    current: str | None = None
    for ln in lines:
        stripped = ln.rstrip("\r\n")
        if stripped and not stripped[0].isspace() and not stripped.lstrip().startswith("#"):
            if ":" in stripped:
                current = stripped.split(":", 1)[0].strip()
        block_of.append(current)

    hits = [
        i
        for i, ln in enumerate(lines)
        if ln.lstrip().startswith("dict_size:") and block_of[i] == "sae"
    ]
    other = [
        i
        for i, ln in enumerate(lines)
        if ln.lstrip().startswith("dict_size:") and block_of[i] != "sae"
    ]

    if len(hits) != 1:
        raise RefusedEdit(
            f"expected exactly 1 `dict_size` inside the `sae:` block, found {len(hits)} "
            f"at lines {[i + 1 for i in hits]}"
        )
    if not other:
        raise RefusedEdit(
            "found no `dict_size` OUTSIDE the `sae:` block -- expected "
            "`reference.dict_size` to exist. Refusing, because that means the "
            "block detection is wrong, not that the file changed."
        )

    i = hits[0]
    old = lines[i]
    # Substitute ONLY the numeric value, preserving indentation, the padding
    # before any trailing comment, and the line ending. At the anchor width
    # this makes the rewrite a true byte-level no-op, so the generated config
    # inherits the primary ensemble's config_hash.
    new, n = re.subn(
        r"^(\s*dict_size:\s*)\d+", lambda m: f"{m.group(1)}{width}", old, count=1
    )
    if n != 1:
        raise RefusedEdit(f"could not substitute the value on line {i + 1}: {old!r}")

    if d_model:
        if width % d_model:
            raise RefusedEdit(
                f"width {width} is not a whole multiple of d_model {d_model}; "
                "refusing to write an expansion comment that would be wrong"
            )
        new, m = re.subn(r"(#\s*)\d+x expansion", rf"\g<1>{width // d_model}x expansion", new, count=1)
        if m and "expansion" not in new:
            raise RefusedEdit("expansion comment rewrite corrupted the line")
    lines[i] = new

    return "".join(lines)


def verify(src_text: str, out_text: str, width: int) -> dict[str, object]:
    """Assert the edit did what it claims, in both directions."""
    src = yaml.safe_load(src_text)
    out = yaml.safe_load(out_text)

    if out["sae"]["dict_size"] != width:
        raise RefusedEdit(f"sae.dict_size is {out['sae']['dict_size']!r}, expected {width}")

    # The comparator's width is untouched.
    if out["reference"]["dict_size"] != src["reference"]["dict_size"]:
        raise RefusedEdit(
            f"reference.dict_size moved from {src['reference']['dict_size']!r} to "
            f"{out['reference']['dict_size']!r} -- REFUSED (InterPLM's ARTIFACT-tier width)"
        )

    # Nothing else moved. Compare the full parsed trees with sae.dict_size masked.
    a, b = yaml.safe_load(src_text), yaml.safe_load(out_text)
    a["sae"]["dict_size"] = b["sae"]["dict_size"] = None
    if a != b:
        raise RefusedEdit("a key other than sae.dict_size changed -- REFUSED")

    # Exactly one text line differs (or zero, for the no-op anchor width).
    sl, ol = src_text.splitlines(), out_text.splitlines()
    if len(sl) != len(ol):
        raise RefusedEdit(f"line count moved {len(sl)} -> {len(ol)}")
    diff = [i for i, (x, y) in enumerate(zip(sl, ol)) if x != y]
    expected = 0 if src["sae"]["dict_size"] == width else 1
    if len(diff) != expected:
        raise RefusedEdit(
            f"expected {expected} changed line(s), found {len(diff)} at "
            f"{[i + 1 for i in diff]}"
        )
    return {
        "width": width,
        "changed_lines": [i + 1 for i in diff],
        "byte_identical": src_text == out_text,
        "reference_dict_size": out["reference"]["dict_size"],
    }


def self_test(src_text: str) -> None:
    """Prove the guards can FAIL before trusting a pass (gotcha 13)."""
    checks: list[tuple[str, bool]] = []

    # 1. A substitution that hits the reference block must be refused.
    naive = src_text.replace("dict_size: 10240", "dict_size: 40960")
    try:
        verify(src_text, naive, 40960)
        checks.append(("naive global replace is refused", False))
    except RefusedEdit as e:
        checks.append((f"naive global replace is refused ({str(e)[:44]}...)", True))

    # 2. A no-op must be refused when a real change was requested.
    try:
        verify(src_text, src_text, 40960)
        checks.append(("silent no-op is refused", False))
    except RefusedEdit:
        checks.append(("silent no-op is refused", True))

    # 3. An unrelated collateral edit must be refused.
    collateral = rewrite_sae_dict_size(src_text, 40960).replace("k: 32", "k: 64", 1)
    try:
        verify(src_text, collateral, 40960)
        checks.append(("collateral edit to another key is refused", False))
    except RefusedEdit:
        checks.append(("collateral edit to another key is refused", True))

    # 4. The legitimate edit must PASS -- a guard that only fails is useless
    #    (gotcha 25).
    try:
        verify(src_text, rewrite_sae_dict_size(src_text, 40960), 40960)
        checks.append(("the legitimate edit passes", True))
    except RefusedEdit as e:
        checks.append((f"the legitimate edit passes -- {e}", False))

    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in checks):
        raise SystemExit("SELF-TEST FAILED -- the guards do not discriminate; refusing to write")
    print("  self-test: guards discriminate in both directions")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("configs/width"))
    p.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    p.add_argument("--self-test-only", action="store_true")
    args = p.parse_args()

    src_text = args.config.read_text()

    print(f"self-test against {args.config}:")
    self_test(src_text)
    if args.self_test_only:
        return

    src = yaml.safe_load(src_text)
    print(f"\nsource sae.dict_size={src['sae']['dict_size']} "
          f"k={src['sae']['k']} d_model={src['model']['d_model']} "
          f"reference.dict_size={src['reference']['dict_size']}")

    d_model = src["model"]["d_model"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for w in args.widths:
        out_text = rewrite_sae_dict_size(src_text, w, d_model)
        info = verify(src_text, out_text, w)
        # The expansion comment must state THIS width's factor, not the
        # source's. A stale comment in an authoritative file is a defect.
        line = next(
            ln for ln in out_text.splitlines()
            if ln.lstrip().startswith("dict_size:") and str(w) in ln
        )
        if "expansion" in line:
            want = f"{w // d_model}x expansion"
            if want not in line:
                raise SystemExit(f"stale expansion comment for width {w}: {line!r}")
            info["expansion_comment"] = want
        dest = args.out_dir / f"mvp_w{w}.yaml"
        dest.write_text(out_text, newline="")
        print(f"  wrote {dest}  {info}")


if __name__ == "__main__":
    sys.exit(main())
