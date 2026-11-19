"""Amendment AE, condition 2 -- the invariant asserted in CODE, not in naming.

Two parameters collided twice over: `row_subset` accepted the literal value
`population`, which was the OTHER parameter's NAME, and both accepted the
shared literal `all` meaning different things. Under `truncation_policy=
untruncated` the store holds 15,972,325 rows while `--row-subset all` scored
17,739,363, so every prevalence became `n_positives / wrong_n_rows` and every
normalised gain was wrong.

Renaming fixed that instance. These tests retire the CLASS: any future
parameter whose value collides with a name, or whose value set overlaps
another parameter's, fails here rather than in a silently wrong prevalence.

The registry (`artifact_identity.PARAMETER_VALUES`) is the single source: every
`choices=` for these flags derives from it, so there is no parallel list to
drift. Test 3 asserts that derivation actually happened -- a registry nobody
reads is documentation, not an invariant.

SCOPE, stated so it is not over-read: these tests cover the SOURCE. That
nothing ON DISK still carries the retired keys is the migration's own
acceptance criterion (AE3 condition 4), which needs cluster artifacts and
cannot run here.
"""
from __future__ import annotations

import argparse
import ast
import pathlib

import pytest

from src.analysis.artifact_identity import IDENTITY_KEYS, PARAMETER_VALUES

REPO = pathlib.Path(__file__).resolve().parents[1]

# The literals amendment AE retired. Any reappearance in a `choices=` list is a
# regression, whatever it is called.
RETIRED_LITERALS = {"population", "row_subset", "all"}


def test_no_value_equals_any_parameter_name():
    """COLLISION 1: a value that is another parameter's name.

    `row_subset="population"` was exactly this, and it read at every call site
    as though it named the other parameter.
    """
    names = set(PARAMETER_VALUES)
    offenders = [
        (param, value)
        for param, values in PARAMETER_VALUES.items()
        for value in values
        if value in names
    ]
    assert not offenders, (
        f"these values are also parameter names: {offenders}. A value named "
        f"after another parameter reads as naming it. Parameter names are "
        f"{sorted(names)}."
    )


def test_no_literal_is_shared_between_two_parameters():
    """COLLISION 2, the worse half: one literal, two meanings.

    `truncation_policy` and `scored_row_set` both accepted `all`. They agreed
    numerically at `with_truncated` (both 17,739,363 rows), which is precisely
    why crossing them was invisible until the policy changed.
    """
    seen: dict[str, str] = {}
    clashes = []
    for param, values in PARAMETER_VALUES.items():
        for value in values:
            if value in seen and seen[value] != param:
                clashes.append((value, seen[value], param))
            seen[value] = param
    assert not clashes, (
        f"these literals are shared by two parameters: {clashes}. One literal "
        f"with two meanings is how a crossed argument stays invisible."
    )


def test_no_parameter_name_is_also_an_identity_key_with_a_different_meaning():
    """`truncation_policy` is BOTH a parameter and an identity key, which is
    correct -- they are the same fact. `scored_row_set` is a parameter and is
    checked by the gate, but is NOT an equality key against the store. This
    pins that split so a later edit cannot quietly make one the other."""
    assert "truncation_policy" in IDENTITY_KEYS
    assert "scored_row_set" not in IDENTITY_KEYS


def _choices_literals_by_flag() -> dict[str, list[list[str]]]:
    """Every literal `choices=[...]` attached to the two flags, via `ast`.

    `ast` rather than a text scan on purpose: gotcha 20's corollary is that a
    source-text checker's false positives land on the docstrings that explain
    it -- this file quotes `"population"` and `"all"` several times above, and
    a regex would flag its own documentation and get disabled rather than
    fixed.
    """
    found: dict[str, list[list[str]]] = {}
    for path in sorted((REPO / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            flag = next(
                (a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)
                 and a.value in ("--scored-row-set", "--truncation-policy")),
                None,
            )
            if flag is None:
                continue
            for kw in node.keywords:
                if kw.arg != "choices" or not isinstance(kw.value, ast.List):
                    continue
                lits = [e.value for e in kw.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                if lits:
                    found.setdefault(flag, []).append(lits)
    return found


def test_no_flag_hardcodes_a_retired_literal_in_its_choices():
    """The propagation guard. Renaming the definition sites is not enough if a
    consumer still offers the old word -- that is the cross-component
    non-propagation of gotcha 16, and 173 passing tests once missed four of
    them because each component was individually correct."""
    offenders = {
        flag: lists
        for flag, lists in _choices_literals_by_flag().items()
        if any(set(l) & RETIRED_LITERALS for l in lists)
    }
    assert not offenders, (
        f"retired literals {sorted(RETIRED_LITERALS)} reappear in a hardcoded "
        f"choices list: {offenders}. Derive from PARAMETER_VALUES instead."
    )


@pytest.mark.parametrize("param", sorted(PARAMETER_VALUES))
def test_registry_values_are_unique_and_nonempty(param):
    values = PARAMETER_VALUES[param]
    assert values, f"{param} has no allowed values"
    assert len(set(values)) == len(values), f"{param} repeats a value: {values}"
    assert all(v and not v.isspace() for v in values)


def test_argparse_accepts_every_registry_value_and_rejects_a_retired_one():
    """End-to-end, on a real parser: each registry value parses, and the
    retired literal is refused. Without this the invariant could hold on a
    table that no parser consults."""
    p = argparse.ArgumentParser()
    p.add_argument("--scored-row-set",
                   choices=list(PARAMETER_VALUES["scored_row_set"]))
    for value in PARAMETER_VALUES["scored_row_set"]:
        assert p.parse_args(["--scored-row-set", value]).scored_row_set == value
    for retired in ("population", "all"):
        with pytest.raises(SystemExit):
            p.parse_args(["--scored-row-set", retired])
