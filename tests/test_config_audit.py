"""Tests for the config <-> pre-registration audit (amendment AB decision B).

THE FOUNDING CASE IS A TEST HERE. `metrics.feature_level.similarity` read
`min(activation_correlation, decoder_cosine)` for the entire life of the
project while the pre-registration specified decoder cosine alone. The first
test below restores that exact value and requires the audit to FAIL on it. If
that test ever passes, this module has stopped doing the only job it was
built for.

Mutations run and killed (see the AB commit message):
  * `audit` returning verdict PASS unconditionally
        -> kills every failure test
  * the uncovered-key sweep deleted
        -> kills test_a_new_ungoverned_key_is_a_failure
  * ABSENT treated as agreement
        -> kills test_absent_required_key_is_a_failure
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.analysis.config_audit import (
    AUDITED_SECTIONS,
    REGISTRY,
    UNGOVERNED,
    audit,
    flatten,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "mvp.yaml"


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CFG.read_text())


def test_the_real_config_passes(cfg):
    rep = audit(cfg)
    assert rep["verdict"] == "PASS", rep["problems"]


def test_the_founding_defect_is_caught(cfg):
    """The exact value that sat in the config unchallenged for the whole
    project. If this does not FAIL, the audit is decorative."""
    bad = copy.deepcopy(cfg)
    bad["metrics"]["feature_level"]["similarity"] = \
        "min(activation_correlation, decoder_cosine)"
    rep = audit(bad)
    assert rep["verdict"] == "FAIL"
    assert any("feature_level.similarity" in p for p in rep["problems"])


@pytest.mark.parametrize("path,bad_value", [
    ("model.layer", 25),
    ("sae.dict_size", 4096),
    ("sae.k", 64),
    ("sae.n_seeds", 40),
    ("sae.b_dec_trainable", False),
    ("data.cluster_members", "representatives"),
    ("metrics.primary", "auroc"),
    ("metrics.feature_level.matching", "argmax_cosine"),
    ("confound_control.caliper", 0.5),
])
def test_each_governed_key_is_actually_checked(cfg, path, bad_value):
    """Gotcha 13 extension 3: verify each branch fails FOR ITS OWN REASON, not
    because some other check happened to fire. Every one of these is a locked
    decision that a config edit must not be able to make quietly."""
    bad = copy.deepcopy(cfg)
    node = bad
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = bad_value
    rep = audit(bad)
    assert rep["verdict"] == "FAIL"
    assert any(path in p for p in rep["problems"]), rep["problems"]


def test_absent_required_key_is_a_failure(cfg):
    """Gotcha 15: an absent key is reported absent, never treated as
    matching."""
    bad = copy.deepcopy(cfg)
    del bad["metrics"]["feature_level"]["similarity"]
    rep = audit(bad)
    assert rep["verdict"] == "FAIL"
    row = next(r for r in rep["rows"]
               if r["path"] == "metrics.feature_level.similarity")
    assert row["status"] == "ABSENT"


def test_a_new_ungoverned_key_is_a_failure(cfg):
    """Coverage is the difference between a checklist and a gate: adding an
    experiment-defining key without deciding its provenance must break."""
    bad = copy.deepcopy(cfg)
    bad["sae"]["some_new_knob"] = 3
    rep = audit(bad)
    assert rep["verdict"] == "FAIL"
    assert "sae.some_new_knob" in rep["uncovered_keys"]


def test_every_config_key_is_governed_or_explicitly_ungoverned(cfg):
    """The coverage invariant itself, asserted against the live config."""
    flat = flatten({s: cfg[s] for s in AUDITED_SECTIONS if s in cfg})
    registered = {r[0] for r in REGISTRY}
    missing = sorted(k for k in flat
                     if k not in registered and k not in UNGOVERNED)
    assert missing == [], missing


def test_registry_has_no_duplicate_paths():
    paths = [r[0] for r in REGISTRY]
    assert len(paths) == len(set(paths))


def test_every_registry_entry_cites_a_clause():
    """A registry row without a governing clause is an opinion, not an audit."""
    for path, _req, clause, _note in REGISTRY:
        assert clause and clause.strip(), path


def test_ungoverned_and_registry_are_disjoint():
    assert not ({r[0] for r in REGISTRY} & set(UNGOVERNED))


def test_main_exits_nonzero_on_divergence(tmp_path, cfg):
    """Gotcha 13 extension 1 -- the verdict must be readable by its consumer.
    A gate that prints FAIL and exits 0 is worse than no gate."""
    bad = copy.deepcopy(cfg)
    bad["metrics"]["feature_level"]["similarity"] = "something_else"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.config_audit", "--config", str(p)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1, r.stdout
    assert "DIVERGES" in r.stdout


def test_main_exits_zero_on_the_real_config():
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.config_audit",
         "--config", str(CFG)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


# ------------------------------------------- the semantic config hash (5h) --

def test_semantic_hash_ignores_comments_but_not_values(cfg, tmp_path):
    """`train.py`'s `config_hash` is a sha256 of the whole config FILE, so it
    moves when a comment moves -- which is exactly what forced the S=40 seeds
    onto a worktree pinned at the ensemble's SHA (amendment AD4), even though
    every section training reads was byte-identical.

    This hash is over the audited keys' VALUES, so prose cannot move it and a
    real change cannot hide from it.
    """
    import copy
    from src.analysis.config_audit import semantic_config_hash
    base = semantic_config_hash(cfg)
    assert semantic_config_hash(copy.deepcopy(cfg)) == base

    changed = copy.deepcopy(cfg)
    changed["sae"]["k"] = cfg["sae"]["k"] + 1
    assert semantic_config_hash(changed) != base, (
        "a change to sae.k must move the semantic hash")


def test_semantic_hash_covers_every_key_the_coverage_gate_knows_about(cfg):
    """The coupling that makes this a guard rather than a checksum.

    The hash walks REGISTRY + UNGOVERNED -- the same enumeration
    `test_every_config_key_is_governed_or_explicitly_ungoverned` walks. So a new
    experiment-defining key cannot escape the hash without ALSO failing
    coverage, and there is no second list to keep in step.
    """
    from src.analysis.config_audit import (
        AUDITED_SECTIONS, REGISTRY, UNGOVERNED, flatten, semantic_config_hash,
    )
    hashed = {p for p, *_ in REGISTRY} | set(UNGOVERNED)
    present = {k for k in flatten(cfg)
               if k.split(".")[0] in AUDITED_SECTIONS}
    assert present <= hashed, (
        f"audited config keys absent from the semantic hash: "
        f"{sorted(present - hashed)}")

    # AND -- this is the half mutation testing showed was missing -- the
    # assertion above only describes the REGISTRY. It passed while the hash
    # itself was narrowed to REGISTRY alone, letting an UNGOVERNED key escape.
    # So perturb one key of EACH kind and require the hash to move: that tests
    # the function's coverage rather than the table's.
    import copy
    base = semantic_config_hash(cfg)

    governed = copy.deepcopy(cfg)
    governed["model"]["layer"] = 99
    assert semantic_config_hash(governed) != base, "a governed key must be hashed"

    ungoverned_path = sorted(set(UNGOVERNED) & present)[0]
    section, *rest = ungoverned_path.split(".")
    probe = copy.deepcopy(cfg)
    node = probe[section]
    for k in rest[:-1]:
        node = node[k]
    node[rest[-1]] = "__PERTURBED_FOR_THE_COVERAGE_TEST__"
    assert semantic_config_hash(probe) != base, (
        f"{ungoverned_path} is on UNGOVERNED but does not move the semantic "
        f"hash -- a key can escape the hash while still passing coverage")


# ===========================================================================
# AN1 -- the tokenised-value SCREEN, and AN2 -- justification pinning.
#
# Mutation kills RUN:
#   * dropping the value-token route -> kills
#     test_the_screen_would_have_caught_headline_test.
#   * allowing an empty justification -> kills
#     test_every_ungoverned_key_carries_a_written_justification.
#   * removing the stopword filter -> kills
#     test_generic_single_words_do_not_drive_the_screen.
# ===========================================================================

def test_every_ungoverned_key_carries_a_written_justification():
    """AN2. This does NOT make a classification correct -- it stops it being
    SILENT, which is the property that let three misclassifications survive
    (`regression`, `headline_test`, `match_on`). Adding an entry without a
    justification fails here, so the entry and its reason land in one diff."""
    from src.analysis.config_audit import UNGOVERNED
    assert isinstance(UNGOVERNED, dict)
    for key, why in UNGOVERNED.items():
        assert isinstance(why, str), key
        assert len(why.strip()) >= 30, (key, why)
        # A justification must say something about the key, not restate it.
        assert why.strip().lower() != key.lower(), key


def test_no_removed_key_is_still_listed_as_ungoverned():
    """`regression` and `headline_test` are gone from the config; `match_on` and
    `analysis_min_firing_rows` moved to REGISTRY. If any reappears on
    UNGOVERNED, the promotion or removal was undone."""
    from src.analysis.config_audit import REGISTRY, UNGOVERNED
    for gone in ("confound_control.regression",
                 "confound_control.headline_test"):
        assert gone not in UNGOVERNED, gone
        assert gone not in {r[0] for r in REGISTRY}, gone
    for promoted in ("confound_control.match_on",
                     "confound_control.analysis_min_firing_rows"):
        assert promoted not in UNGOVERNED, promoted
        assert promoted in {r[0] for r in REGISTRY}, promoted


def test_match_on_and_caliper_are_both_governed_by_section_5():
    """AN3. Section 5 says "Matching: on `log10` firing rate, caliper 0.1 log10
    units", so the clause names BOTH. AM2 called `match_on` ungoverned
    mechanics; that was a third misclassification, made while fixing two."""
    from src.analysis.config_audit import REGISTRY
    reg = {r[0]: r for r in REGISTRY}
    for k, v in (("confound_control.match_on", "log_firing_rate"),
                 ("confound_control.caliper", 0.1)):
        assert k in reg, k
        assert reg[k][1] == v, k
        assert "section 5" in reg[k][2], (k, reg[k][2])


def test_the_screen_would_have_caught_headline_test():
    """THE DISCRIMINATING CASE, and the reason the value route exists.

    The route is isolated with a SYNTHETIC clause that names the test but never
    the key -- which is the situation H1 was actually in. Using the real
    PREREGISTRATION.md would not isolate it: amendment AM2 now DOCUMENTS the key
    name, so the name route fires too. That is the self-reference effect again,
    and it means the screen would catch this key today by either route -- but it
    would not have, before AM2 existed, which is what is being tested."""
    from src.analysis.config_audit import screen_ungoverned, value_tokens
    import src.analysis.config_audit as ca

    toks = value_tokens("firing_rate_matched_paired_wilcoxon")
    assert "wilcoxon" in toks

    # H1's actual shape: it discusses the test at length and never writes the key.
    clause = ("The firing-rate-matched paired Wilcoxon signed-rank test is "
              "demoted to a ROBUSTNESS ARM; its mechanics are unchanged.")
    assert "headline_test" not in clause

    saved = dict(ca.UNGOVERNED)
    try:
        ca.UNGOVERNED.clear()
        ca.UNGOVERNED["confound_control.headline_test"] = "x" * 40
        r = screen_ungoverned(
            {"confound_control.headline_test":
             "firing_rate_matched_paired_wilcoxon"}, clause)
        row = r["rows"][0]
        assert row["name_hits"] == 0, (
            "the NAME route must find nothing here -- that is the whole point")
        assert row["flagged"] is True
        assert row["route"] == "value"
        assert "wilcoxon" in row["value_token_hits"]
    finally:
        ca.UNGOVERNED.clear()
        ca.UNGOVERNED.update(saved)


def test_the_name_only_screen_would_have_MISSED_it():
    """The counterfactual, stated as an assertion: with the value route removed,
    the same input is reported clean. That is what happened."""
    from src.analysis.config_audit import screen_ungoverned
    import src.analysis.config_audit as ca
    clause = ("The firing-rate-matched paired Wilcoxon signed-rank test is "
              "demoted to a ROBUSTNESS ARM.")
    saved = dict(ca.UNGOVERNED)
    try:
        ca.UNGOVERNED.clear()
        ca.UNGOVERNED["confound_control.headline_test"] = "x" * 40
        # a value the tokeniser cannot use -> only the name route can fire
        r = screen_ungoverned({"confound_control.headline_test": None}, clause)
        assert r["rows"][0]["flagged"] is False
    finally:
        ca.UNGOVERNED.clear()
        ca.UNGOVERNED.update(saved)


def test_generic_single_words_do_not_drive_the_screen():
    from src.analysis.config_audit import value_tokens
    for junk in ("all", "none", "mean", "sum", "log_mean", "true"):
        assert not any(t.lower() in {"all", "none", "mean", "sum", "true", "log"}
                       for t in value_tokens(junk)), junk


def test_a_non_string_value_yields_no_tokens():
    from src.analysis.config_audit import value_tokens
    for v in (None, 0.1, 200, True, ["a"], {"b": 1}):
        assert value_tokens(v) == set()


def test_the_screen_declares_itself_a_screen_not_a_gate():
    """The limitation must travel with the output: a clause governing a key by
    describing a ROLE names neither key nor value, and no grep reaches a
    paraphrase. A clean screen is not evidence the classification is right."""
    from src.analysis.config_audit import screen_ungoverned
    r = screen_ungoverned({}, "nothing here")
    assert "NOT A GATE" in r["status"]
    assert "ROLE" in r["limitation"]
    assert r["n_flagged"] == 0


def test_the_screen_reports_context_rather_than_filtering(tmp_path):
    """Two current flags are hits on amendment AM2 itself. Filtering them would
    be heuristic, and a heuristic filter is how an instrument gets switched off
    rather than fixed (AJ4). Context is reported so a person can see it."""
    import yaml
    from pathlib import Path
    from src.analysis.config_audit import flatten, screen_ungoverned
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "mvp.yaml").read_text())
    from src.analysis.prereg_log import read_log
    if not (repo / "docs" / "record" / "preregistration").is_dir():
        # a release export does not carry the log: it is in the project record, which is deposited
        # apart with restricted files. Against no text the screen flags nothing and this test would
        # pass without having looked at anything (gotcha 28)
        import pytest
        pytest.skip("needs the pre-registration log, which is in the project record and not in the "
                    "released repository (README.md, 'What is released')")
    pre = read_log()        # the whole log as one text (the single file, until section CT18-D10)
    r = screen_ungoverned(flatten(cfg), pre)
    assert r["known_self_references"], "the self-references must be named"
    for row in r["rows"]:
        if row["flagged"]:
            assert row["match_context"], row["key"]


def test_a_flag_raised_by_the_BACKTICKED_LEAF_still_carries_its_context():
    """AMENDMENT CD. `flagged` counts the dotted key OR the backticked leaf, so
    the context must be built from BOTH -- it was built from the dotted key
    alone, which made "flagged with an empty context" reachable by construction
    for every leaf-route hit (gotcha 25: the instrument could not produce output
    for one of the two routes that raise the flag).

    The fixture DISCRIMINATES: the dotted key `data.eval_cluster_fraction` does
    NOT appear in this text, so only the leaf route can fire, and the old code
    returned `[]` here while still setting `flagged`.
    """
    from src.analysis.config_audit import flatten, screen_ungoverned
    text = ("Some preamble that governs nothing in particular. The split holds "
            "out `eval_cluster_fraction` of the CLUSTERS, which is the control "
            "the headline comparison rests on. More prose follows.")
    r = screen_ungoverned(flatten({"data": {"eval_cluster_fraction": 0.10}}),
                          text)
    row = next(x for x in r["rows"] if x["key"] == "data.eval_cluster_fraction")
    assert row["flagged"] is True
    assert row["route"] == "name"
    assert "data.eval_cluster_fraction" not in text      # only the leaf can fire
    assert row["match_context"], "a leaf-route flag must still be readable"
    assert "eval_cluster_fraction" in row["match_context"][0]
