"""
CONFIG <-> PRE-REGISTRATION AUDIT. Amendment AB, decision B's CORRECT clause.

THE HOLE THIS FILLS, stated plainly because it is a class of defect and not
one bug. `src/analysis/preflight.py` checks artifacts against each other and
against the label store -- a PEER check. `src/analysis/artifact_identity.py`
checks that an artifact is the one a consumer expects -- also a peer check.
**Nothing checked the config against the pre-registration.**

That gap surfaced when `metrics.feature_level.similarity` was found reading
`min(activation_correlation, decoder_cosine)` while the pre-registration
(H3/J2b) specifies decoder cosine alone. It surfaced because somebody noticed
during implementation. **That is luck, not machinery**, and the contradiction
had been sitting in the file since the original pre-registration -- through
every amendment, every pre-flight run, and 336 passing tests.

This module is the machinery. Every experiment-defining config key is listed
with the clause that governs it and the value that clause requires. Divergence
FAILS, and the failure is readable by its consumer -- `SystemExit(1)`, because
a gate whose failure cannot propagate is worse than no gate (gotcha 13
extension 1).

WHAT THIS DELIBERATELY DOES NOT DO. It does not check that the config is
INTERNALLY consistent, and it does not check values whose governing decision is
still open (those are declared `OPEN` and asserted to be absent or null, which
is itself a check -- a key that has quietly acquired a value while its decision
is open is exactly as wrong as one that diverges).

ON COVERAGE, and this is the honest limitation. A registry can only check keys
somebody put in it, so an unlisted experiment-defining key is invisible to this
audit exactly as it was before. `test_config_audit.py` therefore asserts that
every key under the audited sections is EITHER in the registry OR on an
explicit `UNGOVERNED` list -- so adding a key to the config without deciding
its provenance breaks the build. That is what turns a checklist into a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

# Sections whose keys must every one be accounted for (registry or UNGOVERNED).
AUDITED_SECTIONS = ("model", "sae", "data", "metrics", "confound_control")

# (config path, required value, governing clause, note)
# `required` of None means "must be null/absent while the decision is open".
REGISTRY: list[tuple[str, object, str, str]] = [
    # ---- locked decisions (CLAUDE.md "Locked decisions -- do not relitigate")
    ("model.name", "facebook/esm2_t33_650M_UR50D", "CLAUDE.md locked",
     "ESM-2-650M"),
    ("model.layer", 24, "CLAUDE.md locked; validated job 19490036",
     "residual stream, argmax FVE at every k"),
    ("model.d_model", 1280, "CLAUDE.md locked", ""),
    ("model.apply_final_layer_norm", False, "CLAUDE.md locked", ""),
    ("sae.arch", "topk", "CLAUDE.md locked", ""),
    ("sae.dict_size", 10240, "CLAUDE.md locked", "8x expansion"),
    ("sae.k", 32, "CLAUDE.md locked", ""),
    ("sae.n_seeds", 20, "CLAUDE.md locked; extension is amendment Y3",
     "raising N requires an amendment, not a config edit"),
    ("sae.b_dec_trainable", True, "open Q5 RESOLVED 2026-07-24",
     "trainable from a shared geometric-median init, NOT frozen"),
    ("sae.b_dec_source", "geometric_median", "open Q5", ""),
    ("data.cluster_members", "all", "open Q1 DECIDED 2026-07-23",
     "clustering only makes the split homology-disjoint; it discards nothing"),
    ("data.cluster_identity", 0.3, "CLAUDE.md locked", "mmseqs2 at 30%"),
    # ---- PROMOTED FROM `UNGOVERNED` by the AN1 screen's VALUE route, which is
    # the whole point of adding that route: the key NAME appears nowhere, but the
    # VALUE `swissprot` does, and Swiss-Prot is a CLAUDE.md locked decision. My
    # justification ("the corpus name is not the corpus") was too clever -- the
    # config key IS what selects the corpus.
    ("data.source", "swissprot", "CLAUDE.md locked",
     "Swiss-Prot, mmseqs2 at 30% identity, cluster-disjoint train/eval"),
    ("data.split", "cluster_disjoint", "CLAUDE.md locked", ""),
    ("data.train_max_residues", None, "open Q2 DECIDED: no cap", ""),
    ("data.split_seed", 0, "pinned in swissprot/split/split.json", ""),

    # ---- ensemble recipe, set from the pilot (CLAUDE.md current state)
    ("sae.lr", 2e-4, "pilot job 19507709", ""),
    ("sae.batch_size", 4096, "pilot job 19507709", ""),
    ("sae.n_steps", 30000, "pilot job 19507709", ""),
    ("sae.optimizer", "adamw", "open Q3 adopted", ""),
    ("sae.grad_clip", 1.0, "open Q3 adopted", ""),
    ("sae.decoder_grad_projection", True, "open Q3 adopted", ""),
    ("sae.loss_reduction", "d_model_mean", "open Q3", "vs d_model_sum, ~1280x"),
    ("sae.aux_k", 512, "AuxK locked in as insurance", ""),
    ("sae.dead_tokens_threshold", 500000, "lowered so AuxK can engage",
     "never fired: 0 dead features in all 20 seeds"),

    # ---- metrics
    ("metrics.primary", "average_precision", "CLAUDE.md locked",
     "residue labels are extremely imbalanced"),
    ("metrics.concept_level.statistic", "best_feature_ap",
     "CLAUDE.md locked; H14 records it had no selection control", ""),
    ("metrics.feature_level.matching", "mutual_nearest_neighbour",
     "amendment X1", "one-way does not supply a cross-seed identity"),

    # ---- THE ONE THIS MODULE EXISTS FOR
    ("metrics.feature_level.similarity", "decoder_cosine",
     "amendment AB decision B; H3/J2b",
     "`min(activation_correlation, decoder_cosine)` let the correlation term "
     "VETO a decoder-cosine match, reintroducing b_dec dependence into match "
     "membership, concentrated in the low-firing tail"),
    ("metrics.feature_level.similarity_robustness_arm", "activation_correlation",
     "amendment AB decision B", "its own arm, never combined via min"),
    ("metrics.feature_level.threshold", 0.70, "amendment AB decision A",
     "REFERENCE theta only; the full sensitivity curve is primary"),
    ("metrics.feature_level.threshold_role",
     "reference_only_full_curve_is_primary", "amendment AB decision A", ""),
    ("metrics.feature_level.threshold_selection_rule",
     "retired_see_amendment_AB_decision_A", "amendment AB decision A",
     "X2's smallest-qualifying tiebreak is VOID BY FAILED PREMISE"),

    # ---- confound control
    ("confound_control.caliper", 0.1, "original pre-registration section 5",
     "log10 firing-rate units; report unmatched count. A DEGREE OF FREEDOM, not "
     "mechanics: caliper width decides which pairs enter the matched comparison "
     "and therefore moves that arm's estimate (amendment AN3). Pinned at 0.1 by "
     "section 5, and restated in the specification record for that reason"),
    # ---- PROMOTED FROM `UNGOVERNED` (amendment AN3). AM2 called this
    # "mechanics" with no governing clause. WRONG: section 5 says in terms
    # "Matching: on `log10` firing rate, caliper 0.1 log10 units", so the clause
    # names BOTH keys. That was a THIRD misclassification, made while fixing the
    # first two, and it is why item 3 asked the question.
    ("confound_control.match_on", "log_firing_rate",
     "original pre-registration section 5",
     "section 5 fixes the matching axis explicitly; it is not a free knob"),
    ("confound_control.local_id_min_firing_set", 200, "gotcha 4",
     "TwoNN is unstable on small firing sets, which ARE low-rate features"),
    # ---- PROMOTED FROM `UNGOVERNED` (amendment AL4). This key was filed as
    # plumbing and is not: amendment A4a fixes it, and AL3 makes the primary
    # regression inherit it, so it now selects the population of BOTH arms.
    ("confound_control.analysis_min_firing_rows", 200,
     "amendment A4a; inherited by the design-3 regression per AL3",
     "~1.1e-5 of the 17.7M eval rows. '0 dead features' was measured at a "
     "~2e-6 threshold, so a population firing at 1e-5-1e-4 is compatible with "
     "it and is effectively dead for AP and unreliable for MNN"),
    ("confound_control.alignment_fdr_q", 0.05,
     "original pre-registration section 3", ""),
    ("confound_control.alignment_min_normalised_gain", 0.05,
     "original pre-registration; magnitudes in amendment A2", ""),
    ("confound_control.corrector_scan.min_firing_rate", 0.5,
     "open Q5 GATE; job 19565987 clean", ""),
    ("confound_control.corrector_scan.min_bias_fraction", 0.5,
     "open Q5 GATE; job 19565987 clean", ""),
]

# Keys deliberately not governed by a pre-registration clause: paths, plumbing,
# and tuning knobs with no inferential content. Listing them is what makes the
# coverage test meaningful -- silence is not the same as a decision.
#
# ============================================================================
# THE COVERAGE GATE PROVES A KEY IS *ACCOUNTED FOR*, NOT THAT ITS ACCOUNTING IS
# *CORRECT* (amendment AL4, and it is gotcha 15 one level up).
# ============================================================================
# `test_every_config_key_is_governed_or_explicitly_ungoverned` walks
# REGISTRY + UNGOVERNED and fails on an unlisted key. That closes the coverage
# hole the similarity key escaped through. It CANNOT close this one: a key with
# real inferential content, filed here, is invisible to the gate BY
# CONSTRUCTION -- membership of this set is the assertion that no clause governs
# it, and the gate has no way to check that assertion.
#
# Two keys were misfiled here and both were found by asking, not by the gate:
#   * `confound_control.regression` held a STRING OF THE PRIMARY INFERENTIAL
#     MODEL and was stale against F1 for the whole of F's life. Removed from
#     the config entirely (AL4) rather than corrected, so there is no second
#     source of truth for the model to drift again.
#   * `confound_control.analysis_min_firing_rows` is amendment A4a's population
#     floor and AL3 makes the regression inherit it. PROMOTED to REGISTRY.
#
# So the rule this list needs, which no test can enforce: an entry here is a
# CLAIM that the key has no inferential content, and it is re-read whenever an
# amendment touches the thing the key describes. Same standing as gotcha 14's
# finding about the locked-decisions list -- "locked" and "ungoverned" both
# insulate an item from churn by design, and therefore also from re-examination.
# ============================================================================
# EVERY ENTRY CARRIES A WRITTEN JUSTIFICATION (amendment AN2), and a test fails
# if one is added without it. That is the same construct AM4 applied to the
# retired-literal allowlist, one layer over: it does NOT make the classification
# correct, but it stops it being SILENT -- which is the property that let three
# misclassifications survive (`regression`, `headline_test`, `match_on`).
# ============================================================================
UNGOVERNED: dict[str, str] = {
    "model.max_len": "ESM-2's own positional limit, not our choice: 1024 "
                     "positions minus <cls> and <eos>",
    "sae.b_dec_path": "a filesystem path. The VALUE it points at is governed "
                      "(sae.b_dec_source, open Q5); the path is not",
    "sae.b_dec_sample_rows": "sample size for the geometric-median INIT only. "
                             "b_dec is trainable, so the init is a starting "
                             "point and not a fixed value",
    "sae.aux_coef": "1/32, inherited with the AuxK recipe. AuxK never fired "
                    "(0 dead features in every seed), so it moves nothing",
    "data.cluster_coverage": "mmseqs2 -c. Erring toward over-clustering is the "
                             "safe direction for a homology-disjoint split, "
                             "and the value is documented in the config",
    "data.cluster_cov_mode": "mmseqs2 bidirectional mode; same reasoning",
    "data.cluster_sensitivity": "mmseqs2 -s. Erring high: under-clustering "
                                "means missed homologs, i.e. leakage",
    "data.eval_cluster_fraction": "the split proportion. The realised split is "
                                  "pinned in swissprot/split/split.json, which "
                                  "is what any consumer reads",
    "data.eval_max_residues": "null. Eval is NEVER subsampled -- that is the "
                              "governed fact, and it is the value being null",
    "metrics.secondary": "f1_at_natural_threshold, reported alongside AP and "
                         "never substituted for it. AP is the governed primary",
    "metrics.concept_level.report": "which summaries to print. A presentation "
                                    "list; it changes no computed quantity",
    "confound_control.alignment_min_normalised_gain_sensitivity":
        "the PRE-SPECIFIED sensitivity value beside the governed primary "
        "(0.05). Both are reported; this one is the arm, not the threshold",
    # `confound_control.headline_test` was here and is GONE from the config
    # (AM2). It named a ROLE that amendment H1 had already changed.
    # `confound_control.match_on` was here and is now in REGISTRY (AN3) --
    # section 5 governs it and AM2 got that wrong.
    # `confound_control.regression` was here and is GONE (AL4).
    # `confound_control.analysis_min_firing_rows` was here and is in REGISTRY
    # (AL4). No removed name is listed, so if any reappears in the config the
    # coverage gate fails -- which is the behaviour wanted.
}


# ---------------------------------------------------------------------------
# THE UNGOVERNED SCREEN (amendment AN1). A SCREEN, NOT A GATE.
# ---------------------------------------------------------------------------
# The first version grepped KEY NAMES against the pre-registration and found
# nothing, while `confound_control.headline_test` was misclassified the whole
# time. The reason is that a clause can name the key's VALUE rather than its
# name: H1 discusses the firing-rate-matched paired Wilcoxon test at length and
# never writes `headline_test`, but the value
# `firing_rate_matched_paired_wilcoxon` TOKENISES to `wilcoxon`, which H1
# contains. So the screen searches names AND tokenised values, which strictly
# dominates the name-only version and would have caught the actual instance.
#
# WHY IT IS STILL NOT A GATE, and this is the part that must not be lost: **a
# clause governing a key by describing a ROLE names neither the key nor its
# value.** "The inferential primary is the regression" governs
# `confound_control.regression` without containing the string `regression` in a
# form tied to that key, and no grep reaches a paraphrase. Necessary, not
# sufficient. The complement is a person reading each key's value against the
# clause governing what the value names, and that is not mechanisable.
#
# Generic single words are dropped, because the leaf names here include ordinary
# English (`report`, `source`, `secondary`) and a loose match lands its false
# positives on the prose that explains the rule (gotcha 20's corollary).
_SCREEN_STOPWORDS = frozenset({
    "all", "none", "null", "true", "false", "mean", "sum", "max", "min", "log",
    "rate", "test", "type", "mode", "path", "source", "report", "secondary",
    "primary", "on", "of", "the", "and", "or", "per", "set", "list", "value",
    # Added after running the screen (AN1): `metrics.secondary` =
    # `f1_at_natural_threshold` flagged on `threshold` (131 hits) and `natural`
    # (6) purely because this document discusses thresholds constantly. Neither
    # hit is about that key.
    "threshold", "natural", "matched", "firing", "paired", "count", "score",
})
_SCREEN_MIN_TOKEN = 5


def value_tokens(value) -> set[str]:
    """Tokens worth searching for from a config VALUE."""
    if not isinstance(value, str):
        return set()
    parts = {t for t in value.replace("-", "_").split("_") if t}
    parts.add(value)
    return {t for t in parts
            if len(t) >= _SCREEN_MIN_TOKEN and t.lower() not in _SCREEN_STOPWORDS}


def screen_ungoverned(flat: dict, prereg_text: str) -> dict:
    """Flag UNGOVERNED keys that the pre-registration may in fact govern.

    Two routes, reported separately so a hit says WHICH one fired:
      * `name`  -- the full dotted key, or the exact backticked leaf;
      * `value` -- a tokenised value (this is the route that would have caught
                   `headline_test`).
    """
    import re as _re
    rows = []
    for key in sorted(UNGOVERNED):
        leaf = key.split(".")[-1]
        n_name = (len(_re.findall(_re.escape(key), prereg_text))
                  + len(_re.findall("`" + _re.escape(leaf) + "`", prereg_text)))
        val = flat.get(key, "__ABSENT__")
        hits = {}
        for tok in value_tokens(val):
            c = len(_re.findall(r"\b" + _re.escape(tok) + r"\b", prereg_text,
                                _re.I))
            if c:
                hits[tok] = c
        # CONTEXT, not a filter. Two of the current flags are hits on amendment
        # AM2 itself -- the paragraph that documents these very keys as false
        # positives -- which is gotcha 20's corollary for the third time in this
        # project: a check that scans text lands its false positives on its own
        # documentation. Filtering them would be heuristic, and a heuristic
        # filter is how an instrument gets switched off rather than fixed
        # (AJ4). So the context is reported and a person reads it.
        #
        # THE CONTEXT MUST BE BUILT FROM THE SAME PATTERNS THAT SET THE FLAG.
        # It was not, and amendment CD found it: `n_name` counts the dotted key
        # PLUS the backticked leaf, while this loop searched the dotted key
        # ALONE -- so a key flagged purely by its backticked leaf was flagged
        # with an EMPTY context, by construction, every time. That is gotcha
        # 25's shape: the instrument that is supposed to let a person adjudicate
        # a flag could not produce output for one of the two routes that raise
        # it. Surfaced when CD's own text wrote `eval_cluster_fraction` in
        # backticks for the first time, which flagged the key and left a reader
        # nothing to read.
        #
        # The value route is included for the same reason -- a value-token hit
        # is a flag, so it must also be readable.
        pats = [_re.escape(key), "`" + _re.escape(leaf) + "`"]
        pats += [r"\b" + _re.escape(t) + r"\b" for t in hits]
        ctx = []
        seen = set()
        for pat in pats:
            for m in list(_re.finditer(pat, prereg_text, _re.I))[:2]:
                a = max(0, m.start() - 90)
                s = " ".join(prereg_text[a:m.end() + 90].split())
                if s not in seen:
                    seen.add(s)
                    ctx.append(s)
            if len(ctx) >= 2:
                break
        ctx = ctx[:2]
        rows.append({
            "key": key, "value": val,
            "name_hits": n_name, "value_token_hits": hits,
            "match_context": ctx,
            "justification": UNGOVERNED[key],
            # DERIVED (gotcha 20).
            "flagged": bool(n_name or hits),
            "route": ("name" if n_name else "value" if hits else None),
        })
    flagged = [r for r in rows if r["flagged"]]
    return {
        "rows": rows,
        "n_flagged": len(flagged),
        "flagged_keys": [r["key"] for r in flagged],
        "status": "SCREEN_ONLY -- NOT A GATE",
        "known_self_references": [
            "data.source and metrics.secondary are flagged ONLY because "
            "amendment AM2 names them while explaining that a naive leaf-name "
            "grep produced them as false positives. Read `match_context` before "
            "treating either as a finding (amendment AN1)."],
        "limitation": (
            "a clause governing a key by describing a ROLE names neither the key "
            "nor its value, and no grep reaches a paraphrase. A clean screen is "
            "not evidence the classification is right (amendment AN1)."),
    }


def flatten(d: dict, pre: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        p = f"{pre}.{k}" if pre else k
        if isinstance(v, dict):
            out.update(flatten(v, p))
        else:
            out[p] = v
    return out


def get(flat: dict, path: str):
    return flat.get(path, "__ABSENT__")


def semantic_config_hash(cfg: dict) -> str:
    """Hash of the config's MEANING: every audited key's value, and nothing else.

    `train.py`'s `config_hash` is a sha256 of the whole config FILE, so it moves
    when a comment moves. That is not academic -- it is what forced the S = 40
    seeds to launch from a worktree pinned at the ensemble's SHA (amendment AD4):
    `train.py` and every section it reads were byte-identical, yet `config_hash`
    differed because the `annotations:` prose had grown.

    This hash is over `REGISTRY` + `UNGOVERNED` -- the SAME enumeration the
    coverage gate walks. That coupling is the point: a new key cannot slip out
    of the hash without first failing
    `test_every_config_key_is_governed_or_explicitly_ungoverned`. There is no
    second list to keep in step.

    Recorded ALONGSIDE `config_hash`, never replacing it: seeds 20-39 are
    recording the file hash as this lands, and redefining the existing key would
    manufacture exactly the provenance split AD4 avoided. Absent on older
    checkpoints, which is honest and readable.
    """
    flat = flatten(cfg)
    paths = sorted({p for p, *_ in REGISTRY} | set(UNGOVERNED))
    payload = json.dumps([[p, get(flat, p)] for p in paths],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def audit(cfg: dict) -> dict:
    """Compare the config against the registry. Returns a report."""
    flat = flatten({s: cfg[s] for s in AUDITED_SECTIONS if s in cfg})
    rows, problems = [], []

    for path, required, clause, note in REGISTRY:
        actual = get(flat, path)
        if actual == "__ABSENT__":
            # ABSENT is reported as absent, never treated as agreement
            # (gotcha 15). A required-null key may legitimately be absent.
            status = "ABSENT_OK" if required is None else "ABSENT"
            if status == "ABSENT":
                problems.append(f"{path}: ABSENT, {clause} requires {required!r}")
        elif required is None:
            status = "OK" if actual is None else "SHOULD_BE_NULL"
            if status != "OK":
                problems.append(
                    f"{path}: {actual!r}, but {clause} leaves it open/null")
        elif isinstance(required, float) or isinstance(actual, float):
            ok = (isinstance(actual, (int, float))
                  and abs(float(actual) - float(required)) <= 1e-12)
            status = "OK" if ok else "DIVERGES"
            if not ok:
                problems.append(
                    f"{path}: {actual!r} but {clause} requires {required!r}")
        else:
            status = "OK" if actual == required else "DIVERGES"
            if status != "OK":
                problems.append(
                    f"{path}: {actual!r} but {clause} requires {required!r}")
        rows.append({"path": path, "actual": actual, "required": required,
                     "clause": clause, "status": status, "note": note})

    registered = {r[0] for r in REGISTRY}
    uncovered = sorted(k for k in flat if k not in registered
                       and k not in UNGOVERNED)
    for k in uncovered:
        problems.append(
            f"{k}: in the config but neither governed nor listed UNGOVERNED "
            "-- decide its provenance or add it to UNGOVERNED explicitly")

    return {
        "rows": rows,
        "uncovered_keys": uncovered,
        "n_registered": len(REGISTRY),
        "n_config_keys_audited": len(flat),
        "problems": problems,
        # Derived verdict, never an asserted sentence (gotcha 20).
        "verdict": "PASS" if not problems else "FAIL",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rep = audit(cfg)
    rep["semantic_config_sha16"] = semantic_config_hash(cfg)

    for r in rep["rows"]:
        if r["status"] not in ("OK", "ABSENT_OK"):
            print(f"  {r['status']:14s} {r['path']}: {r['actual']!r} "
                  f"(requires {r['required']!r} -- {r['clause']})")
    for k in rep["uncovered_keys"]:
        print(f"  UNCOVERED      {k}")
    print(f"[config-audit] {rep['n_registered']} governed keys, "
          f"{rep['n_config_keys_audited']} config keys audited, "
          f"{len(rep['problems'])} problems -> {rep['verdict']}")
    # The MEANING of the config, insensitive to comments and formatting --
    # unlike train.py's whole-file `config_hash`, which moved on prose and
    # forced the S=40 launch onto a pinned worktree (AD4).
    print(f"[config-audit] semantic_config_sha16 = "
          f"{rep['semantic_config_sha16']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2, default=str))

    if rep["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
