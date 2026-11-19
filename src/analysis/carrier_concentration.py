"""BB section 3 -- THE LAST ATTRIBUTION ATTEMPT FOR THE CARRIER EXCESS.

=============================================================================
THE STATE OF PLAY, AND WHY THIS IS THE LAST ATTEMPT
=============================================================================
Carriers match at **0.5617** against a firing-rate-matched ambient of
**0.4895** -- an excess of **+7.22 pp** (AY7). It is **UNATTRIBUTED**.
§AZ11 stratified additionally on homology support and the excess survived, but
§BA10 then measured carriers at the **50.52nd percentile** of their
firing-matched peers' `d_fs`: they were never support-enriched, so the second
axis had nothing to remove and that null result carries **no evidential
weight** (§BA6's rule -- a reading can be repaired after the fact; a
measurement that cannot distinguish two causes cannot).

The remaining pre-computed candidate is the FEATURE-SIDE pair: activation
`perplexity` and `concentration`, which exist as covariates from AL8b's
pre-declared sensitivity arm.

=============================================================================
BOTH HALVES ARE REQUIRED, AND THAT IS THE LESSON FROM THE LAST ATTEMPT
=============================================================================
**An enrichment check alone cannot attribute the excess.** Enrichment in a
covariate explains the excess only if that covariate ALSO predicts
reproducibility. §AZ11 measured one half and could not distinguish two causes.
So this module runs BOTH, and neither is optional:

  1. **ENRICHMENT.** Where do carriers sit in the concentration and perplexity
     distributions of their FIRING-RATE-MATCHED peers? Percentile position,
     median and IQR -- on **§BA10's OWN pre-declared bands (45 / 55)**, not on
     bands chosen now that the question has changed.
  2. **THE MATCHED AMBIENT, RE-DRAWN.** Cells matched on the nine firing-rate
     strata x concentration strata, reported beside the singly- and
     doubly-matched figures.

=============================================================================
"TRIPLY-MATCHED" IS AMBIGUOUS, SO ALL FOUR ARMS ARE COMPUTED
=============================================================================
The brief names the new arm "triply-matched" and then specifies "the existing
nine firing-rate strata x concentration strata", which is two axes. Rather
than adjudicate a wording question, **every cell index is run through the one
function** (`carrier_rates_by_cell`, which is generic in its cell index):

  * `single`  -- firing rate alone                     (AY7's own cell)
  * `double`  -- firing rate x homology support        (AZ11's cell)
  * `firing_x_concentration` -- the literal reading    (two axes)
  * `triple`  -- firing x support x concentration      (three axes)

The ambiguity is resolved by MEASUREMENT rather than by choosing a reading,
which costs one extra `bincount` and removes a judgement call.

=============================================================================
THE STOPPING RULE, DECLARED BEFORE THE RUN (brief section 3c)
=============================================================================
**If this returns not-enriched, the excess stays unattributed, the writeup
says so, and there are NO further attribution attempts.** Two null enrichment
checks plus an explicit statement that the excess could not be attributed is a
better report than a third search: eliminating candidates one at a time until
something sticks is how a post-hoc mechanism story gets assembled by accident,
and directional mechanism claims stand at **2 of 10** in this project's ledger.

=============================================================================
THE ACCEPTANCE CRITERIA -- FOUR, AND THEY ARE THE LOAD-BEARING PART
=============================================================================
  1. the per-anchor MNN counts are BIT-IDENTICAL to `rf_counts_theta070_n19`
     (`carrier_identity`'s criterion, reused);
  2. the SINGLE arm reproduces `carrier_identity_ay_v3.json` EXACTLY;
  3. the DOUBLE arm reproduces `carrier_support_ba.json` EXACTLY;
  4. **the `d_fs` enrichment distribution reproduces `carrier_support_ba.json`'s
     §BA9 block EXACTLY.** This is the one that matters for THIS module:
     `support_distribution` gained a `value_name` LABEL so the concentration
     arm could go through the same function, and criterion 4 is what proves
     the generalisation changed no arithmetic. Without it, a difference
     between the `d_fs` and `concentration` answers could be the covariate or
     could be the edit.

**THIS JOB CROSSES BOTH SEED-ORDER SPLITS** (gotcha 29). AP and `gtilde` are
LEXICOGRAPHIC; `d_fs` and `rf_counts` are NUMERIC; and the feature-side
covariates are resolved PER CHECKPOINT, which is a third path. Positions 0 and
1 are the only ones that coincide, so every axis is permuted BY LABEL and the
covariate arm's per-anchor provenance is asserted POSITIONALLY
(`verify_feature_side_seed_provenance`, reused).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from src.analysis.carrier_identity import (
    THETA_REFERENCE,
    carriers_from_ap,
    firing_bin,
    mnn_partner_matrix,
    order_decoders_to,
    seed_label_to_int,
    verify_counts_bit_identical,
)
from src.analysis.carrier_support import (
    align_to_ap_order,
    cell_index,
    cell_occupancy,
    carrier_rates_by_cell,
    excess_comparison,
    reproduces_ay,
    support_bin_within,
    support_distribution,
)
from src.analysis.concept_repro import (
    REFERENCE_PERCENTILE,
    assert_labels_align,
    detection_matrix,
    detection_rate,
    null_thresholds,
    observed_best_ap,
    to_gain,
)
from src.analysis.corrector_scan import BIN_EDGES
from src.analysis.design3_join import verify_feature_side_seed_provenance
from src.analysis.rf_marginal import load_decoder

N_SUPPORT_BINS = 3
N_CONC_BINS = 3

# §BA7.1's bands, DECLARED BEFORE BA10's RUN and reused here unchanged. The
# brief forbids choosing new bands now, and it is right to: a band chosen after
# one covariate has been measured is a band chosen knowing what it must say.
ENRICHED_AT = 55.0
DEPLETED_BELOW = 45.0

# AZ-P1's cut, reused rather than invented: "the excess shrinks materially"
# means the same thing here it meant there -- below HALF the singly-matched
# excess.
SHRINKS_MATERIALLY_BELOW_FRACTION = 0.5

# A cell with fewer than this many features cannot supply a baseline that means
# anything. Reported as unusable rather than pooled into a neighbour, because
# silently pooling would make the finest stratification look like the coarsest.
MIN_USABLE_CELL = 2


# ---------------------------------------------------------------------------
# The enrichment reading
# ---------------------------------------------------------------------------

def enrichment_verdict(dist: dict, covariate: str) -> dict:
    """§BA7.1's bands applied to any covariate, DERIVED from the measured
    median (gotcha 20).

    Total over the statistic's whole domain AND over the CAUSES of each branch
    (gotcha 30 / §BA6): each branch states what it establishes about the
    matched-ambient arm and, where the answer is "nothing", says so rather than
    letting an uninformative null read as support.

    **`had_something_to_remove` is the binary the outcome grid keys on, and it
    is NOT "enriched".** A DEPLETED covariate is also a covariate the two
    groups do not share, so stratifying on it can also move the excess. Reading
    the grid off "enriched" alone would leave the depleted case unmapped --
    which is gotcha 30's defect, one level up.
    """
    if dist.get("status") != "COMPUTED":
        return {
            "covariate": covariate,
            "verdict": "NOT_COMPUTED",
            "median_percentile": None,
            "had_something_to_remove": None,
            "reading": "no carrier cell had a usable peer pool, so the "
                       "question was never asked rather than answered in the "
                       "negative",
        }
    m = float(dist["median_percentile"])
    if m >= ENRICHED_AT:
        v = "ENRICHED"
        reading = (f"carriers sit HIGHER in {covariate} than their "
                   "firing-matched peers, so a stratification on it has "
                   "something to remove. Whether it removes the excess is the "
                   "matched-ambient arm's question, not this one.")
    elif m >= DEPLETED_BELOW:
        v = "NOT_MATERIALLY_ENRICHED"
        reading = (f"carriers were NOT materially different from their "
                   f"firing-matched peers in {covariate}, so stratifying on "
                   "it could not have changed anything. A matched ambient on "
                   "this axis says NOTHING either way, and an uninformative "
                   "null is not support.")
    else:
        v = "DEPLETED"
        reading = (f"carriers sit LOWER in {covariate} than their "
                   "firing-matched peers. The axis is still informative -- "
                   "the two groups differ on it -- but in the opposite "
                   "direction. NO MECHANISM IS OFFERED (BA5).")
    return {
        "covariate": covariate,
        "verdict": v,
        "median_percentile": m,
        "iqr_percentile": dist.get("iqr_percentile"),
        "share_above_50th": dist.get("share_above_50th"),
        "bands": {"ENRICHED": f">= {ENRICHED_AT}",
                  "NOT_MATERIALLY_ENRICHED":
                      f"{DEPLETED_BELOW} to {ENRICHED_AT}",
                  "DEPLETED": f"< {DEPLETED_BELOW}"},
        "bands_declared": "PREREGISTRATION BA7.1, before BA10's run; reused "
                          "unchanged by BB section 3a rather than re-chosen",
        "had_something_to_remove": bool(v != "NOT_MATERIALLY_ENRICHED"),
        "reading": reading,
    }


def outcome_grid(enrichment: dict, single: dict, matched: dict,
                 arm_name: str) -> dict:
    """BB section 3b's 2x2, TOTAL, with every cell mapped including the one
    declared impossible.

    | | excess shrinks materially | excess survives |
    |---|---|---|
    | had something to remove | the excess is substantially that covariate |
      it is not the explanation; the alignment reading strengthens by
      elimination |
    | had nothing to remove | **CANNOT OCCUR** -- an inconsistency, and that is
      the finding | matching had nothing to remove; UNINFORMATIVE, the excess
      stays unattributed |

    The bottom-left cell is included because a rule needs every cell mapped
    (gotcha 24) -- and because the one time this project left a cell unmapped,
    the data landed in it (AC9, X2).
    """
    e1 = single.get("excess_A_minus_matched_pp")
    e2 = matched.get("excess_A_minus_matched_pp")
    had = enrichment.get("had_something_to_remove")
    if had is None or not np.isfinite(e1) or not np.isfinite(e2) or e1 == 0.0:
        return {
            "arm": arm_name,
            "cell": "UNDEFINED",
            "single_excess_pp": e1, "matched_excess_pp": e2,
            "ratio": None,
            "reading": "one of the two inputs is undefined, so no cell of the "
                       "grid is entered and none is manufactured",
        }
    ratio = e2 / e1
    shrinks = bool(ratio < SHRINKS_MATERIALLY_BELOW_FRACTION)
    if had and shrinks:
        cell = "ENRICHED_AND_SHRINKS"
        reading = (f"the excess is substantially {enrichment['covariate']}, "
                   "NOT alignment.")
    elif had and not shrinks:
        cell = "ENRICHED_AND_SURVIVES"
        reading = (f"{enrichment['covariate']} is not the explanation; the "
                   "alignment reading strengthens BY ELIMINATION -- which is "
                   "weaker than positive evidence and is stated as such.")
    elif (not had) and not shrinks:
        cell = "NOT_ENRICHED_AND_SURVIVES"
        reading = ("matching had nothing to remove, so this arm is "
                   "UNINFORMATIVE and the excess remains UNATTRIBUTED. Per "
                   "the stopping rule declared at BB section 3c, there are no "
                   "further attribution attempts.")
    else:
        cell = "NOT_ENRICHED_AND_SHRINKS__DECLARED_IMPOSSIBLE"
        reading = ("INCONSISTENCY. Carriers do not differ from their peers on "
                   "this covariate, yet stratifying on it moved the excess by "
                   "more than half. Something is wrong with the "
                   "stratification, and THAT is the finding. Reported, not "
                   "explained away.")
    return {
        "arm": arm_name,
        "cell": cell,
        "covariate": enrichment["covariate"],
        "enrichment_verdict": enrichment["verdict"],
        "had_something_to_remove": had,
        "single_excess_pp": e1,
        "matched_excess_pp": e2,
        "ratio_matched_over_single": float(ratio),
        "shrinks_materially": shrinks,
        "shrink_cut": SHRINKS_MATERIALLY_BELOW_FRACTION,
        "shrink_cut_provenance": "AZ-P1's own cut, reused rather than invented",
        "reading": reading,
    }


def score_bb_p2(enrichment: dict) -> dict:
    """BB-P2, computed from the value beside it (gotcha 20).

    Filed **NOT SEALED** (§BB1): the per-feature concentration is a bincount
    and an entropy over the row-index cache, and `assemble_covariates` has
    already been executed by the join's feature-side arm -- its output was
    reduced before storage, which AZ3.2 refuses to count as a seal.

    **THE CATEGORY IS RECORDED EVEN THOUGH IT IS NOT TALLIED**, because the
    brief asks for the filing to be visible either way: the reasoning behind
    it ("average-precision selection rewards a feature whose activation mass
    sits on a concept's residues, which is a form of concentration") is a
    MECHANISM story about how the selection criterion interacts with the
    covariate, not a comparison between two computed distributions. It would
    belong in the **directional mechanism** bin (2 of 10), not the
    within-system one.
    """
    v = enrichment.get("verdict")
    return {
        "statement": "carriers ARE materially enriched in activation "
                     "concentration relative to their firing-rate-matched "
                     "peers, on BA10's bands",
        "standing": "FILED, NOT SEALED -- NOT TALLIED (BB1)",
        "category_if_it_were_tallied": "directional mechanism (the 2/10 bin)",
        "category_note": "the reasoning is a story about how AP selection "
                         "interacts with the covariate, not a comparison "
                         "between two quantities we compute. Filed visibly "
                         "rather than in the bin with the better record.",
        "median_percentile": enrichment.get("median_percentile"),
        "verdict": v,
        "outcome": (None if v in (None, "NOT_COMPUTED")
                    else "MATERIALLY_ENRICHED" if v == "ENRICHED"
                    else "NOT_MATERIALLY_ENRICHED"),
    }


# ---------------------------------------------------------------------------
# Acceptance criterion 4: the d_fs arm against BA10
# ---------------------------------------------------------------------------

BA9_NUMERIC_FIELDS = (
    "n_carrier_cells", "n_carrier_cells_without_a_peer_pool",
    "median_percentile", "mean_percentile", "p05", "p95",
    "share_above_50th", "across_seed_median_of_medians",
    "across_seed_sd_of_medians", "across_seed_se_of_medians",
)


def reproduces_ba9(dist: dict, reference: Path) -> dict:
    """ACCEPTANCE CRITERION 4 -- the `d_fs` arm against BA10's stored block.

    EXACT float equality on every numeric field, plus the per-seed medians,
    the IQR, the composition-confounded pooled median and every stratum row.
    An ABSENT key is a problem, never agreement (gotcha 15).

    **`covariate` is EXCLUDED from the comparison: it is the ADDITION.**
    Comparing a key the reference could not have carried would report a
    difference for the one thing that is supposed to be new -- the same
    exclusion `_combined_fingerprint` makes for BA11's new verdict.
    """
    doc = json.loads(reference.read_text())
    ref = (doc.get("section_BA9_carrier_support_vs_firing_matched_peers", {})
           .get("distribution"))
    if not isinstance(ref, dict):
        return {"reference": str(reference), "verdict": "DIFFERS",
                "problems": ["the reference carries no BA9 distribution block"],
                "comparisons": []}
    rows, problems = [], []

    def cmp(key, a, b):
        same = (a == b)
        rows.append({"key": key, "reference": b, "this_run": a,
                     "same": bool(same)})
        if not same:
            problems.append(f"{key}: reference {b!r} vs this run {a!r}")

    for k in BA9_NUMERIC_FIELDS:
        if k not in ref:
            rows.append({"key": k, "reference": "<ABSENT>",
                         "this_run": dist.get(k), "same": False})
            problems.append(f"{k}: ABSENT (absent is not matching)")
            continue
        cmp(k, dist.get(k), ref[k])
    cmp("iqr_percentile", dist.get("iqr_percentile"), ref.get("iqr_percentile"))
    cmp("per_seed_median_percentile", dist.get("per_seed_median_percentile"),
        ref.get("per_seed_median_percentile"))
    cmp("composition_confounded_pooled.median_percentile",
        dist.get("composition_confounded_pooled", {}).get("median_percentile"),
        ref.get("composition_confounded_pooled", {}).get("median_percentile"))
    cmp("deduplicated.median_percentile",
        dist.get("deduplicated_per_seed_feature", {}).get("median_percentile"),
        ref.get("deduplicated_per_seed_feature", {}).get("median_percentile"))

    ref_str = {r.get("stratum"): r for r in ref.get("by_firing_stratum", [])}
    for r in dist.get("by_firing_stratum", []):
        s = r.get("stratum")
        if s not in ref_str:
            problems.append(f"stratum {s}: ABSENT from the reference")
            rows.append({"key": f"stratum_{s}", "reference": "<ABSENT>",
                         "this_run": r.get("status"), "same": False})
            continue
        rr = ref_str[s]
        cmp(f"stratum_{s}.status", r.get("status"), rr.get("status"))
        if r.get("status") != "COMPUTED":
            continue
        for k in ("n_carrier_cells", "median_percentile_in_peer_d_fs",
                  "carrier_median_d_fs_mean_over_seeds",
                  "peer_median_d_fs_mean_over_seeds",
                  "shift_nats_mean_over_seeds"):
            cmp(f"stratum_{s}.{k}", r.get(k), rr.get(k))
    return {
        "reference": str(reference),
        "reference_identity": doc.get("identity"),
        "n_compared": len(rows),
        "comparisons": rows,
        "problems": problems,
        "verdict": "REPRODUCES_BA9_EXACTLY" if not problems else "DIFFERS",
        "scope": "every numeric field of the pooled distribution, the per-seed "
                 "medians, the composition-confounded and deduplicated "
                 "medians, and every stratum row. `covariate` is EXCLUDED -- "
                 "it is the addition.",
    }


def reproduces_ba_arm(arm: dict, reference: Path, which: str) -> dict:
    """ACCEPTANCE CRITERION 3 -- the single or double arm against BA's rates."""
    doc = json.loads(reference.read_text())
    key = {"single": "single_arm_firing_only",
           "double": "double_arm_firing_and_support"}[which]
    ref = doc.get("section_3a_double_matching", {}).get(key)
    if not isinstance(ref, dict):
        return {"reference": str(reference), "arm": which, "verdict": "DIFFERS",
                "problems": [f"the reference carries no {key} block"],
                "comparisons": []}
    rows, problems = [], []
    for k in ("n_cells", "n_concept_seedpair_cells_evaluated",
              "rate_B_carrier_match", "rate_A_carrier_has_any_partner",
              "conditional_B_given_A", "ambient_match_rate_same_pair_set",
              "matched_ambient", "excess_A_minus_matched_pp",
              "ratio_A_over_matched",
              "n_carriers_without_usable_cell_baseline",
              "n_carrier_cells_with_a_usable_baseline"):
        if k not in ref:
            rows.append({"key": k, "reference": "<ABSENT>",
                         "this_run": arm.get(k), "same": False})
            problems.append(f"{k}: ABSENT (absent is not matching)")
            continue
        a, b = arm.get(k), ref[k]
        same = (a == b)
        rows.append({"key": k, "reference": b, "this_run": a,
                     "same": bool(same)})
        if not same:
            problems.append(f"{k}: reference {b!r} vs this run {a!r}")
    return {"reference": str(reference), "arm": which, "comparisons": rows,
            "problems": problems,
            "verdict": f"REPRODUCES_BA_{which.upper()}_EXACTLY"
                       if not problems else "DIFFERS"}


# ---------------------------------------------------------------------------
# Cell usability
# ---------------------------------------------------------------------------

def cell_usability(cell: np.ndarray, n_cells: int,
                   min_usable: int = MIN_USABLE_CELL) -> dict:
    """Per-cell occupancy across ALL seeds, and how many fall below usable.

    Reported rather than pooled: a cell with one feature contributes no
    baseline at all, and silently folding it into a neighbour would make the
    finest stratification quietly behave like a coarser one -- the failure the
    `support_bin_within` docstring warns about, arriving from the other side.
    """
    counts = np.zeros((cell.shape[0], n_cells), dtype=np.int64)
    for s in range(cell.shape[0]):
        counts[s] = np.bincount(cell[s], minlength=n_cells)
    below = counts < min_usable
    return {
        "n_cells": int(n_cells),
        "min_usable_cell": int(min_usable),
        "n_seed_cells_below_usable": int(below.sum()),
        "n_seed_cells_total": int(counts.size),
        "share_seed_cells_below_usable": float(below.mean()),
        "n_features_in_unusable_cells": int(counts[below].sum()),
        "min_occupancy": int(counts.min()),
        "median_occupancy": float(np.median(counts)),
        "max_occupancy": int(counts.max()),
        "occupancy_seed0": counts[0].tolist(),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


def _attrs(hf) -> dict:
    out = {}
    for k, v in hf.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def build_feature_covariates(ap_seeds: list[str], ckpts_root: Path,
                             row_index_cache: Path, eval_cache: Path,
                             labels: Path, labels_json: Path,
                             concepts: list[str], n_rows: int,
                             n_features: int, scored_row_set: str,
                             k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """`perplexity` and `concentration` per (seed, feature), in the AP's
    LEXICOGRAPHIC row order.

    Row `s` MUST hold the covariates of the seed named at `ap_seeds[s]`, and
    that is asserted POSITIONALLY rather than assumed: this module joins
    carrier identities (lexicographic) to a covariate resolved per CHECKPOINT,
    which is a third path across gotcha 29's split. A set-equality or a count
    is invariant under a permutation and could not see one.
    """
    from src.analysis.smooth_mu import assemble_covariates, resolve_row_index

    perp = np.empty((len(ap_seeds), n_features), dtype=np.float64)
    conc = np.empty((len(ap_seeds), n_features), dtype=np.float64)
    records = []
    for s, label in enumerate(ap_seeds):
        n = seed_label_to_int(label) if str(label).startswith("seed_") \
            else int(label)
        ck = ckpts_root / f"seed_{n}"
        ri = resolve_row_index(row_index_cache, ck, eval_cache,
                               scored_row_set, k, n_rows)
        with h5py.File(ri, "r") as h:
            ck_sha = str(h.attrs.get("checkpoint_sha256_16"))
        records.append({
            "anchor_position": s,
            "anchor_label": str(n),
            "ckpt_dir": ck.name,
            "row_index": ri.name,
            "checkpoint_sha256_16": ck_sha,
            "ckpt_dir_matches_anchor_label": bool(ck.name == f"seed_{n}"),
            "row_index_name_carries_the_ckpt_dir": bool(ck.name in ri.name),
        })
        cov = assemble_covariates(labels, labels_json, ri, concepts, n_rows,
                                  n_features)["feature"]
        perp[s] = cov["perplexity"]
        conc[s] = cov["concentration"]
        print(f"  [cov] row {s:2d} = seed {n:2d}: {ri.name}", flush=True)
    prov = verify_feature_side_seed_provenance(records)
    return perp, conc, prov


def run(ap_path: Path, null_path: Path, gtilde_path: Path, d_fs_path: Path,
        ckpts: list[Path], ckpts_root: Path, row_index_cache: Path,
        eval_cache: Path, labels: Path, labels_json: Path,
        counts_reference: Path, ay_reference: Path, ba_reference: Path,
        out_path: Path, device: str = "cpu",
        theta: float = THETA_REFERENCE, stratum: str = "pooled",
        n_support: int = N_SUPPORT_BINS, n_conc: int = N_CONC_BINS,
        scored_row_set: str = "label_store", k: int = 32) -> dict:
    with h5py.File(ap_path, "r") as hf:
        g = hf[stratum]
        ap_concepts = _decode(g["concepts"][:])
        ap_seeds = _decode(hf["seeds"][:])
        in_pool = g["in_argmax_pool"][:].astype(bool)
        ap = g["ap"][:]
        prevalence = g["prevalence"][:]
        fire_counts = hf["fire_counts"][:]
        ap_identity = _attrs(hf)
    n_rows = int(ap_identity["n_rows"])

    with h5py.File(null_path, "r") as hf:
        null_concepts = _decode(hf["concepts"][:])
        null_seeds = _decode(hf["seeds"][:])
        null_cmax = hf["null_cmax_b"][:]

    assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)
    pool_concepts = [c for c, kk in zip(ap_concepts, in_pool) if kk]
    assert_labels_align("concept labels (AP pool vs null)",
                        pool_concepts, null_concepts)

    with h5py.File(gtilde_path, "r") as hf:
        gt_seeds = _decode(hf["seeds"][:])
        gt_identity = _attrs(hf)

    with h5py.File(d_fs_path, "r") as hf:
        d_raw = hf["d_fs_cluster_entropy"][:]
        d_seeds = [int(x) for x in hf["seeds"][:]]
        d_identity = _attrs(hf)
    perm_d = align_to_ap_order([str(x) for x in d_seeds], ap_seeds,
                               str(d_fs_path))
    d_fs = d_raw[perm_d]

    # ---- decoders, in the AP artifact's LEXICOGRAPHIC order ---------------
    meta_seed: dict[Path, int] = {}
    loaded: dict[Path, torch.Tensor] = {}
    for p in ckpts:
        W, meta = load_decoder(p, device)
        meta_seed[p] = int(meta["seed"])
        loaded[p] = W
    ordered_paths = order_decoders_to(ap_seeds, meta_seed)
    decoders = [loaded[p] for p in ordered_paths]
    assert_labels_align("decoder seed order vs AP seed labels",
                        [f"seed_{meta_seed[p]}" for p in ordered_paths],
                        ap_seeds)
    lex_to_numeric = np.argsort([meta_seed[p] for p in ordered_paths])

    n_anchors = len(decoders)
    partner, mnn_sim = mnn_partner_matrix(decoders, theta)

    # ---- ACCEPTANCE 1 ------------------------------------------------------
    acc_counts = verify_counts_bit_identical(
        mnn_sim, n_anchors, theta, counts_reference, lex_to_numeric)
    if acc_counts["verdict"] != "BIT_IDENTICAL":
        print("ACCEPTANCE (counts) FAILED", json.dumps(acc_counts, indent=2),
              flush=True)
        raise SystemExit(1)
    print(f"[conc] acceptance 1: {acc_counts['verdict']}", flush=True)

    # ---- detection, carriers, cells ---------------------------------------
    ap_pool = ap[:, :, in_pool]
    best_ap = observed_best_ap(ap_pool)
    best_gain = to_gain(best_ap, prevalence[in_pool])
    thr = null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled")
    detected = detection_matrix(best_gain, thr)
    d = detection_rate(detected)

    carrier, ties = carriers_from_ap(ap_pool)
    fbin = firing_bin(fire_counts.astype(np.float64) / float(n_rows))
    sbin = support_bin_within(fbin, d_fs, n_support)
    n_fbins = len(BIN_EDGES) - 1
    n_features = int(ap_pool.shape[1])

    # ---- the feature-side covariates --------------------------------------
    print(f"[conc] assembling perplexity and concentration for "
          f"{n_anchors} anchors", flush=True)
    perp, conc, cov_prov = build_feature_covariates(
        ap_seeds, ckpts_root, row_index_cache, eval_cache, labels,
        labels_json, pool_concepts, n_rows, n_features, scored_row_set, k)
    if cov_prov["problems"]:
        raise SystemExit(
            "FATAL: the covariate arm's per-anchor provenance does not check "
            "out. Refusing to build covariates on an axis that cannot be "
            "verified.\n" + json.dumps(cov_prov, indent=2))
    print(f"[conc] covariate seed axis: {cov_prov['verdict']}", flush=True)

    cbin = support_bin_within(fbin, conc, n_conc)

    # ---- section 3a: ENRICHMENT -------------------------------------------
    dist_d_fs = support_distribution(carrier, detected, d_fs, fbin, n_fbins,
                                     value_name="d_fs")
    acc_ba9 = reproduces_ba9(dist_d_fs, ba_reference)
    print(f"[conc] acceptance 4: {acc_ba9['verdict']}", flush=True)
    if acc_ba9["verdict"] != "REPRODUCES_BA9_EXACTLY":
        print(json.dumps(acc_ba9, indent=2)[:8000], flush=True)
        raise SystemExit(1)

    dist_conc = support_distribution(carrier, detected, conc, fbin, n_fbins,
                                     value_name="concentration")
    dist_perp = support_distribution(carrier, detected, perp, fbin, n_fbins,
                                     value_name="perplexity")
    read_conc = enrichment_verdict(dist_conc, "concentration")
    read_perp = enrichment_verdict(dist_perp, "perplexity")

    # ---- section 3b: the matched ambients ---------------------------------
    arms_spec = {
        "single_firing_only": (cell_index(fbin, None, n_support), n_fbins),
        "double_firing_x_support": (cell_index(fbin, sbin, n_support),
                                    n_fbins * n_support),
        "firing_x_concentration": (cell_index(fbin, cbin, n_conc),
                                   n_fbins * n_conc),
        "triple_firing_x_support_x_concentration": (
            (np.asarray(fbin, dtype=np.int64) * n_support
             + np.asarray(sbin, dtype=np.int64)) * n_conc
            + np.asarray(cbin, dtype=np.int64),
            n_fbins * n_support * n_conc),
    }
    arms, usability = {}, {}
    for name, (cell, n_cells) in arms_spec.items():
        print(f"[conc] arm {name}: {n_cells} cells", flush=True)
        arms[name] = carrier_rates_by_cell(carrier, detected, partner, cell,
                                           n_cells)
        usability[name] = cell_usability(cell, n_cells)

    single = arms["single_firing_only"]
    acc_single = reproduces_ay(single, ay_reference)
    if acc_single["verdict"] != "REPRODUCES_AY7_EXACTLY":
        print("ACCEPTANCE (AY7 rates) FAILED",
              json.dumps(acc_single, indent=2), flush=True)
        raise SystemExit(1)
    print(f"[conc] acceptance 2: {acc_single['verdict']}", flush=True)

    acc_ba_single = reproduces_ba_arm(single, ba_reference, "single")
    acc_ba_double = reproduces_ba_arm(arms["double_firing_x_support"],
                                      ba_reference, "double")
    for a in (acc_ba_single, acc_ba_double):
        print(f"[conc] acceptance 3 ({a['arm']}): {a['verdict']}", flush=True)
        if a["verdict"].startswith("DIFFERS"):
            print(json.dumps(a, indent=2)[:8000], flush=True)
            raise SystemExit(1)

    comparisons = {
        name: excess_comparison(single, arms[name])
        for name in arms if name != "single_firing_only"
    }
    grid = {
        "firing_x_concentration": outcome_grid(
            read_conc, single, arms["firing_x_concentration"],
            "firing_x_concentration"),
        "triple_firing_x_support_x_concentration": outcome_grid(
            read_conc, single, arms["triple_firing_x_support_x_concentration"],
            "triple_firing_x_support_x_concentration"),
    }

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": "PREREGISTRATION amendment BB section 3 (brief "
                         "section 3) -- the last attribution attempt for the "
                         "7.22 pp carrier excess",
        "exploratory": "AZ7/BA5 -- cannot become primary under any outcome; "
                       "no mechanism sentence is written for any of it",
        "stopping_rule": (
            "DECLARED BEFORE THE RUN (brief section 3c): if the enrichment "
            "check returns NOT_MATERIALLY_ENRICHED, the excess stays "
            "UNATTRIBUTED, the writeup says so, and there are NO further "
            "attribution attempts. Eliminating candidates one at a time until "
            "something sticks is how a post-hoc mechanism story gets "
            "assembled by accident."),
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "carrier_concentration_BB3"},
        "theta": float(theta),
        "reference_percentile": REFERENCE_PERCENTILE,
        "acceptance_criteria": {
            "1_counts_vs_rf_counts": acc_counts,
            "2_single_arm_vs_carrier_identity_AY": acc_single,
            "3_single_and_double_vs_carrier_support_BA": {
                "single": acc_ba_single, "double": acc_ba_double},
            "4_d_fs_enrichment_vs_BA9": acc_ba9,
        },
        "seed_axis": {
            "ap_seed_labels_lexicographic": ap_seeds,
            "decoder_meta_seeds_in_that_order":
                [meta_seed[p] for p in ordered_paths],
            "permutation_lexicographic_to_numeric": lex_to_numeric.tolist(),
            "gtilde_seed_labels": gt_seeds,
            "d_fs_seed_labels": d_seeds,
            "permutation_d_fs_to_ap": perm_d.tolist(),
            "covariate_arm_provenance": cov_prov,
        },
        "argmax_ties": ties,
        "strata": {
            "firing_source": "corrector_scan.BIN_EDGES",
            "firing_edges": list(BIN_EDGES),
            "n_firing_bins": int(n_fbins),
            "n_support_bins": int(n_support),
            "n_concentration_bins": int(n_conc),
            "support_definition": "quantile bins of d_fs_cluster_entropy "
                                  "computed WITHIN each (seed, firing bin)",
            "concentration_definition":
                "quantile bins of the activation concentration computed "
                "WITHIN each (seed, firing bin) -- the same construction, and "
                "for the same reason: a global bin nested inside a firing "
                "stratum is close to constant and the axis would do nothing "
                "while appearing to",
            "double_cell_occupancy_seed0": cell_occupancy(
                cell_index(fbin, cbin, n_conc), conc, n_fbins * n_conc),
        },
        "section_3a_enrichment": {
            "d_fs_reference_arm": {"distribution": dist_d_fs,
                                   "reading": enrichment_verdict(dist_d_fs,
                                                                 "d_fs")},
            "concentration": {"distribution": dist_conc, "reading": read_conc},
            "perplexity": {"distribution": dist_perp, "reading": read_perp},
        },
        "section_3b_matched_ambients": {
            "arms": arms,
            "cell_usability": usability,
            "comparisons_against_the_single_arm": comparisons,
            "note": "'triply-matched' is ambiguous in the brief -- it names "
                    "three axes and specifies two. Both are computed, so the "
                    "wording question is settled by measurement.",
        },
        "outcome_grid": grid,
        "detection": {"n_concepts": int(d.size),
                      "share_D_eq_1": float((d == 1.0).mean()),
                      "share_D_eq_0": float((d == 0.0).mean())},
        "prediction_BB_P2": score_bb_p2(read_conc),
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "gtilde": str(gtilde_path), "d_fs": str(d_fs_path),
                   "labels": str(labels), "row_index_cache":
                       str(row_index_cache), "eval_cache": str(eval_cache),
                   "counts_reference": str(counts_reference),
                   "ay_reference": str(ay_reference),
                   "ba_reference": str(ba_reference),
                   "ap_identity": ap_identity,
                   "gtilde_identity": gt_identity,
                   "d_fs_identity": d_identity},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--d-fs", type=Path, required=True)
    p.add_argument("--ckpts", nargs="+", type=Path, required=True)
    p.add_argument("--ckpts-root", type=Path, required=True)
    p.add_argument("--row-index-cache", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, default=None)
    p.add_argument("--counts-reference", type=Path, required=True)
    p.add_argument("--ay-reference", type=Path, required=True)
    p.add_argument("--ba-reference", type=Path, required=True,
                   help="carrier_support_ba.json -- acceptance 3 and 4")
    p.add_argument("--expect-ckpts", type=int, default=None)
    p.add_argument("--n-support-bins", type=int, default=N_SUPPORT_BINS)
    p.add_argument("--n-concentration-bins", type=int, default=N_CONC_BINS)
    p.add_argument("--scored-row-set", default="label_store")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--theta", type=float, default=THETA_REFERENCE)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.expect_ckpts is not None and len(args.ckpts) != args.expect_ckpts:
        print(f"PRE-FLIGHT FAIL: --expect-ckpts {args.expect_ckpts} but "
              f"{len(args.ckpts)} supplied. A count mismatch silently changes "
              f"which ensemble every rate below describes.", flush=True)
        raise SystemExit(1)

    out = run(args.ap, args.null, args.gtilde, args.d_fs, args.ckpts,
              args.ckpts_root, args.row_index_cache, args.eval_cache,
              args.labels, args.labels_json or args.labels.with_suffix(".json"),
              args.counts_reference, args.ay_reference, args.ba_reference,
              args.out, args.device, args.theta,
              n_support=args.n_support_bins, n_conc=args.n_concentration_bins,
              scored_row_set=args.scored_row_set, k=args.k)

    print("\n[3a] carrier percentile within its own (seed, firing bin) PEER "
          "pool:")
    for nm in ("d_fs_reference_arm", "concentration", "perplexity"):
        blk = out["section_3a_enrichment"][nm]
        sd, rd = blk["distribution"], blk["reading"]
        if sd.get("status") != "COMPUTED":
            print(f"      {nm:<20} {sd.get('status')}")
            continue
        print(f"      {nm:<20} median {sd['median_percentile']:6.2f}  "
              f"IQR [{sd['iqr_percentile'][0]:6.2f}, "
              f"{sd['iqr_percentile'][1]:6.2f}]  "
              f"above-50th {sd['share_above_50th']:.2%}  -> {rd['verdict']}")

    print("\n[3b] matched ambients:")
    for nm, a in out["section_3b_matched_ambients"]["arms"].items():
        u = out["section_3b_matched_ambients"]["cell_usability"][nm]
        print(f"      {nm:<42} cells {a['n_cells']:>3}  "
              f"A {a['rate_A_carrier_has_any_partner']:.4f}  "
              f"matched {a['matched_ambient']:.4f}  "
              f"excess {a['excess_A_minus_matched_pp']:+.3f} pp  "
              f"(unusable carriers {a['n_carriers_without_usable_cell_baseline']:,}"
              f", cells below {u['min_usable_cell']}: "
              f"{u['n_seed_cells_below_usable']})")

    print("\n[3b] the outcome grid:")
    for nm, g in out["outcome_grid"].items():
        print(f"      {nm}: {g['cell']}")
        print(f"        {g['reading']}")

    bp = out["prediction_BB_P2"]
    print(f"\n  BB-P2 ({bp['standing']}): {bp['outcome']}; median percentile "
          f"{bp['median_percentile']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
