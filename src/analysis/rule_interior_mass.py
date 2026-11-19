"""AMENDMENT CO ITEM B -- the THREE-WAY SPLIT of the per-(anchor, feature)
match-count distribution under matching rules 1, 2 and 3.

AMENDMENT CS (2026-09-16): the anchor list, and hence `n_compare`, are CLI
parameters (`--anchors`, default the record's 20); the positional control is
optional; the record-constant controls (C-B0, C-B1, C-B2, C-B4) report
NOT_APPLICABLE off the record pool. The default run IS the CO run. The CO
docstring follows, unchanged.

`n_compare = 19`, `theta = 0.70`, 190 unordered anchor pairs. Rule 1's
construction matched exactly, so the three rules differ in the RULE and in
nothing else.

WHY THIS IS A MEASUREMENT AND NOT A REDUCTION. The CJ driver computed each
rule's per-feature match vector and reduced it to a mean ON THE SAME LINE
(`float((sim >= t).mean())`), so no per-feature count survived and the
DISTRIBUTION of match counts -- which is what "interior mass" is -- was
unrecoverable from `cj_rule_sensitivity.json` at any price short of re-running.
This run accumulates a per-`(anchor, feature)` counter instead of discarding
the boolean, and PERSISTS IT, so no successor pays the 37 minutes again.

WHAT IS COMPARED, AND WHAT MUST NOT BE. The comparison is against
`docs/NUMBERS_OF_RECORD.md` section 14.1's **`n_compare = 19`** interior mass
(0.813916). It is NOT comparable to the 86.5% headline, which is an
`n_compare = 39` object; section 14.1's first mandatory qualifier forbids the two
travelling together as a range, an average or a ratio.

NO RULE IS PROMOTED BY ANYTHING HERE. Rule 1 -- MNN on decoder cosine -- remains
the pre-registered primary and the figure of record, exactly as at CJ. No
mechanism is offered for whichever pattern the splits show; amendment CO
declares that in advance so it is not decided by the answer.

THE ARITHMETIC DELIBERATELY REPLICATES THE CJ DRIVER'S: torch float32 matmul on
column-normalised decoders and row-normalised encoders, cast to float64
afterwards. That is not a style choice -- rule 1's mean here is checked against
CJ's stored `0.495814144736842`, and a float64 product would move the last
digits of a control whose whole value is its tightness.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import h5py
import numpy as np

from src.analysis.rule_sensitivity import (
    encoder_decoder_from_assignments,
    hungarian_assignment,
    mnn_partner,
    mnn_similarity,
    partner_view_from_b,
)
from src.analysis.seed_axis import align_to_reference, stored_seed_labels

THETA = 0.70

# THE RECORD'S GEOMETRY -- the 20 anchors at n_compare = 19 that every control
# target below (14.1's counts, CJ9a/CJ9b's means) was measured on. Until
# amendment CS these were the module's only geometry, as module constants.
N_COMPARE_RECORD = 19
N_ANCHORS_RECORD = 20
DEFAULT_ANCHORS = list(range(N_ANCHORS_RECORD))


def pool_geometry(anchors: list[int]) -> tuple[int, int]:
    """`(n_anchors, n_compare)` for a pool of distinct anchor seeds, all of
    which compare against every other: `n_compare = n_anchors - 1` because an
    anchor never compares against its own dictionary (CLAUDE.md pinned
    notation). Amendment CS: DERIVED from the list, never a constant -- the
    width sweep runs 10 seeds per width at n_compare = 9."""
    anchors = [int(a) for a in anchors]
    if len(anchors) < 2:
        raise SystemExit(f"FATAL: {len(anchors)} anchor(s); a pairwise "
                         "statistic needs at least two.")
    if len(set(anchors)) != len(anchors):
        raise SystemExit(f"FATAL: duplicate anchor seeds in {anchors}.")
    return len(anchors), len(anchors) - 1


# The record's anchors live here; the same labels under another root (sae_ckpts_k64,
# §CT28-pre item 8) are a different pool.
RECORD_CKPT_ROOT = "$PROJECT/sae_ckpts"


def record_controls_apply(anchors: list[int], ckpt_root) -> bool:
    """The record-constant controls (C-B0, C-B1, C-B2, C-B4) compare against
    objects measured on the 20 anchors at n_compare = 19 and mean nothing on
    any other pool; they are reported NOT_APPLICABLE there, never skipped
    silently (gotcha 15). The pool is the record's only if the labels AND the
    checkpoint root are: labels 0-19 under `sae_ckpts_k64` are k = 64
    dictionaries (§CT28-pre item 8)."""
    same_root = str(ckpt_root).replace("\\", "/").rstrip("/") == RECORD_CKPT_ROOT
    return [int(a) for a in anchors] == DEFAULT_ANCHORS and same_root

RULES = ("rule1_mnn_decoder", "rule2_hungarian_decoder",
         "rule3_hungarian_enc_and_dec")

# ---------------------------------------------------------------------------
# CONTROL TARGETS -- quoted from the record, and declared at CO before the run.
# ---------------------------------------------------------------------------
# docs/NUMBERS_OF_RECORD.md 14.1, n_compare = 19 column, over 204,800 cells.
# The shares round-trip to exact integer counts, which is the form checked:
#   0.122104 * 204800 = 25007,  0.813916 * 204800 = 166690,
#   0.063979 * 204800 = 13103,  and 25007 + 166690 + 13103 = 204800.
RECORD_N19_COUNTS = {"k_zero": 25007, "interior": 166690, "k_max": 13103}
# The bar, in CELLS. Ground: BB12.3 measured the MNN counts differing in 4 of
# 204,800 cells between a CPU partition and rtx_5000_ada, so at most 4 cells can
# change endpoint membership across a device class. 20 is five times that and
# still 1e-4 of the population -- loose enough not to fail on float noise, tight
# enough that any real disagreement fires.
SPLIT_BAR_CELLS = 20
# Elementwise bar on the positional comparison. A check on the three SHARES is
# a check on totals and cannot see a permutation (gotcha 27); this one can.
POSITIONAL_BAR_CELLS = 20
POSITIONAL_BAR_MAX_ABS_DELTA = 1
# CJ9a/CJ9b, cj_rule_sensitivity.json, rule 1 mean at theta = 0.70.
CJ_RULE1_MEAN = 0.495814144736842
CJ_MEAN_BAR = 1e-6           # ~4 count-units of 3,891,200; the device scale
CJ_RULE_MEANS = {"rule1_mnn_decoder": 0.4958141447,
                 "rule2_hungarian_decoder": 0.4908,
                 "rule3_hungarian_enc_and_dec": 0.3038}
CJ_AGREE_FRAC = 0.8773


def load_checkpoints(root: Path, anchors: list[int], normalise_encoder: bool = True):
    """Column-normalised decoders and row-normalised encoders, as torch fp32.

    C1, HARD: decoder columns must already be unit-norm -- the trainer
    renormalises after every step, so a failure means the wrong checkpoint.
    ENCODER ROWS ARE NOT unit-norm (ours run 0.960 to 1.720) and are normalised
    here; a raw product would silently rank by row magnitude.

    `normalise_encoder=False` (amendment CT, §CT27(b)) returns the RAW encoder
    rows instead -- the dot-product variant, whose ranking by row magnitude is
    the point of that arm. The decoders are unchanged either way.
    """
    import torch

    D, E, c1 = {}, {}, []
    for s in anchors:
        sd = torch.load(root / f"seed_{s}" / "model.pt", map_location="cpu")
        Wd = sd["W_dec"].float()
        cn = Wd.norm(dim=0)
        c1.append({"seed": s, "min_col_norm": float(cn.min()),
                   "max_col_norm": float(cn.max())})
        if not torch.allclose(cn, torch.ones_like(cn), atol=1e-4):
            raise SystemExit(f"C1 FAILED: seed {s} decoder columns are not "
                             f"unit-norm ({cn.min():.6f}..{cn.max():.6f}).")
        D[s] = Wd / cn
        We = sd["W_enc"].float()
        E[s] = We / We.norm(dim=1, keepdim=True) if normalise_encoder else We
    return D, E, c1


def store_pair_sims(sim: dict, rule: str, ia: int, ib: int,
                    sa: np.ndarray, sb: np.ndarray) -> None:
    """Amendment CT §CT20: the per-COMPARATOR similarity of every anchor
    feature under one rule -- `sim[rule][anchor_row, comparator_row, feature]`
    -- the object a threshold turns into a match indicator. The diagonal stays
    -inf (an anchor never compares against itself), so summing `sim >= theta`
    over the comparator axis IS the persisted count at that theta."""
    sim[rule][ia, ib] = sa
    sim[rule][ib, ia] = sb


def three_way_split(counts: np.ndarray, n_compare: int) -> dict:
    """`P(k = 0)`, interior, `P(k = n_compare)` -- reported TOGETHER, always.

    Interior mass alone is not the shape; amendment CO requires the full split
    for every rule so a move in one endpoint is not read as a move in the other.
    """
    tot = int(counts.size)
    n0 = int((counts == 0).sum())
    nm = int((counts == n_compare).sum())
    if counts.max() > n_compare or counts.min() < 0:
        raise SystemExit(
            f"FATAL: match counts run [{counts.min()}, {counts.max()}] outside "
            f"[0, {n_compare}]. An anchor was credited against itself, or a "
            "pair was counted twice.")
    return {"n_cells": tot, "n_k_zero": n0, "n_interior": tot - n0 - nm,
            "n_k_max": nm, "P_k_zero": n0 / tot,
            "interior_mass": (tot - n0 - nm) / tot, "P_k_max": nm / tot,
            "mean_k_over_n_compare": float(counts.mean() / n_compare)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--anchors", type=int, nargs="+", default=DEFAULT_ANCHORS,
                    help="anchor seed labels (seed_<s> under --ckpt-root); "
                         "n_compare is DERIVED as len - 1 (amendment CS). "
                         "Default: the record's 20 anchors.")
    ap.add_argument("--rf-counts-n19", "--rf-counts-control", dest="rf_counts_n19",
                    type=Path, default=None,
                    help="rf_counts_theta070_n19.h5 -- the POSITIONAL control "
                         "(C-B1). Optional since amendment CS; applies only to "
                         "the record's 20 anchors.")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-h5", type=Path, required=True,
                    help="the per-(anchor, feature) counts, PERSISTED")
    ap.add_argument("--out-sim-h5", type=Path, default=None,
                    help="amendment CT §CT20: ALSO persist every rule's "
                         "per-comparator similarity (anchor, comparator, "
                         "feature), so any theta and a jackknife are a "
                         "threshold, not a re-run. Off by default.")
    args = ap.parse_args()

    anchors = [int(a) for a in args.anchors]
    N_ANCHORS, N_COMPARE = pool_geometry(anchors)
    record_pool = record_controls_apply(anchors, args.ckpt_root)
    D, E, c1 = load_checkpoints(args.ckpt_root, anchors)
    n = D[anchors[0]].shape[1]
    pairs = list(itertools.combinations(anchors, 2))
    if len(pairs) != N_ANCHORS * N_COMPARE // 2:
        raise SystemExit(f"FATAL: {len(pairs)} pairs, expected "
                         f"{N_ANCHORS * N_COMPARE // 2}.")
    print(f"[CO-B] anchors {anchors}: n_anchors={N_ANCHORS} "
          f"n_compare={N_COMPARE}; record controls "
          f"{'APPLY' if record_pool else 'NOT APPLICABLE'}", flush=True)
    print(f"[CO-B] {len(pairs)} unordered anchor pairs, n_features={n}",
          flush=True)

    counts = {r: np.zeros((N_ANCHORS, n), dtype=np.int32) for r in RULES}
    sims = ({r: np.full((N_ANCHORS, N_ANCHORS, n), -np.inf, dtype=np.float64)
             for r in RULES} if args.out_sim_h5 is not None else None)
    pos = {s: i for i, s in enumerate(anchors)}      # seed label -> row
    per_pair, agree_fracs = [], []
    t0 = time.time()
    for idx, (a, b) in enumerate(pairs):
        Cd = (D[a].T @ D[b]).numpy().astype(np.float64)
        Ce = (E[a] @ E[b].T).numpy().astype(np.float64)

        # rule 1 -- MNN on decoder cosine
        s1a = mnn_similarity(Cd)
        s1b = partner_view_from_b(mnn_partner(Cd), s1a, n)

        # rule 2 -- Hungarian one-to-one on decoder cosine. Solved ONCE and
        # reused by rule 3; solving it twice would cost ~28 minutes over the
        # run for an answer already in hand.
        p2, s2a = hungarian_assignment(Cd)
        s2b = partner_view_from_b(p2, s2a, n)

        # rule 3 -- the two matchings must agree
        pe, se = hungarian_assignment(Ce)
        p3, s3a = encoder_decoder_from_assignments(p2, s2a, pe, se)
        s3b = partner_view_from_b(p3, s3a, n)
        agree_fracs.append(float((p3 >= 0).mean()))

        for rule, (sa, sb) in zip(RULES, ((s1a, s1b), (s2a, s2b), (s3a, s3b))):
            counts[rule][pos[a]] += (sa >= THETA)
            counts[rule][pos[b]] += (sb >= THETA)
            if sims is not None:
                store_pair_sims(sims, rule, pos[a], pos[b], sa, sb)

        per_pair.append({
            "pair": [a, b],
            "rule1_frac_a": float((s1a >= THETA).mean()),
            "rule1_frac_b": float((s1b >= THETA).mean()),
            "rule2_frac_a": float((s2a >= THETA).mean()),
            "rule3_frac_a": float((s3a >= THETA).mean()),
            "matchings_agree_frac": agree_fracs[-1],
        })
        if idx % 20 == 0:
            print(f"  pair {idx + 1}/{len(pairs)}  "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)

    splits = {r: three_way_split(counts[r], N_COMPARE) for r in RULES}

    # ---- CONTROLS ---------------------------------------------------------
    controls = {"C1_decoder_unit_norm": {"verdict": "PASS", "per_seed": c1}}

    # C-B1 POSITIONAL. The one that can see a permutation. Record pool only.
    if not record_pool or args.rf_counts_n19 is None:
        controls["C_B1_positional_vs_stored_rf_counts"] = {
            "verdict": "NOT_APPLICABLE",
            "why": ("the stored rf_counts_theta070_n19.h5 is a 20-anchor, "
                    "n_compare = 19 object" if not record_pool else
                    "no control file was supplied"),
        }
    else:
      with h5py.File(args.rf_counts_n19, "r") as hf:
          stored = hf["match_count"][:].astype(np.int32)
          st_attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                      for k, v in hf.attrs.items()}
      st_labels = stored_seed_labels(args.rf_counts_n19)
      if st_labels is None:
          raise SystemExit("FATAL: the stored n19 counts carry no seed labels; "
                           "a positional join would be an assumption (gotcha 29).")
      stored = align_to_reference(st_labels, stored, [str(s) for s in anchors])
      if int(st_attrs.get("n_compare_seeds", -1)) != N_COMPARE:
          raise SystemExit(f"FATAL: control artifact says n_compare_seeds="
                           f"{st_attrs.get('n_compare_seeds')}, not {N_COMPARE}.")
      delta = counts["rule1_mnn_decoder"].astype(np.int64) - stored.astype(np.int64)
      n_diff = int((delta != 0).sum())
      max_abs = int(np.abs(delta).max())
      controls["C_B1_positional_vs_stored_rf_counts"] = {
          "artifact": str(args.rf_counts_n19),
          "n_cells_differing": n_diff,
          "max_abs_delta": max_abs,
          "signed_total_delta": int(delta.sum()),
          "bar_n_cells": POSITIONAL_BAR_CELLS,
          "bar_max_abs_delta": POSITIONAL_BAR_MAX_ABS_DELTA,
          "verdict": ("PASS" if n_diff <= POSITIONAL_BAR_CELLS
                      and max_abs <= POSITIONAL_BAR_MAX_ABS_DELTA else "FAIL"),
          "note": ("The stored artifact was computed on a different device class; "
                   "BB12.3 measured 4 of 204,800 MNN cells differing between CPU "
                   "and rtx_5000_ada, and CJ9b's mean minus this artifact's mean "
                   "is 4 count-units, so a handful of single-unit differences is "
                   "the expected reading and is DERIVED, not predicted."),
      }

    # C-B0 THE BRIEF'S GATE, in its own terms: rule 1's SPLIT against 14.1.
    NA = {"verdict": "NOT_APPLICABLE",
          "why": "control target is a 20-anchor, n_compare = 19 record object"}
    if not record_pool:
        controls["C_B0_rule1_split_vs_record_14_1"] = dict(NA)
        controls["C_B2_mean_vs_cj9b"] = dict(NA)
    got = {"k_zero": splits["rule1_mnn_decoder"]["n_k_zero"],
           "interior": splits["rule1_mnn_decoder"]["n_interior"],
           "k_max": splits["rule1_mnn_decoder"]["n_k_max"]}
    dev = {k: int(got[k] - RECORD_N19_COUNTS[k]) for k in got}
    if record_pool:
      controls["C_B0_rule1_split_vs_record_14_1"] = {
        "recorded_counts": RECORD_N19_COUNTS, "measured_counts": got,
        "deviation_cells": dev, "bar_cells": SPLIT_BAR_CELLS,
        "verdict": ("PASS" if max(abs(v) for v in dev.values())
                    <= SPLIT_BAR_CELLS else "FAIL"),
    }

    # C-B2 the internal identity against CJ's mean. It CANNOT see a permutation
    # (a scatter defect leaves the total untouched) -- C-B1 is the one that can,
    # and this is recorded so the pair is not read as two independent checks.
    m1 = splits["rule1_mnn_decoder"]["mean_k_over_n_compare"]
    if record_pool:
      controls["C_B2_mean_vs_cj9b"] = {
        "measured": m1, "cj_stored": CJ_RULE1_MEAN,
        "abs_difference": abs(m1 - CJ_RULE1_MEAN), "bar": CJ_MEAN_BAR,
        "verdict": "PASS" if abs(m1 - CJ_RULE1_MEAN) <= CJ_MEAN_BAR else "FAIL",
        "identity": ("mean over cells of k/n_compare equals the mean over the "
                     "190 pairs of the per-pair fraction, because each "
                     "unordered pair is credited to both of its anchors"),
        "blind_to": "a permutation of the feature axis (gotcha 27)",
    }

    # C-B3 DOMINANCE, verified rather than predicted: rule 3's matched set is a
    # strict subset of rule 2's, so k3 <= k2 in every cell. Arithmetic, so a
    # violation is an implementation defect (gotcha 44: asserted, never scored).
    viol = int((counts["rule3_hungarian_enc_and_dec"]
                > counts["rule2_hungarian_decoder"]).sum())
    strict = int((counts["rule3_hungarian_enc_and_dec"]
                  < counts["rule2_hungarian_decoder"]).sum())
    controls["C_B3_rule3_dominated_by_rule2"] = {
        "n_cells_violating_k3_le_k2": viol,
        "n_cells_strictly_below": strict,
        "verdict": "PASS" if viol == 0 and strict > 0 else "FAIL",
        "why_strict_matters": ("zero strict cells would mean the two rules "
                               "returned the same object and the instrument "
                               "does not discriminate (gotcha 28)"),
    }

    # C-B4 the three per-pair means against CJ9a, a second reproduction.
    means = {"rule1_mnn_decoder": float(np.mean([r["rule1_frac_a"]
                                                 for r in per_pair])),
             "rule2_hungarian_decoder": float(np.mean([r["rule2_frac_a"]
                                                       for r in per_pair])),
             "rule3_hungarian_enc_and_dec": float(np.mean([r["rule3_frac_a"]
                                                           for r in per_pair]))}
    if record_pool:
        controls["C_B4_per_pair_means_vs_cj9a"] = {
            "measured": means, "cj_stored": CJ_RULE_MEANS,
            "max_abs_difference": max(abs(means[k] - CJ_RULE_MEANS[k])
                                      for k in means),
            "matchings_agree_frac_mean": float(np.mean(agree_fracs)),
            "cj_agree_frac": CJ_AGREE_FRAC,
        }
    else:
        controls["C_B4_per_pair_means_vs_cj9a"] = dict(
            NA, measured=means,
            matchings_agree_frac_mean=float(np.mean(agree_fracs)))
    # C-B5. NOT a symmetry check on the DATA -- mutual-nearest-neighbour is
    # direction-free, so equality is guaranteed IF the scatter is faithful.
    # What it tests is the wiring: `mnn_partner` and `mnn_similarity` are two
    # separate functions (gotcha 27) and nothing but this forces them to agree
    # on WHICH rows matched. If they disagree, the B-side loses or invents a
    # match and the two fractions part company.
    sym = max(abs(r["rule1_frac_a"] - r["rule1_frac_b"]) for r in per_pair)
    controls["C_B5_partner_and_similarity_agree"] = {
        "max_abs_difference_a_vs_b": sym,
        "verdict": "PASS" if sym == 0.0 else "FAIL",
        "what_it_tests": ("that the partner index and the similarity vector "
                          "mark the SAME matched set; exact equality, not "
                          "approximate"),
    }

    gates = ["C_B0_rule1_split_vs_record_14_1",
             "C_B1_positional_vs_stored_rf_counts", "C_B2_mean_vs_cj9b",
             "C_B3_rule3_dominated_by_rule2",
             "C_B5_partner_and_similarity_agree"]
    applicable = [g for g in gates if controls[g]["verdict"] != "NOT_APPLICABLE"]
    failed = [g for g in applicable if controls[g]["verdict"] != "PASS"]

    out = {
        "measurement": "three_way_match_count_split_by_rule_CO",
        "governed_by": "PREREGISTRATION amendment CO item B",
        "theta": THETA, "n_compare": N_COMPARE, "n_anchors": N_ANCHORS,
        "n_pairs": len(pairs), "n_features": int(n), "device_class": "CPU",
        "comparison_set": (("docs/NUMBERS_OF_RECORD.md 14.1, the n_compare = 19 "
                            "column. NOT the 86.5% headline, which is an "
                            "n_compare = 39 object") if record_pool else
                           (f"a within-pool object at n_compare = {N_COMPARE} "
                            f"over anchors {anchors}; comparable to NOTHING at "
                            "n_compare = 19 or 39 (amendment CS0)")),
        "splits": splits,
        "controls": controls,
        "gating_controls_failed": failed,
        "reportable": not failed,
        "controls_not_applicable": [g for g in gates if g not in applicable],
        "anchor_labels": anchors,
        "ckpt_root": str(args.ckpt_root),
        "elapsed_s": round(time.time() - t0, 1),
        "per_pair": per_pair,
    }
    from src.sae.train import git_dirty, git_sha
    out["git_sha"], out["git_dirty"] = git_sha(), git_dirty()

    # PERSIST THE COUNTS. The defect this run exists to repair was a reduction
    # that threw the per-feature vector away; writing only a summary here would
    # repeat it one level up.
    with h5py.File(args.out_h5, "w") as hf:
        for r in RULES:
            d = hf.create_dataset(r, data=counts[r].astype(np.int16))
            d.attrs["rule"] = r
            d.attrs["theta"] = THETA
            d.attrs["n_compare_seeds"] = N_COMPARE
        hf.attrs["seed_labels"] = json.dumps(anchors)
        hf.attrs["measurement"] = out["measurement"]
        hf.attrs["theta"] = THETA
        hf.attrs["n_compare_seeds"] = N_COMPARE
        hf.attrs["n_anchors"] = N_ANCHORS
        hf.attrs["dict_size"] = int(n)
        hf.attrs["similarity"] = "decoder_cosine (rules 1,2); + encoder (rule 3)"
        hf.attrs["device_class"] = "CPU"
        hf.attrs["git_sha"] = out["git_sha"]
        hf.attrs["git_dirty"] = bool(out["git_dirty"])

    if sims is not None:
        # THE GATE THAT MAKES THE PERSISTED OBJECT THE COUNTS' OWN: stored in
        # float64 (the precision the counts were thresholded in -- float32 could
        # flip a cell sitting just below theta) and summed over comparators,
        # every rule must reproduce the counts this run computed, cell for cell.
        gate = {r: int(((sims[r] >= THETA).sum(axis=1)
                        != counts[r]).sum()) for r in RULES}
        out["sim_persistence_gate_cells_differing"] = gate
        with h5py.File(args.out_sim_h5, "w") as hf:
            for r in RULES:
                hf.create_dataset(f"sim_{r}", data=sims[r], compression="gzip")
            hf.attrs.update({"anchor_seed_labels": json.dumps(anchors),
                             "column_seed_labels": json.dumps(anchors),
                             "n_compare_seeds": N_COMPARE, "dict_size": int(n),
                             "device_class": "CPU", "governing_amendment": "CT20",
                             "count_gate_cells_differing": json.dumps(gate),
                             "git_sha": out["git_sha"],
                             "git_dirty": bool(out["git_dirty"])})
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_pair"},
                     indent=2))
    if failed:
        raise SystemExit(
            f"GATING CONTROL(S) FAILED: {failed}. Under amendment CO that is "
            "THE FINDING, and rules 2 and 3 are NOT reported. The artifact is "
            "written so the failure is inspectable.")


if __name__ == "__main__":
    main()
