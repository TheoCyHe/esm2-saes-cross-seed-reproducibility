"""AY §2b -- CARRIER IDENTITY. Does a concept keep the same feature across seeds?

THE QUESTION. When concept `c` is detected in both anchor seed `s` and anchor
seed `t`, is the best-matching feature in `s` a MUTUAL-NEAREST-NEIGHBOUR
PARTNER of the best-matching feature in `t`, at the reference threshold
`theta = 0.70`?

WHY IT IS THE BRIDGE BETWEEN THE TWO HEADLINES. Feature-level reproducibility
does not track alignment; concept-level reproducibility is strongly positive.
The candidate reading is that a concept is carried by DIFFERENT features in
different seeds -- the biology stable, the carriers rotating. That reading is a
directional mechanism claim (the bin this project's ledger scores 2 of 10) and
must not ship as one. This module measures the rates; it writes no mechanism.

=============================================================================
THREE RATES, NOT ONE, BECAUSE TWO OF THEM ARE DIFFERENT EVENTS
=============================================================================
This is the specification point that decides whether the comparison means
anything, and it is stated here because it is easy to miss:

  A. `carrier_has_partner_rate`  P(the carrier in `s` has ANY MNN partner in
                                 `t`, at theta)
  B. `carrier_match_rate`        P(the carrier in `s` has an MNN partner in
                                 `t` AND that partner IS the carrier in `t`)

**B is a strict SUBSET of A.** The ambient rate -- mean `r_f`, the share of
comparison dictionaries in which a feature has a partner -- is an A-shaped
event. So comparing B against ambient compares a STRICTER event against a
LOOSER one, and B falling below ambient is partly structural rather than
evidence about carriers.

Therefore all three are reported, and each comparison is labelled with which
event it is:
  * **B vs the firing-rate-matched ambient** is the quantity AY-P1 names, and
    it is scored on that. Its structural asymmetry is stated beside it.
  * **A vs the firing-rate-matched ambient** is the LIKE-FOR-LIKE comparison --
    the same event on both sides -- and is the one that isolates whether
    carriers are ordinary features of their firing rate.
  * **B / A** is the conditional: given the carrier matches something, how
    often is that something the other seed's carrier. This is the carrier
    IDENTITY question with the availability of any partner divided out.

=============================================================================
WHY THE FIRING-RATE-MATCHED BASELINE IS NOT OPTIONAL
=============================================================================
Carriers are selected on average precision, AP correlates with firing rate, and
firing rate is the dominant driver of reproducibility (the feature-level
result). So ANY excess of a carrier rate over the raw ambient rate could be
entirely a firing-rate effect. For each carrier the baseline is drawn from the
SAME firing-rate stratum in the SAME seed against the SAME comparison seed --
`corrector_scan.BIN_EDGES`, the nine strata this project already uses
everywhere -- and the carrier is EXCLUDED from its own baseline, so the
comparison is not partly a feature compared against itself.

=============================================================================
THE SEED-ORDER SPLIT, AND THE ACCEPTANCE CRITERION THAT CATCHES IT
=============================================================================
This module JOINS an AP-family artifact to a decoder-derived quantity, which is
exactly the crossing gotcha 29 is about: AP artifacts carry LEXICOGRAPHIC seed
labels (`seed_0, seed_1, seed_10, ...`), `rf_counts_*` carry NUMERIC ones
(`0..19`), and only positions 0 and 1 coincide -- so a positional join
mis-pairs 18 of 20 seeds and yields a complete, internally consistent, entirely
wrong answer. Every identity key the standard gates compare is invariant under
a seed permutation, so no gate can see it.

TWO DEFENCES, and the second is the real one:

  1. Decoders are loaded by PATH and ordered by each checkpoint's OWN
     `meta["seed"]`, then permuted into the AP artifact's label order, with the
     permutation asserted elementwise.

  2. THE ACCEPTANCE CRITERION. The per-anchor MNN counts this module computes
     at theta = 0.70 must be **BIT-IDENTICAL** to `match_count` in the existing
     `rf_counts_theta070_n19.h5`, which is stored in NUMERIC order. The job
     EXITS NON-ZERO if they are not.

     That check is what makes defence 1 more than a promise, and it is
     load-bearing in a specific way: the SAME computed array is consumed in
     NUMERIC order (for the acceptance check) and in LEXICOGRAPHIC order (for
     the carrier join). A fixture whose arms share one order cannot test an
     alignment (gotcha 29's corollary), and here the two arms genuinely differ,
     on real data. If the orders were confused the bit-identity fails.

     The comparison is elementwise on the raw buffers -- shape, dtype and total
     all survive a permutation (gotcha 27), so none of them is the check.

MNN IS RECOMPUTED HERE, AND TIED TO THE ARTIFACT OF RECORD BIT-FOR-BIT.
`mnn_partner_matrix` (this module) rebuilds the MNN similarity from the
decoders `rf_marginal.load_decoder` loads; `verify_counts_bit_identical` then
reproduces `rf_counts_theta070_n19.h5`'s counts through the imported
`rf_marginal.counts_at_theta` and refuses on anything but exact equality.
That is what makes a second implementation safe: if it disagreed with the
artifact that produced 0.4958, the acceptance criterion would say so.
(`pair_statistics` is NOT imported; this docstring said it was -- corrected
2026-09-16, docs/CODEBASE.md section 24a has the same statement.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from src.analysis.corrector_scan import BIN_EDGES
from src.analysis.concept_repro import (
    REFERENCE_PERCENTILE,
    assert_labels_align,
    detection_matrix,
    detection_rate,
    null_thresholds,
    observed_best_ap,
    to_gain,
)
from src.analysis.rf_marginal import counts_at_theta, load_decoder

THETA_REFERENCE = 0.70


# --------------------------------------------------------- the seed axis

def seed_label_to_int(label: str) -> int:
    """`'seed_7'` -> 7. Refuses anything else rather than guessing.

    A silent fallback here would reintroduce the exact defect this module's
    acceptance criterion exists to catch.
    """
    s = str(label)
    if not s.startswith("seed_"):
        raise ValueError(
            f"expected an AP seed label of the form 'seed_<n>', got {s!r}. "
            "Refusing to guess: the seed axis is the one place in this repo "
            "where a wrong-but-plausible mapping produces a complete and "
            "entirely wrong result (gotcha 29).")
    return int(s.split("_", 1)[1])


def order_decoders_to(ap_seed_labels: list[str],
                      meta_seed_by_path: dict[Path, int]) -> list[Path]:
    """Permute checkpoint paths into the AP artifact's label order.

    The mapping is built from each checkpoint's OWN `meta['seed']`, never from
    its filename and never from the order a glob returned. An absent or
    duplicated seed is fatal, not defaulted.
    """
    want = [seed_label_to_int(s) for s in ap_seed_labels]
    by_seed: dict[int, Path] = {}
    for p, s in meta_seed_by_path.items():
        if s in by_seed:
            raise SystemExit(
                f"FATAL: seed {s} supplied twice ({by_seed[s]}, {p}). A "
                "duplicated dictionary would be counted as an independent "
                "comparison seed.")
        by_seed[s] = p
    missing = [s for s in want if s not in by_seed]
    if missing:
        raise SystemExit(
            f"FATAL: the AP artifact names seeds {missing} and no checkpoint "
            f"carries them. Available: {sorted(by_seed)}")
    extra = [s for s in by_seed if s not in want]
    if extra:
        raise SystemExit(
            f"FATAL: checkpoints supplied for seeds {extra} which the AP "
            "artifact does not name. Refusing rather than silently dropping "
            "them -- a comparison set that is not the anchor set changes what "
            "every rate below describes.")
    return [by_seed[s] for s in want]


# -------------------------------------------------------- partner matrices

def mnn_partner_matrix(decoders: list[torch.Tensor], theta: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """For every ordered pair (a, b): the MNN partner index of each feature of
    `a` in `b`, or -1 where there is none at `theta`.

    Returns `(partner, mnn_sim)`:
      partner[a, b, i]  = j if (i, j) is mutual-best AND cos >= theta, else -1
      mnn_sim[a, b, i]  = the mutual-pair cosine, or -inf

    `mnn_sim` is returned so the caller can reproduce `rf_marginal`'s counts
    from the SAME object the partners came from -- if partners and counts were
    computed independently, the acceptance criterion would not constrain the
    partners.
    """
    n = len(decoders)
    dim = decoders[0].shape[1]
    partner = np.full((n, n, dim), -1, dtype=np.int32)
    mnn_sim = np.full((n, n, dim), -np.inf, dtype=np.float32)

    for a in range(n):
        for b in range(a + 1, n):
            C = decoders[a].transpose(0, 1) @ decoders[b]
            a_sim, a_j = C.max(dim=1)
            b_sim, b_i = C.max(dim=0)
            ia = torch.arange(C.shape[0], device=C.device)
            ib = torch.arange(C.shape[1], device=C.device)
            mut_a = b_i[a_j] == ia
            mut_b = a_j[b_i] == ib

            neg = float("-inf")
            sa = torch.where(mut_a, a_sim, torch.full_like(a_sim, neg))
            sb = torch.where(mut_b, b_sim, torch.full_like(b_sim, neg))
            mnn_sim[a, b] = sa.float().cpu().numpy()
            mnn_sim[b, a] = sb.float().cpu().numpy()

            ok_a = (mut_a & (a_sim >= theta)).cpu().numpy()
            ok_b = (mut_b & (b_sim >= theta)).cpu().numpy()
            pa = a_j.cpu().numpy().astype(np.int32)
            pb = b_i.cpu().numpy().astype(np.int32)
            partner[a, b] = np.where(ok_a, pa, -1)
            partner[b, a] = np.where(ok_b, pb, -1)
        print(f"[carrier] pairs done through seed index {a}", flush=True)
    return partner, mnn_sim


def verify_counts_bit_identical(mnn_sim: np.ndarray, n_anchors: int,
                                theta: float, ref_path: Path,
                                numeric_order: np.ndarray) -> dict:
    """THE ACCEPTANCE CRITERION -- bit-identical, not close.

    `mnn_sim` is in the AP artifact's LEXICOGRAPHIC seed order; the reference
    is in NUMERIC order. `numeric_order` is the permutation taking the former
    to the latter, so this function exercises the same permutation the carrier
    join relies on, in the opposite direction.

    Identity is asserted BEFORE the values (gotcha 15): a reference built at a
    different theta, rule, similarity or `n_compare` is not a reference for
    this array at all, and an ABSENT key is reported absent, never as
    agreement.
    """
    off = ~np.eye(mnn_sim.shape[0], dtype=bool)
    counts_lex = counts_at_theta(mnn_sim, off, n_anchors, theta)
    counts_numeric = counts_lex[numeric_order]

    with h5py.File(ref_path, "r") as hf:
        if "match_count" not in hf:
            return {"verdict": "REFERENCE_MALFORMED",
                    "problem": f"{ref_path} has no `match_count`"}
        ref = hf["match_count"][:]
        at = {k: (v.decode() if isinstance(v, bytes) else v)
              for k, v in hf.attrs.items()}

    problems, identity = [], {}
    for key, want in (("theta", theta), ("rule", "mnn"),
                      ("similarity", "decoder_cosine"),
                      ("n_compare_seeds", n_anchors - 1),
                      ("n_anchors", n_anchors)):
        if key not in at:
            identity[key] = "<ABSENT>"
            problems.append(f"{key}: ABSENT (absent is not matching)")
            continue
        got = at[key]
        identity[key] = got.item() if hasattr(got, "item") else got
        same = (abs(float(got) - want) < 1e-12 if isinstance(want, float)
                else int(got) == want if isinstance(want, int)
                else str(got) == str(want))
        if not same:
            problems.append(f"{key}: reference {got!r} vs this run {want!r}")

    same_shape = tuple(ref.shape) == tuple(counts_numeric.shape)
    bit = bool(same_shape and ref.dtype == counts_numeric.dtype
               and ref.tobytes()
               == np.ascontiguousarray(counts_numeric).tobytes())
    return {
        "reference": str(ref_path),
        "reference_identity": identity,
        "identity_problems": problems,
        "same_shape": same_shape,
        "n_cells_differing": int((ref != counts_numeric).sum()) if same_shape
                             else None,
        "total_reference": int(ref.sum()) if same_shape else None,
        "total_this_run": int(counts_numeric.sum()),
        # A total agrees under ANY permutation, so it is reported for
        # diagnosis and is never the criterion.
        "bit_identical": bit,
        "verdict": ("BIT_IDENTICAL" if (bit and not problems)
                    else "IDENTITY_MISMATCH" if problems else "DIFFERS"),
    }


# ------------------------------------------------------------- the carriers

def carriers_from_ap(ap_pool: np.ndarray) -> tuple[np.ndarray, dict]:
    """`carrier[s, c] = argmax_f AP[s, f, c]`, plus a TIE REPORT.

    Ties matter and are not hypothetical: a concept whose AP is flat (all
    zeros, say) has an arbitrary argmax that `np.argmax` resolves to index 0,
    and a "carrier" chosen by tie-break is not a carrier. The count of tied
    cells is reported so a reader can see whether any conclusion rests on one.
    """
    carrier = ap_pool.argmax(axis=1).astype(np.int64)
    mx = ap_pool.max(axis=1)
    n_tied = (ap_pool == mx[:, None, :]).sum(axis=1)
    return carrier, {
        "n_cells": int(carrier.size),
        "n_cells_with_tied_argmax": int((n_tied > 1).sum()),
        "share_tied": float((n_tied > 1).mean()),
        "max_tie_multiplicity": int(n_tied.max()),
    }


def firing_bin(fire_rate: np.ndarray) -> np.ndarray:
    """Stratum index per feature, using this project's nine standing strata.

    `corrector_scan.BIN_EDGES` and not a fresh set: the strata are the ones the
    corrector gate, the predictor's denominator and the design-3 panel all use,
    and a private binning would make this comparison incomparable to those.
    """
    edges = np.asarray(BIN_EDGES, dtype=np.float64)
    return np.clip(np.digitize(fire_rate, edges[1:-1], right=False),
                   0, len(edges) - 2)


def carrier_rates(carrier: np.ndarray, detected: np.ndarray,
                  partner: np.ndarray, fbin: np.ndarray) -> dict:
    """The three rates, the two baselines, and the per-concept distribution.

    Iterates over UNORDERED anchor pairs and counts each once. The carrier
    match is evaluated in the `s -> t` direction; MNN is symmetric, so
    `partner[s, t, i] == j` implies `partner[t, s, j] == i`, and evaluating
    both directions would double-count one event rather than measure two.
    """
    n_seeds, n_concepts = carrier.shape
    n_feat = partner.shape[2]

    # Per-seed, per-comparison match indicator for EVERY feature: the ambient
    # event, and the raw material for the stratum baselines.
    has_partner = partner >= 0                                  # (s, t, i)

    # Stratum sums per (s, t, bin): how many features of seed s in that bin
    # have a partner in t, and how many features are in the bin at all.
    n_bins = len(BIN_EDGES) - 1
    bin_n = np.zeros((n_seeds, n_bins), dtype=np.int64)
    bin_hit = np.zeros((n_seeds, n_seeds, n_bins), dtype=np.int64)
    for s in range(n_seeds):
        bin_n[s] = np.bincount(fbin[s], minlength=n_bins)
        for t in range(n_seeds):
            if s == t:
                continue
            bin_hit[s, t] = np.bincount(fbin[s], weights=has_partner[s, t],
                                        minlength=n_bins).astype(np.int64)

    b_hits = a_hits = 0
    n_pairs_evaluated = 0
    matched_baseline_sum = 0.0
    matched_baseline_n = 0
    per_concept_num = np.zeros(n_concepts, dtype=np.int64)
    per_concept_den = np.zeros(n_concepts, dtype=np.int64)
    unusable_baseline = 0

    for s in range(n_seeds):
        for t in range(s + 1, n_seeds):
            both = detected[s] & detected[t]            # (concept,)
            cs = carrier[s][both]
            ct = carrier[t][both]
            if cs.size == 0:
                continue
            p = partner[s, t, cs]
            a = p >= 0
            b = p == ct
            a_hits += int(a.sum())
            b_hits += int(b.sum())
            n_pairs_evaluated += int(cs.size)
            idx = np.flatnonzero(both)
            np.add.at(per_concept_num, idx, b.astype(np.int64))
            np.add.at(per_concept_den, idx, 1)

            # Firing-rate-matched baseline, EXCLUDING each carrier from its own
            # stratum so the comparison is never a feature against itself.
            bins = fbin[s, cs]
            n_in = bin_n[s, bins]
            hit_in = bin_hit[s, t, bins]
            ok = n_in > 1
            unusable_baseline += int((~ok).sum())
            if ok.any():
                num = hit_in[ok] - a[ok].astype(np.int64)
                den = n_in[ok] - 1
                matched_baseline_sum += float((num / den).sum())
                matched_baseline_n += int(ok.sum())

    # Ambient over the SAME pair set: every anchor feature against every other
    # anchor dictionary. This is mean r_f at n_compare = n_seeds - 1, NOT the
    # 39-dictionary figure, and mixing the two would be a silent N change.
    off = ~np.eye(n_seeds, dtype=bool)
    ambient = float(has_partner[off].mean())

    with np.errstate(invalid="ignore", divide="ignore"):
        per_concept = np.where(per_concept_den > 0,
                               per_concept_num / np.maximum(per_concept_den, 1),
                               np.nan)
    ev = per_concept[per_concept_den > 0]

    matched = (matched_baseline_sum / matched_baseline_n
               if matched_baseline_n else float("nan"))
    b_rate = b_hits / n_pairs_evaluated if n_pairs_evaluated else float("nan")
    a_rate = a_hits / n_pairs_evaluated if n_pairs_evaluated else float("nan")

    return {
        "n_concept_seedpair_cells_evaluated": int(n_pairs_evaluated),
        "n_unordered_seed_pairs": int(n_seeds * (n_seeds - 1) // 2),
        "n_compare_seeds_for_ambient": int(n_seeds - 1),
        "rate_B_carrier_match": b_rate,
        "rate_A_carrier_has_any_partner": a_rate,
        "conditional_B_given_A": (b_hits / a_hits) if a_hits else float("nan"),
        "ambient_match_rate_same_pair_set": ambient,
        "firing_rate_matched_ambient": matched,
        "n_carriers_without_usable_stratum_baseline": int(unusable_baseline),
        "ratios": {
            "B_over_firing_matched": (b_rate / matched
                                      if matched else float("nan")),
            "A_over_firing_matched": (a_rate / matched
                                      if matched else float("nan")),
            "B_over_ambient": b_rate / ambient if ambient else float("nan"),
            "A_over_ambient": a_rate / ambient if ambient else float("nan"),
        },
        "per_concept_carrier_match_rate": {
            "n_concepts_with_at_least_one_pair": int(ev.size),
            "mean": float(ev.mean()) if ev.size else None,
            "median": float(np.median(ev)) if ev.size else None,
            "share_above_0_9": float((ev > 0.9).mean()) if ev.size else None,
            "share_exactly_1": float((ev == 1.0).mean()) if ev.size else None,
            "share_exactly_0": float((ev == 0.0).mean()) if ev.size else None,
            "histogram_10_bins": np.histogram(
                ev, bins=10, range=(0.0, 1.0))[0].tolist() if ev.size else None,
        },
        "_per_concept_values": per_concept.tolist(),
    }


def score_predictions(rates: dict) -> dict:
    """AY-P1 and AY-P2, DERIVED from the measured values (gotcha 20).

    Each verdict is computed from the numbers beside it, never asserted, and
    each carries the caveat that belongs to its own predicate rather than a
    generic one.
    """
    b = rates["rate_B_carrier_match"]
    a = rates["rate_A_carrier_has_any_partner"]
    m = rates["firing_rate_matched_ambient"]
    share = rates["per_concept_carrier_match_rate"]["share_above_0_9"]
    return {
        "AY_P1": {
            "statement": "the carrier-match rate exceeds the "
                         "firing-rate-matched ambient rate",
            "scored_on": "rate_B_carrier_match vs firing_rate_matched_ambient",
            "carrier_match_rate_B": b,
            "firing_rate_matched_ambient": m,
            "ratio": b / m if m else None,
            "outcome": ("EXCEEDS" if b > m else "DOES_NOT_EXCEED"),
            "structural_caveat": (
                "B is a STRICT SUBSET of the baseline's event: the baseline "
                "asks whether a feature has ANY partner, B asks whether the "
                "carrier's partner IS the other seed's carrier. A "
                "DOES_NOT_EXCEED outcome is therefore partly structural. The "
                "like-for-like comparison is A vs the same baseline, reported "
                "beside it and NOT the scored predicate."),
            "like_for_like_A_vs_matched": {
                "carrier_has_any_partner_rate_A": a,
                "ratio": a / m if m else None,
                "outcome": "EXCEEDS" if a > m else "DOES_NOT_EXCEED",
            },
        },
        "AY_P2": {
            "statement": "fewer than 40% of concepts have a per-concept "
                         "carrier-match rate above 0.9",
            "measured_share_above_0_9": share,
            "outcome": (None if share is None
                        else "UNDER_40_PERCENT" if share < 0.40
                        else "AT_OR_ABOVE_40_PERCENT"),
        },
    }


# ------------------------------------------------------------------ driver

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


def run(ap_path: Path, null_path: Path, ckpts: list[Path], out_path: Path,
        counts_reference: Path, device: str = "cpu",
        theta: float = THETA_REFERENCE, stratum: str = "pooled") -> dict:
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
    pool_concepts = [c for c, k in zip(ap_concepts, in_pool) if k]
    assert_labels_align("concept labels (AP pool vs null)",
                        pool_concepts, null_concepts)

    # ---- decoders, ordered into the AP artifact's LEXICOGRAPHIC label order
    meta_seed: dict[Path, int] = {}
    loaded: dict[Path, torch.Tensor] = {}
    for p in ckpts:
        W, meta = load_decoder(p, device)
        meta_seed[p] = int(meta["seed"])
        loaded[p] = W
    ordered_paths = order_decoders_to(ap_seeds, meta_seed)
    decoders = [loaded[p] for p in ordered_paths]

    # The permutation is asserted elementwise, not inherited from the sort.
    assert_labels_align(
        "decoder seed order vs AP seed labels",
        [f"seed_{meta_seed[p]}" for p in ordered_paths], ap_seeds)
    lex_to_numeric = np.argsort([meta_seed[p] for p in ordered_paths])

    n_anchors = len(decoders)
    partner, mnn_sim = mnn_partner_matrix(decoders, theta)

    # ---- THE ACCEPTANCE CRITERION, before anything is reported ------------
    acceptance = verify_counts_bit_identical(
        mnn_sim, n_anchors, theta, counts_reference, lex_to_numeric)
    if acceptance["verdict"] != "BIT_IDENTICAL":
        print("ACCEPTANCE CRITERION FAILED", json.dumps(acceptance, indent=2),
              flush=True)
        raise SystemExit(1)
    print(f"[carrier] acceptance: {acceptance['verdict']} against "
          f"{counts_reference.name}", flush=True)

    # ---- detection and carriers -------------------------------------------
    ap_pool = ap[:, :, in_pool]
    best_gain = to_gain(observed_best_ap(ap_pool), prevalence[in_pool])
    thr = null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled")
    detected = detection_matrix(best_gain, thr)
    d = detection_rate(detected)

    carrier, ties = carriers_from_ap(ap_pool)
    fbin = firing_bin(fire_counts.astype(np.float64) / float(n_rows))

    rates = carrier_rates(carrier, detected, partner, fbin)
    per_concept_vals = rates.pop("_per_concept_values")

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": "PREREGISTRATION amendment AY section 2b",
        # This artifact's OWN provenance, not only its inputs'. An artifact
        # that records what it consumed but not what produced it cannot be tied
        # back to a commit.
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "carrier_identity_AY"},
        "theta": float(theta),
        "reference_percentile": REFERENCE_PERCENTILE,
        "acceptance_criterion": acceptance,
        "seed_axis": {
            "ap_seed_labels_lexicographic": ap_seeds,
            "decoder_meta_seeds_in_that_order":
                [meta_seed[p] for p in ordered_paths],
            "permutation_lexicographic_to_numeric":
                lex_to_numeric.tolist(),
            "note": "the SAME computed array is consumed in numeric order for "
                    "the acceptance check and in lexicographic order for the "
                    "carrier join; confusing them fails the bit-identity",
        },
        "argmax_ties": ties,
        "firing_strata": {"source": "corrector_scan.BIN_EDGES",
                          "edges": list(BIN_EDGES)},
        "rates": rates,
        "predictions": score_predictions(rates),
        "detection": {
            "n_concepts": int(d.size),
            "share_D_eq_1": float((d == 1.0).mean()),
            "share_D_eq_0": float((d == 0.0).mean()),
        },
        "per_concept": [
            {"concept": c, "D": float(d[i]),
             "carrier_match_rate": (None if not np.isfinite(per_concept_vals[i])
                                    else float(per_concept_vals[i]))}
            for i, c in enumerate(pool_concepts)],
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "counts_reference": str(counts_reference),
                   "ap_identity": ap_identity},
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
    p.add_argument("--ckpts", nargs="+", type=Path, required=True,
                   help="the ANCHOR checkpoints -- seeds 0-19, the ones with AP")
    p.add_argument("--counts-reference", type=Path, required=True,
                   help="rf_counts_theta070_n19.h5. The per-anchor counts this "
                        "run computes must be BIT-IDENTICAL to its "
                        "`match_count`, in NUMERIC seed order. Non-zero exit "
                        "if not.")
    p.add_argument("--expect-ckpts", type=int, default=None,
                   help="refuse unless EXACTLY this many checkpoints were "
                        "supplied -- an identity check, not an existence one")
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

    out = run(args.ap, args.null, args.ckpts, args.out,
              args.counts_reference, args.device, args.theta)

    r = out["rates"]
    print(f"\n[2b] over {r['n_concept_seedpair_cells_evaluated']:,} "
          f"(concept, seed-pair) cells where the concept is detected in both, "
          f"across {r['n_unordered_seed_pairs']} anchor pairs:")
    print(f"     B  carrier match (partner IS the other carrier) "
          f"{r['rate_B_carrier_match']:.4f}")
    print(f"     A  carrier has ANY partner                      "
          f"{r['rate_A_carrier_has_any_partner']:.4f}")
    print(f"     B|A conditional                                 "
          f"{r['conditional_B_given_A']:.4f}")
    print(f"     ambient (same pair set, n_compare="
          f"{r['n_compare_seeds_for_ambient']})            "
          f"{r['ambient_match_rate_same_pair_set']:.4f}")
    print(f"     firing-rate-matched ambient                     "
          f"{r['firing_rate_matched_ambient']:.4f}")
    pr = out["predictions"]
    print(f"\n  AY-P1 {pr['AY_P1']['outcome']} "
          f"(B/matched = {pr['AY_P1']['ratio']:.4f}); like-for-like "
          f"A/matched = {pr['AY_P1']['like_for_like_A_vs_matched']['ratio']:.4f} "
          f"-> {pr['AY_P1']['like_for_like_A_vs_matched']['outcome']}")
    print(f"  AY-P2 {pr['AY_P2']['outcome']} "
          f"(share of concepts above 0.9 = "
          f"{pr['AY_P2']['measured_share_above_0_9']:.2%})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
