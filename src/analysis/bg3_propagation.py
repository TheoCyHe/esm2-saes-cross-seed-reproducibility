"""
AC7 -- does BG2's collision-induced positive loss reach the statistic, and
what is BG3's COMBINED anti-conservative bound?

POST-HOC BY CONSTRUCTION, and that is not a concession. The instrument
correlating null maxima against a draw's realised prevalence takes the STORED
MAXIMA as an argument, and for the experimental stratum those do not exist
until the null has run. It was post-hoc for the pooled null too.

--------------------------------------------------------------------------
ITEM 1 -- does the loss reach the statistic?
--------------------------------------------------------------------------
`concept_headline_checks.null_max_vs_prevalence` is the existing instrument
and is reused unmodified. It was located by READING THE SOURCE, not by name:
a report, a planning brief and a draft amendment all named a function
(`positives_under_permutation`) that does not exist anywhere in the tree, and
re-reading showed the real one (`realised_positive_counts`) returns counts
rather than positions and hardcodes the cross-protein path, so it could not
have served the purpose it was cited for either (gotcha 34).

We do not need `realised_positive_counts` here at all: brief B section 4b
required the surviving counts to be STORED in the null artifact, so the
quantity is read rather than recomputed -- which also means this check
measures what the null DID rather than what a re-derivation would do.

--------------------------------------------------------------------------
ITEM 2 -- the COMBINED bound, and why the pooled form transfers
--------------------------------------------------------------------------
The brief warns against inheriting `1/(1 - drift)`: it was derived for
PREVALENCE drift, and transporting it to POSITIVE-COUNT loss is a different
reduction. So the form is established here rather than assumed.

    Under BG1 THE MASKED SET IS HELD FIXED ACROSS DRAWS. Therefore `n_eff` is
    a constant of the concept, and the draw's prevalence is

        p_draw = surviving / n_eff        with n_eff fixed.

    The relative prevalence drift is then

        p_draw / p_obs - 1
          = (surviving / n_eff) / (observed / n_eff) - 1
          = surviving / observed - 1,

    which is EXACTLY the relative positive-count loss. The `n_eff` cancels.

So on this stratum the two are not two quantities to compose -- they are ONE
quantity measured once, and the pooled functional form applies to it
unchanged. THAT IS WHY THE BOUND IS COMBINED RATHER THAN A PRODUCT OF TWO
FACTORS: composing them would double-count.

The composition the brief separately forbids -- adding the POOLED stratum's
recorded figures (median -1.83%, worst -15.85%) to the experimental ones -- is
a different error and is not committed here: those were measured on 703
concepts of a different stratum, and nothing below reads them.

WHAT IS AND IS NOT SEPARABLE WITHIN THE ONE FIGURE. Null B's surviving count
loses positives two ways -- the many-to-one rescaling dedup, and BG2's mask
collision -- and the stored count contains both INSEPARABLY. Null A's circular
shift is a bijection within a protein, so it has no dedup loss and its figure
is collision-only. The two arms are therefore reported separately and never
pooled, and the Null A figure is the one that isolates the collision effect.

THE BOUND'S PREMISE IS ITEM 1'S MEASUREMENT. `1/(1 - delta)` bounds the
understatement IF the detection threshold scales with prevalence. Item 1 is
what establishes whether it does. If the correlation is near zero the bound
does not bind and is merely conservative -- which is a different statement
from the bound being small, and both are reported.

--------------------------------------------------------------------------
BH2's EXCLUSION REACHES THE BOUND -- a defect, and its repair
--------------------------------------------------------------------------
**Until this was fixed the bound was computed over EVERY cell, including the
cells BH2 had already dropped.** Item 1 masked correctly (`keep = surv > 0`,
below) and `anti_conservative_bound` did not, so a draw that is not in the
null distribution at all was nonetheless setting the reported worst case.
Under Null B that is a loss of exactly -1.0, `1/(1 + -1.0)` is infinite, and
**Arm B's combined bound reported as inf** -- driven by a cell BH2 excludes.

The repair is that ONE derivation of the exclusion feeds BOTH: the mask comes
from `surviving_draw_mask`, which implements BH1's criterion -- ZERO SURVIVING
POSITIVES -- and names no cell. **The known offender is not hardcoded**: a rule
written against today's answer passes today and fails silently on the next run.

THREE THINGS THAT ARE ENFORCED RATHER THAN ASSERTED IN PROSE:

  * A degenerate cell that reaches the bound raises `DegenerateDrawNotExcluded`
    rather than producing an infinity. The test docstring used to say such a
    cell "should never reach here in practice" -- an invariant stated in a
    comment with nothing checking it, which is the shape this project keeps
    catching (gotcha 13). `bound_factor` stays TOTAL over its argument
    regardless (gotcha 24), and is tested directly, so the totality branch is
    not lost with the addition of the guard.
  * BH2's criterion is checked against an INDEPENDENT derivation of the same
    set -- the stored null's all-NaN cells -- by `exclusion_derivations_agree`,
    which REFUSES on disagreement instead of preferring one of them.
  * BH3's outcome space is mapped in CODE, by reusing `null_pilot.classify_bh3`
    rather than reimplementing it. Two copies of a rule make their agreement a
    statement about the copies (gotcha 16). BH3's second and third cases --
    all draws degenerate under one arm, and under both -- are currently
    UNOCCUPIED, and they are mapped anyway: a category that exists cannot
    silently become something else on a re-run, which is why the null artifact
    carries `not_evaluable_*` even when it is empty.

`NOT_EVALUABLE` IS NOT A VALUE. It is never folded into a bound, a worst case
or a denominator (BH3). Where a concept has no surviving draw at all it is
excluded from the per-concept percentiles and COUNTED SEPARATELY; where an
entire arm has none, the bound itself is `NOT_EVALUABLE` rather than zero,
infinite, or a division over an empty set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.concept_headline_checks import null_max_vs_prevalence
from src.analysis.null_pilot import (
    BH3_BOTH_ARMS,
    BH3_NONE,
    BH3_ONE_ARM,
    BH3_SOME,
    classify_bh3,
)
from src.sae.train import git_dirty, git_sha


class DegenerateDrawNotExcluded(RuntimeError):
    """A (concept, draw) cell with ZERO surviving positives reached the bound.

    BH2 drops such a cell from the null distribution, so its arrival here means
    the exclusion did not happen -- the mask was not passed, was built from the
    wrong array, or was built from a different criterion. Raised rather than
    returned, because the alternative is an infinity that looks like a result.
    """


class ExclusionDerivationsDisagree(RuntimeError):
    """BH2's criterion and the stored null's NaN pattern select different
    cells. One of the two is wrong and this module cannot tell which, so it
    refuses rather than preferring either."""


def surviving_draw_mask(surviving: np.ndarray) -> np.ndarray:
    """BH1's criterion, elementwise: a draw SURVIVES iff at least one observed
    positive survived the displacement. True where the cell is USABLE.

    Derived from the criterion and from nothing else -- no concept, no shift
    and no count is named here, so a future run whose degenerate cells are
    elsewhere is masked correctly without an edit.
    """
    return np.asarray(surviving) > 0


def bound_factor(d: float) -> float:
    """`1 / (1 + d)` for a relative loss `d`, TOTAL over its argument.

    A loss of exactly -1 (every positive lost) makes the factor infinite, and
    that is reported rather than divided (gotcha 24). The totality is kept even
    though `anti_conservative_bound` now refuses to hand it such a value: a
    guard upstream is not a reason for a partial function downstream, and this
    branch is tested directly rather than through the guard.
    """
    return float("inf") if d <= -1.0 else 1.0 / (1.0 + d)


def exclusion_derivations_agree(surviving: np.ndarray,
                                cmax: np.ndarray) -> dict:
    """Two INDEPENDENT derivations of BH2's exclusion set, compared cell by
    cell.

    `surviving` is (concepts, shifts) and gives BH1's criterion directly.
    `cmax` is (seeds, shifts, concepts): a degenerate cell has no maximum in
    ANY seed, so it is all-NaN down the seed axis. The two are produced by
    different code at different times -- the displacement plan versus the null
    run's own fill value -- so their agreement is evidence and their
    disagreement is a defect in one of them.

    A COUNT would agree under any permutation of either axis (gotcha 27), so
    this compares positionally and names the offending cells.
    """
    by_criterion = ~surviving_draw_mask(surviving)          # (concepts, shifts)
    by_stored_nan = np.isnan(cmax).all(axis=0).T            # (concepts, shifts)
    if by_criterion.shape != by_stored_nan.shape:
        raise ExclusionDerivationsDisagree(
            f"surviving_positives is {surviving.shape} and null_cmax "
            f"transposes to {by_stored_nan.shape} -- the two cannot be "
            f"compared cell by cell, so BH2's exclusion is unverified")
    disagree = np.argwhere(by_criterion != by_stored_nan)
    return {
        "n_degenerate_by_bh1_criterion": int(by_criterion.sum()),
        "n_degenerate_by_stored_nan": int(by_stored_nan.sum()),
        "agree": bool(disagree.size == 0),
        "first_disagreements_(concept,shift)": disagree[:20].tolist(),
    }


def relative_positive_loss(surviving: np.ndarray,
                           observed: np.ndarray) -> np.ndarray:
    """`surviving / observed - 1` per (concept, draw). 0 is exact
    preservation; negative is loss. Under BG1's fixed mask this IS the
    relative prevalence drift (see the module docstring's cancellation)."""
    return surviving.astype(np.float64) / observed.astype(np.float64)[:, None] - 1.0


def anti_conservative_bound(loss: np.ndarray, keep: np.ndarray) -> dict:
    """BG3's bound: a relative loss of `-delta` understates a
    prevalence-scaled threshold by at most `1 / (1 - delta)`.

    `keep` is BH2's surviving-draw mask, same shape as `loss`, and it is
    REQUIRED. The bound is computed over surviving draws only -- a degenerate
    draw is not in the null distribution, so it cannot set the null
    distribution's worst case.

    Reported at the WORST and at the MEDIAN, because the two answer different
    questions and the worst is what the word 'bound' means. A ratio is
    reported only alongside both of its components (gotcha 33).

    A concept with NO surviving draw is `NOT_EVALUABLE` (BH3): it is dropped
    from the per-concept percentiles and counted separately, never entered as
    a zero and never counted in a denominator. An ARM with no surviving draw
    anywhere returns `NOT_EVALUABLE` as its status.
    """
    loss = np.asarray(loss, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != loss.shape:
        raise ValueError(f"keep mask is {keep.shape} against loss {loss.shape}")
    usable = keep & np.isfinite(loss)
    not_evaluable = ~usable.any(axis=1)             # per concept, BH3
    n_excluded = int((~keep).sum())
    finite = loss[usable]
    if finite.size == 0:
        # Not zero, not infinity, and not a division over an empty set.
        return {"status": "NOT_EVALUABLE",
                "reason": "every draw is degenerate for every concept in this "
                          "arm, so the arm has no reference distribution and "
                          "therefore no bound (BH3). NOT_EVALUABLE is a "
                          "statement of ignorance, not a value",
                "n_cells": 0,
                "n_cells_excluded_degenerate": n_excluded,
                "n_concepts_not_evaluable": int(not_evaluable.sum())}
    if (finite <= -1.0).any():
        bad = np.argwhere(usable & (loss <= -1.0))
        raise DegenerateDrawNotExcluded(
            f"{len(bad)} cell(s) with a relative loss <= -1 survived BH2's "
            f"mask, e.g. (concept, shift) {bad[:5].tolist()}. A loss of -1 is "
            f"ZERO surviving positives, which BH2 drops; reaching the bound "
            f"means the exclusion was not applied to this array")
    worst = float(finite.min())              # most negative
    med = float(np.median(finite))
    per_concept_worst = np.nanmin(
        np.where(usable, loss, np.nan)[~not_evaluable], axis=1)

    return {
        "status": "COMPUTED",
        "form": "1 / (1 + relative_loss), i.e. 1/(1-delta) for a loss of -delta",
        "form_justification": "under BG1 the masked set is fixed, so n_eff is "
                              "constant per concept and the relative "
                              "prevalence drift EQUALS the relative "
                              "positive-count loss; the pooled form therefore "
                              "applies to this quantity unchanged rather than "
                              "being transported to a different reduction",
        "n_cells": int(finite.size),
        "n_cells_excluded_degenerate": n_excluded,
        "n_concepts_not_evaluable": int(not_evaluable.sum()),
        "n_concepts_in_percentiles": int(per_concept_worst.size),
        "exclusion_note": "the bound is over SURVIVING draws only (BH2). A "
                          "degenerate draw is dropped, never imputed, and a "
                          "NOT_EVALUABLE concept is neither a zero nor a "
                          "member of any denominator here (BH3)",
        "worst_relative_loss": worst,
        "median_relative_loss": med,
        "mean_relative_loss": float(finite.mean()),
        "frac_cells_with_no_loss": float((finite == 0.0).mean()),
        "bound_at_worst": bound_factor(worst),
        "bound_at_median": bound_factor(med),
        "worst_per_concept_loss_percentiles": {
            f"p{q:02d}": float(np.percentile(per_concept_worst, q))
            for q in (0, 1, 5, 10, 25, 50, 75, 90, 100)},
        "bound_at_p99_of_per_concept_worst": bound_factor(
            float(np.percentile(per_concept_worst, 1))),
    }


def worst_cells(loss: np.ndarray, keep: np.ndarray, concepts: list[str],
                n: int = 5) -> list[dict]:
    """The `n` most-negative SURVIVING cells, named. The worst is the figure
    the bound reports; the runner-up is the context that says whether the worst
    is a cliff or a shoulder, and a bound quoted without it has been read as
    typical here before."""
    loss = np.asarray(loss, dtype=np.float64)
    usable = np.asarray(keep, dtype=bool) & np.isfinite(loss)
    flat = np.where(usable, loss, np.inf).ravel()
    order = np.argsort(flat, kind="stable")[:n]
    out = []
    for j in order:
        if not np.isfinite(flat[j]):
            break                                   # fewer than n survivors
        ci, si = divmod(int(j), loss.shape[1])
        out.append({"concept": concepts[ci], "shift": si,
                    "relative_loss": float(loss[ci, si]),
                    "bound": bound_factor(float(loss[ci, si]))})
    return out


def bh3_outcomes(surviving_by_arm: dict[str, np.ndarray],
                 concepts: list[str], n_shifts: int) -> dict:
    """BH3's outcome grid, per concept, over whichever arms are present.

    `null_pilot.classify_bh3` is REUSED rather than reimplemented: it is the
    rule the null artifact's own `bh3_case` dataset was written from, and a
    second copy here would make any agreement between them a statement about
    the copies (gotcha 16).
    """
    degen = {arm: (~surviving_draw_mask(s)).sum(axis=1)
             for arm, s in surviving_by_arm.items()}
    cases = [classify_bh3(None if "a" not in degen else int(degen["a"][i]),
                          None if "b" not in degen else int(degen["b"][i]),
                          n_shifts)
             for i in range(len(concepts))]
    from collections import Counter
    occupancy = dict(Counter(cases))
    # Every case named, including the ones with no members. An empty category
    # that EXISTS cannot silently become something else on a re-run.
    for case in (BH3_NONE, BH3_SOME, BH3_ONE_ARM, BH3_BOTH_ARMS):
        occupancy.setdefault(case, 0)
    thin = {arm: int((np.asarray(s) == 1).sum())
            for arm, s in surviving_by_arm.items()}
    return {
        "occupancy": occupancy,
        "concepts_by_case": {
            case: [concepts[i] for i, c in enumerate(cases) if c == case]
            for case in (BH3_SOME, BH3_ONE_ARM, BH3_BOTH_ARMS)},
        "n_cells_with_exactly_one_surviving_positive": thin,
        "single_positive_note": "BH3's fourth case: AP is DEFINED for a single "
                                "surviving positive -- the reciprocal of its "
                                "rank -- so NO RULE APPLIES and none is "
                                "imposed. Counted because whether thin draws "
                                "inflate null variance is a question for "
                                "measurement, and the count is the input to it",
    }


def run(null_path: Path, out_path: Path) -> dict:
    with h5py.File(null_path, "r") as hf:
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in hf["concepts"][:]]
        observed = hf["n_positives_observed"][:]
        n_eff = hf["n_rows_effective"][:]
        arms = {}
        for arm in ("a", "b"):
            key = f"surviving_positives_{arm}"
            if key in hf and f"null_cmax_{arm}" in hf:
                arms[arm] = (hf[key][:], hf[f"null_cmax_{arm}"][:])
        identity = {k: (v.decode() if isinstance(v, bytes) else
                        v.item() if isinstance(v, np.generic) else v)
                    for k, v in hf.attrs.items()}

    report: dict = {
        "artifact": "ac7_bg3_propagation",
        "governing_amendment": "BG3 items 1-3",
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "null": str(null_path),
        "identity": {k: identity.get(k) for k in (
            "stratum", "negatives", "library_sha", "n_concepts",
            "argmax_pool_scored_n", "argmax_pool_used_n", "truncation_policy",
            "scored_row_set", "n_shifts", "null_seed", "ap_impl", "n_rows")},
        "n_concepts": len(concepts),
        "bh3": bh3_outcomes({a: s for a, (s, _) in arms.items()}, concepts,
                            int(identity.get("n_shifts", 0)) or
                            (next(iter(arms.values()))[0].shape[1]
                             if arms else 0)),
        "arms": {},
    }

    for arm, (surv, cmax) in arms.items():
        loss = relative_positive_loss(surv, observed)
        # ONE derivation of BH2's exclusion, feeding item 1 AND the bound.
        keep_all = surviving_draw_mask(surv)

        # AC1-3: the same set, derived a second way from the stored null.
        # A disagreement means BH2's implementation or the null's NaN handling
        # is not what the record says, and that is not this module's to repair.
        agreement = exclusion_derivations_agree(surv, cmax)
        if not agreement["agree"]:
            raise ExclusionDerivationsDisagree(
                f"arm {arm}: BH1's criterion marks "
                f"{agreement['n_degenerate_by_bh1_criterion']} degenerate "
                f"cell(s) and the stored null is all-NaN in "
                f"{agreement['n_degenerate_by_stored_nan']}; first "
                f"disagreements (concept, shift) "
                f"{agreement['first_disagreements_(concept,shift)']}")

        # ---- item 1: does the loss reach the statistic? -----------------
        per_concept, statuses = {}, []
        for ci, c in enumerate(concepts):
            # A degenerate draw carries no maximum. Drop it from BOTH sides
            # rather than correlating against a NaN -- BH2 drops the draw, and
            # a check that quietly kept it would be scoring an imputed value.
            keep = keep_all[ci]
            if keep.sum() < 2:
                per_concept[c] = {"status": "UNINFORMATIVE_TOO_FEW_SURVIVING_DRAWS",
                                  "n_surviving_draws": int(keep.sum())}
                statuses.append(per_concept[c]["status"])
                continue
            r = null_max_vs_prevalence(surv[ci][keep], int(n_eff[ci]),
                                       cmax[:, keep, ci])
            per_concept[c] = r
            statuses.append(r["status"])

        computed = [v for v in per_concept.values() if v["status"] == "COMPUTED"]
        g = np.array([v["median_spearman_prevalence_vs_null_max_GAIN"]
                      for v in computed
                      if v["median_spearman_prevalence_vs_null_max_GAIN"] is not None])
        a = np.array([v["median_spearman_prevalence_vs_null_max_AP"]
                      for v in computed
                      if v["median_spearman_prevalence_vs_null_max_AP"] is not None])
        from collections import Counter
        scaling = {
            "n_concepts_computed": len(computed),
            "status_counts": dict(Counter(statuses)),
            "note": "A concept whose surviving count is CONSTANT across draws "
                    "has no variation to rank against, and a ranking predicate "
                    "applied to a constant ranks the tie-break. Those are "
                    "UNINFORMATIVE, never a correlation of zero -- absent "
                    "variation is not absent association (gotcha 25b).",
            "median_spearman_GAIN": float(np.median(g)) if g.size else None,
            "median_spearman_AP": float(np.median(a)) if a.size else None,
            "mean_spearman_GAIN": float(g.mean()) if g.size else None,
            "mean_spearman_AP": float(a.mean()) if a.size else None,
            "iqr_spearman_GAIN": ([float(np.percentile(g, 25)),
                                   float(np.percentile(g, 75))]
                                  if g.size else None),
            "iqr_spearman_AP": ([float(np.percentile(a, 25)),
                                 float(np.percentile(a, 75))]
                                if a.size else None),
        }

        report["arms"][arm.upper()] = {
            "loss_contains": ("BG2 mask collisions ONLY -- Null A's "
                              "within-protein circular shift is a bijection, "
                              "so it has no many-to-one dedup loss"
                              if arm == "a" else
                              "BG2 mask collisions AND the many-to-one "
                              "rescaling dedup, INSEPARABLY"),
            "bh2_exclusion": {
                **agreement,
                "degenerate_cells": [
                    {"concept": concepts[ci], "shift": int(si),
                     "observed_positives": int(observed[ci])}
                    for ci, si in np.argwhere(~keep_all)],
                "surviving_draws_min": int(keep_all.sum(axis=1).min()),
                "surviving_draws_max": int(keep_all.sum(axis=1).max()),
            },
            "scaling_check": scaling,
            "combined_bound": anti_conservative_bound(loss, keep_all),
            "worst_surviving_cells": worst_cells(loss, keep_all, concepts),
            "per_concept_scaling": per_concept,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    return report


def format_summary(report: dict) -> str:
    """Total over every shape the report can take -- a completed measurement
    has been reported as a FAILED job because a summary print died on a
    legitimately undefined value (BA17 item 3)."""
    out = [f"AC7 / BG3 -- {report.get('null', 'UNKNOWN')}",
           f"  concepts: {report.get('n_concepts', 'ABSENT')}"]
    bh3 = report.get("bh3")
    if bh3 is not None:
        out.append(f"  BH3 occupancy: {bh3.get('occupancy', 'ABSENT')}")
        out.append(f"  cells with exactly one surviving positive "
                   f"(no rule applies): "
                   f"{bh3.get('n_cells_with_exactly_one_surviving_positive', 'ABSENT')}")
    for arm, blk in report.get("arms", {}).items():
        out.append(f"  --- Null {arm} ---")
        out.append(f"    loss contains: {blk.get('loss_contains', 'ABSENT')}")
        ex = blk.get("bh2_exclusion")
        if ex is not None:
            cells = ex.get("degenerate_cells", [])
            out.append(f"    BH2 degenerate draws excluded: {len(cells)}"
                       + ("" if not cells else "  " + "; ".join(
                           f"{d['concept']} shift {d['shift']} "
                           f"({d['observed_positives']} observed)"
                           for d in cells[:5])))
            out.append(f"    exclusion set agrees with the stored null's NaN "
                       f"pattern: {ex.get('agree', 'ABSENT')}")
        sc = blk.get("scaling_check", {})
        for unit in ("GAIN", "AP"):
            v = sc.get(f"median_spearman_{unit}")
            iqr = sc.get(f"iqr_spearman_{unit}")
            out.append(f"    median Spearman(surviving count, null max {unit}): "
                       + ("UNDEFINED" if v is None else f"{v:+.4f}")
                       + ("" if not iqr else
                          f"  IQR [{iqr[0]:+.4f}, {iqr[1]:+.4f}]"))
        out.append(f"    concepts with a computable correlation: "
                   f"{sc.get('n_concepts_computed', 'ABSENT')}; statuses "
                   f"{sc.get('status_counts', {})}")
        b = blk.get("combined_bound", {})
        if b.get("status") != "COMPUTED":
            out.append(f"    combined bound: {b.get('status', 'ABSENT')}")
            continue
        out.append(f"    worst relative positive loss: "
                   f"{b['worst_relative_loss']:+.4%}   median "
                   f"{b['median_relative_loss']:+.4%}")
        out.append(f"    cells with NO loss: {b['frac_cells_with_no_loss']:.2%}"
                   f"   over {b.get('n_cells', 'ABSENT')} surviving cells "
                   f"({b.get('n_cells_excluded_degenerate', 'ABSENT')} "
                   f"excluded by BH2); NOT_EVALUABLE concepts "
                   f"{b.get('n_concepts_not_evaluable', 'ABSENT')}")
        out.append(f"    COMBINED anti-conservative bound "
                   f"= 1/(1+loss): at worst {b['bound_at_worst']:.4f}x, "
                   f"at median {b['bound_at_median']:.4f}x")
        for w in blk.get("worst_surviving_cells", [])[:2]:
            out.append(f"      driven by {w['concept']} shift {w['shift']}: "
                       f"loss {w['relative_loss']:+.4%} -> {w['bound']:.4f}x")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = run(args.null, args.out)
    print(format_summary(report))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
