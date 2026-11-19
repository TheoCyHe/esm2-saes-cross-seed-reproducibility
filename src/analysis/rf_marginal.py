"""
M1 -- the `r_f` MARGINAL DISTRIBUTIONS. Governed by PREREGISTRATION amendment X.

WHAT `r_f` IS. The per-feature cross-seed reproducibility rate: for an anchor
feature `f` in seed `s`, the fraction of the other 19 seeds in which `f` has a
mutual-nearest-neighbour partner at decoder-cosine >= theta. It is the OUTCOME
variable of the primary regression (amendment F). [Docstring dated 2026-09-16:
the two sentences this paragraph used to end with -- "it has never been
computed" and "`metrics.feature_level.threshold` is still `null` in the
config" -- described the state before amendment X ran. `r_f` has been
measured at `n_compare` = 19 and 39 (`rf_counts_theta070_n19.h5`,
`_n39.h5`); `configs/mvp.yaml` sets `threshold: 0.70` and records the tiebreak
as retired, not retuned. `selected_theta` below is still computed and
reported; it no longer selects anything.]

WHY THIS IS A GATE AND NOT A STEP (amendment W3). Borobia et al.
(arXiv:2603.25325) measure the cross-seed MNN rate at tau = 0.7 at 1.1%-3.9%,
using OUR rule (MNN), at OUR expansion factor (8x), on TopK SAEs. If ours
lands there, `r_f = 0` for nearly every feature, the stable endpoint is empty,
and the primary regression has NO OUTCOME VARIANCE. That is the identical
failure that killed pooled exceedance as the regression predictor -- a ceiling
collapsing the signal -- transposed to the outcome side. So the marginal is
measured, and its shape decides the design, BEFORE anything consuming `r_f`
is built.

=============================================================================
PREDICTIONS -- COMMITTED TO GIT BEFORE THE JOB IS SUBMITTED (amendment X9)
=============================================================================
This docstring is the audit trail. `git log` establishes it preceded the run.

  P1. CORRECTNESS ASSERTION, not a prediction. The MNN match count at
      theta = 0.7 is <= the one-way count at the same theta for EVERY one of
      the 190 pairs and every anchor. Amendment W1 establishes MNN is nested
      inside one-way argmax-cosine, so a violation is an implementation bug.
      The job FAILS on violation rather than reporting it.

  P2. The MNN rate at theta = 0.7 lands in [0.5%, 12%] -- within ~3x of
      Borobia's 1.1%-3.9% band in either direction.

  P3. The rotation-null 99.9th percentile lands BELOW 0.30, i.e. below the old
      geometry-derived value and below the entire theta grid.
      Reasoning, recorded so the prediction can be scored honestly: at
      d = 1280 the cosine between independent random unit vectors has
      SD ~ 1/sqrt(1280) = 0.0280, and the max over 10,240 draws sits near
      0.0280 * sqrt(2 ln 10240) ~ 0.26. This is ARITHMETIC ABOUT A NULL, not a
      mechanism story -- but a real decoder is not a set of independent random
      vectors, so the null is MEASURED and the formula is recorded only as the
      reason for the predicted direction.

  P4. NO theta on the grid satisfies amendment X2's usability clause: the
      `r_f = 0` bin exceeds 80% of features everywhere the floor is cleared.
      THIS IS A PREDICTION THAT THE DESIGN FAILS. It is committed in advance
      precisely so that a floor effect cannot afterwards be narrated as an
      unlucky threshold choice. If it is wrong, that is the better outcome and
      the gotcha-22 ledger records a miss.

  P5. Hungarian-vs-one-way IoU at theta = 0.7 EXCEEDS MNN-vs-one-way IoU,
      both computed by us on the same subsample. (Both Hungarian and one-way
      assign every feature a partner; MNN does not -- amendment W1.)
=============================================================================

THE FIREWALL (amendment X7), and it is STRUCTURAL rather than a promise.
Selecting theta on the MARGINAL distribution of `r_f` is legitimate
pre-registration; selecting it on the `r_f`-to-alignment relationship is
gaming. So this module installs an `h5py.File` guard that HARD-FAILS if any
AP matrix, null artifact, or residue-label store is opened, and it computes no
join between `r_f` and any alignment quantity. Separate script, separate
output, separate job.

WHAT IS MEASURED, AND WHAT IS NOT (amendment X6). This module uses DECODER
COSINE alone. The config specifies
`similarity: min(activation_correlation, decoder_cosine)`, which needs an
activation pass over the eval cache. Note carefully what is NOT claimed:
decoder cosine is NOT an upper bound on the `min(...)` rate, because MNN
membership depends on which partner is the argmax and the argmax under
`min(corr, cos)` need not be the argmax under `cos`. Neither rate bounds the
other analytically. Decoder cosine is used because it is the arm every
comparable measurement in the literature uses (Gerasimov, Borobia, Leask) and
because calibrating theta against a rotation null is only well-posed for a
geometric similarity. The resolution is flagged for the author in X6, not taken here.

A PROPERTY OF MNN WORTH STATING (amendment X5). MNN groups are NOT equivalence
classes: the relation is symmetric but not transitive, so anchor A can match
B and C while B matches only A. `m_f = 1 + matches` sidesteps this by
ANCHORING -- every count is taken from one fixed feature's point of view.

IMPLEMENTATION NOTE that makes the whole theta grid free. Mutual-best-ness
does NOT depend on theta: MNN at theta is {(i,j) : j = argmax_j C[i,:],
i = argmax_i C[:,j], C[i,j] >= theta}, and the two argmaxes are computed
without reference to theta. So per ordered seed pair we store just two vectors
-- the row max (which decides one-way at every theta) and the mutual-pair
similarity or -inf (which decides MNN at every theta) -- and every threshold
is then a comparison, not a recomputation.

=============================================================================
THE ANCHOR / COMPARISON SPLIT (amendments AB4, AD5, AK). Added for N = 39.
=============================================================================
`--ckpts` are the ANCHORS and `--compare-ckpts` are comparison-only
dictionaries. Anchors are the features `r_f` is measured FOR; comparison seeds
are dictionaries `r_f` is measured AGAINST. AB4 fixes the anchors at seeds
0-19 because those are the seeds with AP, and AD5 puts the extension seeds in
a separate directory so extending N stays a deliberate act.

Consequences, all of them arithmetic:
  * `n_compare = n_anchors + n_compare_only - 1` (39 at S = 40), because an
    anchor is compared against every OTHER dictionary including the other
    anchors.
  * Pairs actually computed = anchor-anchor + anchor-to-extension
    = 190 + 400 = **590** at S = 40, against 190 before.
    Extension-extension pairs (another 190) are NOT computed: an extension
    seed is never an anchor, so its row is never read.
  * Every reduction is taken over ANCHOR ROWS ONLY (`anchor_off`), so a
    non-anchor's incomplete row can never enter a statistic. With no
    `--compare-ckpts` every row is an anchor and `anchor_off == off`, which is
    what keeps the N = 19 path bit-identical.

THE GLOB HAZARD THIS EXISTS TO DEFEAT (docs/CODEBASE.md, AD5). Nine sbatch
scripts glob `sae_ckpts/seed_*`. A run that picks up that glob alone measures
the 20 anchors and labels the result N = 39 -- internally consistent, and
describing the wrong ensemble. The count guard therefore asserts EXACTLY the
expected total; a self-consistency check passes at 20.

=============================================================================
PER-ANCHOR COUNTS ACROSS THE WHOLE THETA GRID (amendment AR, route A)
=============================================================================
`--emit-counts` writes `match_count` at the REFERENCE theta only, so `r_fs`
-- the design-3 regression's outcome -- existed at theta = 0.70 and nowhere
else. `marginals_by_theta` covers all 13 grid points but stores MARGINALS,
not per-feature counts, so it cannot carry a regression. That collides with
AB1 decision A and with AK6's reporting standard: reporting the whole
qualifying set rather than a selected member is the declared substitute for
multiplicity control, and it cannot be met from one theta.

`--emit-counts-all-theta` therefore writes one `match_count_theta_X.XX`
dataset per grid point into a single h5. It costs nothing: mutual-best-ness
does not depend on theta (see the implementation note above), so every grid
point is a comparison against the SAME `mnn_sim`, not a recomputation.

TWO PROPERTIES OF THIS ADDITION ARE LOAD-BEARING.

  * `--emit-counts` and its output path are UNTOUCHED in behaviour. Both
    paths now call ONE helper, `counts_at_theta`, so the reference slice
    cannot drift from the single-theta artifact by construction rather than
    by inspection.
  * `--counts-reference` makes that construction CHECKABLE against the
    artifact already on disk: the theta = 0.70 dataset must be BIT-IDENTICAL
    to `match_count` in the existing single-theta file, and the job exits
    non-zero if it is not (gotcha 13 extension 1). The criterion was declared
    in amendment AR before the edit was written, and the comparison is a
    POSITIONAL, elementwise one -- a shape or total check cannot see a
    permutation (gotcha 27).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from src.analysis.artifact_identity import device_class

# ---------------------------------------------------------------------------
# Amendment X2: the pre-declared theta grid. 0.30 to 0.90 step 0.05; 0.70 is
# on the grid exactly (0.30 + 8 * 0.05), which is what X2 requires.
# ---------------------------------------------------------------------------
THETA_GRID = [round(0.30 + 0.05 * i, 2) for i in range(13)]
THETA_LITERATURE = 0.70

# Amendment X2 part 2, and Y5's resolution rule. NOT to be tuned (X2).
ENDPOINT_MAX_SHARE = 0.80
# Amendment X2 part 1. NOT to be tuned (X2).
FLOOR_QUANTILE = 99.9

# ---------------------------------------------------------------------------
# Amendment AG1's ENDPOINT CUTS, as a committed TABLE rather than a formula.
#
# The cuts are proportional in `p`, not absolute in `k`: holding `k <= 1` fixed
# as N doubles would silently tighten the criterion from 0.0526 to 0.0256 -- a
# 2x change in the definition arriving disguised as consistency. AG1 records
# that the proportional mapping lands exactly on integers at both N (0.05263 ->
# 0.05128 and 0.94737 -> 0.94872), so NO ROUNDING RULE WAS NEEDED and none was
# invented.
#
# A TABLE, not a formula, and the refusal is the point: at any other N the
# proportional cut lands between integers and would need a rounding rule that
# nobody pre-registered. Computing one on the fly is exactly the incomplete
# instrument gotcha 24 is about, so this raises instead.
# ---------------------------------------------------------------------------
AG_ENDPOINT_CUTS = {
    19: {"primary_unstable_k_max": 1, "primary_stable_k_min": 18},
    39: {"primary_unstable_k_max": 2, "primary_stable_k_min": 37},
}


def ag_cuts(n_compare: int) -> dict:
    """AG1's pre-declared cuts for this N, or a refusal."""
    if n_compare not in AG_ENDPOINT_CUTS:
        raise KeyError(
            f"amendment AG1 pre-registers endpoint cuts for "
            f"N in {sorted(AG_ENDPOINT_CUTS)} only, not N = {n_compare}. "
            f"The proportional cut would land between integers and needs a "
            f"rounding rule that has not been pre-registered; refusing to "
            f"invent one after the fact.")
    cuts = dict(AG_ENDPOINT_CUTS[n_compare])
    cuts["sensitivity_unstable_k_max"] = 0
    cuts["sensitivity_stable_k_min"] = n_compare
    cuts["n_compare"] = n_compare
    cuts["primary_unstable_p_threshold"] = (
        cuts["primary_unstable_k_max"] / n_compare)
    cuts["primary_stable_p_threshold"] = (
        cuts["primary_stable_k_min"] / n_compare)
    return cuts


# Amendment X7: opening any of these from this module is a firewall breach.
FORBIDDEN_ARTIFACT_TOKENS = (
    "feature_ap",
    "null_pilot",
    "residue_labels",
    "null_pooling",
    "orphan_screen",
)


class FirewallBreach(RuntimeError):
    """Raised when the r_f job touches an alignment artifact (amendment X7)."""


def install_firewall() -> None:
    """Hard-fail if an AP / null / label artifact is opened from this job.

    Amendment X7 requires the firewall be STRUCTURAL rather than a comment
    asserting good behaviour. Patching `h5py.File` is the enforcement point
    because every artifact in this project's analysis tree is an h5 file.
    """
    import h5py

    if getattr(h5py.File, "_rf_firewall", False):
        return
    original = h5py.File

    class GuardedFile(original):  # type: ignore[misc,valid-type]
        _rf_firewall = True

        def __init__(self, name, *args, **kwargs):
            token = str(name).replace("\\", "/").lower()
            for bad in FORBIDDEN_ARTIFACT_TOKENS:
                if bad in token:
                    raise FirewallBreach(
                        f"amendment X7 firewall: the r_f job must not open "
                        f"alignment artifacts, but tried to open {name!r} "
                        f"(matched forbidden token {bad!r})"
                    )
            super().__init__(name, *args, **kwargs)

    h5py.File = GuardedFile  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pre-flight identity (amendment X8 / gotcha 15). An existence check is not an
# identity check, so every one of these compares VALUES across the 20
# checkpoints and an ABSENT key is reported absent, never treated as matching.
# ---------------------------------------------------------------------------

IDENTITY_KEYS = ("git_sha", "config_hash", "b_dec_init_sha256_16", "dict_size",
                 "d_model", "k", "n_steps", "lr", "batch_size")


def verify_ensemble_identity(meta_by_seed: dict[int, dict],
                             accept_git_shas: list[str] | None = None) -> dict:
    """Assert the checkpoints differ ONLY in seed. Returns a report.

    Amendment CS, CS3c: `accept_git_shas` names the EXACT set of `git_sha`
    values the pool may carry (the width sweep's 40,960 arm: seed 0 is the CN4
    probe at eb8657b9..., seeds 1-9 are 42376f67..., and section CN7 established
    the training path byte-identical between the two). A `git_sha` disagreement
    whose distinct set EQUALS the accepted set is recorded as
    DISAGREES_ACCEPTED_BY_FLAG and is not a problem; any other disagreement, a
    superset or a subset, fails as before. Every other key is unchanged.

    CLAUDE.md flags the cross-seed `git_sha` comparison as a post-hoc check
    that is 'worth verifying regardless' and it has never been run. It runs
    here, and it HARD-FAILS -- a validator whose failure cannot propagate is
    worse than no validator (gotcha 13 extension 1).
    """
    report: dict = {"n_seeds": len(meta_by_seed), "keys": {}, "problems": []}
    if not meta_by_seed:
        report["problems"].append("no checkpoints supplied")
        return report

    seeds = sorted(meta_by_seed)
    if seeds != list(range(len(seeds))):
        report["problems"].append(f"seed labels are not 0..N-1: {seeds}")

    for key in IDENTITY_KEYS:
        missing = [s for s in seeds if key not in meta_by_seed[s]]
        if missing:
            # ABSENT is reported as absent, never as agreement (gotcha 15).
            report["keys"][key] = {"status": "ABSENT", "missing_seeds": missing}
            report["problems"].append(f"{key}: absent for seeds {missing}")
            continue
        values = {s: meta_by_seed[s][key] for s in seeds}
        distinct = sorted({json.dumps(v, sort_keys=True) for v in values.values()})
        if len(distinct) == 1:
            report["keys"][key] = {"status": "IDENTICAL", "value": values[seeds[0]]}
        elif (key == "git_sha" and accept_git_shas is not None
              and set(values.values()) == set(accept_git_shas)):
            report["keys"][key] = {"status": "DISAGREES_ACCEPTED_BY_FLAG",
                                   "distinct": distinct,
                                   "accepted_set": sorted(set(accept_git_shas)),
                                   "by_seed": {str(k): v for k, v in values.items()}}
        else:
            report["keys"][key] = {"status": "DISAGREES", "distinct": distinct}
            report["problems"].append(f"{key}: {len(distinct)} distinct values")

    dirty = {s: meta_by_seed[s].get("git_dirty") for s in seeds}
    report["git_dirty"] = dirty
    if any(v is not False for v in dirty.values()):
        report["problems"].append(f"git_dirty not False for all seeds: {dirty}")

    # Derived verdict, never an asserted sentence (gotcha 20).
    report["verdict"] = "PASS" if not report["problems"] else "FAIL"
    return report


def verify_b_dec_distinctness(meta_by_seed: dict[int, dict]) -> dict:
    """The design's invariant: ONE shared `b_dec` init, N DISTINCT final values.

    Trainable-from-shared-init is what CLAUDE.md's open question 5 settled, and
    `verify_ensemble_identity` above only checks the INIT is shared. The other
    half -- that the final values actually diverged -- has never been asserted
    in code, only reported in prose. A collision would mean two seeds converged
    to the same bias, which is not the ensemble the design describes.

    Reported as a DERIVED verdict, and an absent key is reported absent rather
    than treated as agreement (gotcha 15).
    """
    seeds = sorted(meta_by_seed)
    missing = [s for s in seeds if "b_dec_sha256_16" not in meta_by_seed[s]]
    if missing:
        return {"status": "ABSENT", "missing_seeds": missing,
                "verdict": "FAIL",
                "note": "absent is not matching -- gotcha 15"}
    vals = {s: meta_by_seed[s]["b_dec_sha256_16"] for s in seeds}
    n_distinct = len({*vals.values()})
    dupes: dict[str, list[int]] = {}
    for s, v in vals.items():
        dupes.setdefault(v, []).append(s)
    collisions = {v: ss for v, ss in dupes.items() if len(ss) > 1}
    return {
        "status": "PRESENT",
        "n_seeds": len(seeds),
        "n_distinct_final_b_dec": n_distinct,
        "collisions": collisions,
        "verdict": "PASS" if n_distinct == len(seeds) else "FAIL",
    }


def check_unit_norm(W: torch.Tensor, tol: float = 1e-3) -> dict:
    """Decoder columns are documented unit-norm. Measure it; do not trust it.

    Gotcha 15's shape applied to a numeric property: the stated spec of an
    artifact is not its identity.
    """
    norms = W.norm(dim=0).double()
    dev = (norms - 1.0).abs()
    return {
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "max_abs_deviation": float(dev.max()),
        "tol": tol,
        "verdict": "PASS" if float(dev.max()) <= tol else "FAIL",
    }


# ---------------------------------------------------------------------------
# Matching rules
# ---------------------------------------------------------------------------

def pair_statistics(Wa: torch.Tensor, Wb: torch.Tensor) -> dict[str, torch.Tensor]:
    """Both directions of the one-way and MNN statistics for one seed pair.

    `Wa`, `Wb` are (d_model, dict_size) with unit-norm columns, so the full
    cosine matrix is simply `Wa.T @ Wb`.

    Returns, for each direction, two vectors of length dict_size:
      `oneway_sim[i]`  = max_j cos(a_i, b_j)              -- one-way at any theta
      `mnn_sim[i]`     = cos(a_i, b_j*) if mutual else -inf -- MNN at any theta
    Thresholding is then a comparison, so the whole theta grid is free.
    """
    C = Wa.transpose(0, 1) @ Wb                      # (na, nb)
    a_best_sim, a_best_j = C.max(dim=1)              # for each a_i, its best b
    b_best_sim, b_best_i = C.max(dim=0)              # for each b_j, its best a

    na = C.shape[0]
    idx_a = torch.arange(na, device=C.device)
    # a_i is mutual iff the b it chose chose it back.
    mutual_a = b_best_i[a_best_j] == idx_a

    nb = C.shape[1]
    idx_b = torch.arange(nb, device=C.device)
    mutual_b = a_best_j[b_best_i] == idx_b

    neg = torch.full((1,), float("-inf"), device=C.device, dtype=C.dtype)
    return {
        "a_oneway": a_best_sim,
        "a_mnn": torch.where(mutual_a, a_best_sim, neg.expand(na)),
        "b_oneway": b_best_sim,
        "b_mnn": torch.where(mutual_b, b_best_sim, neg.expand(nb)),
    }


def hungarian_subsample(
    Wa: torch.Tensor, Wb: torch.Tensor, m: int, rng: np.random.Generator
) -> dict:
    """Hungarian / one-way / MNN on a COMMON subsample, for the IoU comparison.

    Amendment X4: Hungarian on a dense 10,240 x 10,240 cost matrix is O(n^3)
    and impractical, so it runs on a subsample -- and THAT IS STATED IN THE
    OUTPUT rather than dropped silently. All three rules run on the SAME
    subsampled matrix so the IoU is a like-for-like comparison; a subsampled
    Hungarian is not a restriction of the full-matrix Hungarian and the output
    labels it as an approximation.
    """
    from scipy.optimize import linear_sum_assignment

    ia = np.sort(rng.choice(Wa.shape[1], size=m, replace=False))
    ib = np.sort(rng.choice(Wb.shape[1], size=m, replace=False))
    C = (Wa[:, ia].transpose(0, 1) @ Wb[:, ib]).float().cpu().numpy()

    row, col = linear_sum_assignment(-C)
    hung = {(int(r), int(c)): float(C[r, c]) for r, c in zip(row, col)}

    a_best_j = C.argmax(axis=1)
    a_best_sim = C.max(axis=1)
    b_best_i = C.argmax(axis=0)
    oneway = {(int(i), int(a_best_j[i])): float(a_best_sim[i]) for i in range(m)}
    mnn = {
        (int(i), int(a_best_j[i])): float(a_best_sim[i])
        for i in range(m)
        if b_best_i[a_best_j[i]] == i
    }
    return {"hungarian": hung, "oneway": oneway, "mnn": mnn, "m": m}


def iou(sa: dict, sb: dict, theta: float) -> float:
    """IoU between two rules' matched-pair sets at a threshold."""
    a = {k for k, v in sa.items() if v >= theta}
    b = {k for k, v in sb.items() if v >= theta}
    union = len(a | b)
    return float(len(a & b) / union) if union else float("nan")


# ---------------------------------------------------------------------------
# Rotation null (amendment X2 part 1)
# ---------------------------------------------------------------------------

def rotation_null(
    Wa: torch.Tensor, Wb: torch.Tensor, n_draws: int, seed: int
) -> dict:
    """Null distribution of the per-anchor MAX cosine under random rotation.

    Amendment X2: 'apply random orthonormal rotations in R^1280 to one seed's
    decoder and take the max cosine against another seed's decoder,
    accumulating the distribution of that maximum'. The unit of the
    distribution is therefore the per-anchor max, over anchors x draws.

    Rotating one decoder by a random orthogonal Q preserves its own internal
    geometry exactly while destroying any alignment with the other seed --
    which is the property that makes it the right null for 'would random
    geometry produce a match this good'.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = Wa.shape[0]
    per_draw_q: list[float] = []
    pool: list[np.ndarray] = []

    for _ in range(n_draws):
        A = torch.randn(d, d, generator=g, dtype=torch.float64)
        Q, R = torch.linalg.qr(A)
        # Sign-correct so Q is Haar-distributed rather than QR-biased.
        Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
        Qd = Q.to(device=Wa.device, dtype=Wa.dtype)
        rotated = Qd @ Wa
        rotated = rotated / rotated.norm(dim=0, keepdim=True).clamp_min(1e-12)
        maxcos = (rotated.transpose(0, 1) @ Wb).max(dim=1).values
        arr = maxcos.float().cpu().numpy()
        pool.append(arr)
        per_draw_q.append(float(np.percentile(arr, FLOOR_QUANTILE)))

    allv = np.concatenate(pool)
    floor = float(np.percentile(allv, FLOOR_QUANTILE))
    per_draw = np.asarray(per_draw_q, dtype=np.float64)
    # Monte Carlo error on the percentile: SE across independent draws
    # (amendment X8 -- a percentile this far into the tail is where draw count
    # actually binds, so the error is reported rather than assumed small).
    mc_se = float(per_draw.std(ddof=1) / np.sqrt(len(per_draw))) if len(per_draw) > 1 else float("nan")
    return {
        "n_draws": int(n_draws),
        "n_samples": int(allv.size),
        "quantile": FLOOR_QUANTILE,
        "floor": floor,
        "mc_se": mc_se,
        "per_draw_percentile_min": float(per_draw.min()),
        "per_draw_percentile_max": float(per_draw.max()),
        "null_max_cosine_mean": float(allv.mean()),
        "null_max_cosine_p50": float(np.percentile(allv, 50)),
        "null_max_cosine_p99": float(np.percentile(allv, 99)),
        "null_max_cosine_max": float(allv.max()),
    }


# ---------------------------------------------------------------------------
# theta selection (amendment X2) -- DERIVED, never asserted (gotcha 20)
# ---------------------------------------------------------------------------

def select_theta(
    endpoint_by_theta: dict[float, dict], floor: float
) -> dict:
    """Amendment X2's two-part rule, computed rather than stated.

    Part 1 -- theta must clear the rotation-null floor.
    Part 2 -- among those, take the SMALLEST theta such that neither endpoint
              bin (r_f = 0, r_f = 1) holds more than ENDPOINT_MAX_SHARE.
    """
    rows = []
    for theta in sorted(endpoint_by_theta):
        ep = endpoint_by_theta[theta]
        clears = bool(theta > floor)
        usable = bool(
            ep["frac_r_eq_0"] <= ENDPOINT_MAX_SHARE
            and ep["frac_r_eq_1"] <= ENDPOINT_MAX_SHARE
        )
        rows.append({
            "theta": theta,
            "clears_floor": clears,
            "frac_r_eq_0": ep["frac_r_eq_0"],
            "frac_r_eq_1": ep["frac_r_eq_1"],
            "usable": usable,
            "qualifies": bool(clears and usable),
        })
    qualifying = [r["theta"] for r in rows if r["qualifies"]]
    selected = min(qualifying) if qualifying else None
    return {
        "grid": rows,
        "floor": floor,
        "endpoint_max_share": ENDPOINT_MAX_SHARE,
        "selected_theta": selected,
        # Derived verdict. The failure branch (amendment X3) is a legitimate
        # outcome, not an error -- so the word is computed from the data.
        "verdict": "SELECTED" if selected is not None else "NO_THETA_QUALIFIES",
    }


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------

def per_seed_atoms(counts: np.ndarray, n_compare: int) -> dict:
    """Atom histogram computed SEPARATELY PER SEED, and the spread across them.

    WHY THIS EXISTS (amendment AB). The pooled histogram has 20 x 10,240
    anchors, and treating those as independent gives a binomial SE of
    ~0.05 pp per atom -- under which a 0.17 pp wiggle is a 3-sigma feature.
    But anchors are NOT independent: seed s's anchor matching seed t and
    seed t's anchor matching seed s are the same underlying event seen twice,
    and every anchor in a seed shares that seed's dictionary. So the naive SE
    is an unknown amount too small, and "is there an interior mode" cannot be
    answered with it.

    The across-seed SD of each atom is an EMPIRICAL error bar that carries the
    seed-level dependence, because each seed's histogram is one draw of the
    whole dictionary. It is the right reference for a per-atom claim, and it
    is gotcha 23's move -- compare a sample statistic against its own sampling
    distribution rather than against a textbook constant.
    """
    n_seeds = counts.shape[0]
    per = np.stack([
        np.bincount(counts[s].ravel(), minlength=n_compare + 1)
        / counts.shape[1]
        for s in range(n_seeds)
    ])                                            # (seeds, k)
    return {
        "per_seed_atom_fracs": per.tolist(),
        "atom_mean_over_seeds": per.mean(axis=0).tolist(),
        "atom_sd_over_seeds": per.std(axis=0, ddof=1).tolist(),
        "atom_se_over_seeds": (per.std(axis=0, ddof=1)
                               / np.sqrt(n_seeds)).tolist(),
        # The naive figure, reported ALONGSIDE so the gap is visible rather
        # than the smaller number being quoted by default.
        "atom_se_naive_pooled": (
            np.sqrt(per.mean(axis=0) * (1 - per.mean(axis=0))
                    / counts.size)).tolist(),
        "n_seeds": int(n_seeds),
    }


def marginals(counts: np.ndarray, n_compare: int) -> dict:
    """Summarise the r_f marginal. `counts` is (n_seeds, dict_size) int matches."""
    r = counts.astype(np.float64) / n_compare
    flat = r.ravel()
    hist = np.bincount(counts.ravel(), minlength=n_compare + 1).astype(int)
    return {
        "n_anchors": int(flat.size),
        "n_compare_seeds": int(n_compare),
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "iqr": [float(np.percentile(flat, 25)), float(np.percentile(flat, 75))],
        # Atoms at k/19 stated explicitly (amendment X / M1 output spec).
        "atom_counts_by_k": hist.tolist(),
        "atom_fracs_by_k": (hist / flat.size).tolist(),
        "frac_r_eq_0": float(hist[0] / flat.size),
        "frac_r_eq_1": float(hist[n_compare] / flat.size),
        "frac_k_le_1": float(hist[: min(2, n_compare + 1)].sum() / flat.size),
        "frac_k_ge_nminus1": float(hist[max(0, n_compare - 1):].sum() / flat.size),
        "frac_interior": float(
            1.0 - (hist[0] + hist[n_compare]) / flat.size
        ),
        # Mean pairwise rate = mean r_f, restated as the literature's units.
        "pairwise_match_rate": float(flat.mean()),
        # ---- AG1's PRE-DECLARED cuts, which are NOT `k <= 1` / `k >= N-1` at
        # N = 39. The legacy keys above keep their literal meanings (they are
        # named for exactly what they compute); these are the pre-registered
        # groups, and they are computed from the committed table so the run
        # cannot quietly apply a differently-scaled cut. ADDED at N = 19, where
        # they equal `frac_k_le_1` and `frac_k_ge_nminus1` by construction.
        "ag_endpoint_groups": _ag_group_fracs(hist, flat.size, n_compare),
    }


def _ag_group_fracs(hist: np.ndarray, n_total: int, n_compare: int) -> dict:
    """Observed mass in AG1's four pre-declared endpoint groups."""
    try:
        cuts = ag_cuts(n_compare)
    except KeyError as exc:
        return {"status": "NO_PREREGISTERED_CUTS", "reason": str(exc)}
    ku = cuts["primary_unstable_k_max"]
    ks = cuts["primary_stable_k_min"]
    return {
        "cuts": cuts,
        "primary_unstable_frac": float(hist[: ku + 1].sum() / n_total),
        "primary_stable_frac": float(hist[ks:].sum() / n_total),
        "sensitivity_unstable_frac": float(hist[0] / n_total),
        "sensitivity_stable_frac": float(hist[n_compare] / n_total),
    }


def endpoint_purity(marg: dict, n_compare: int) -> dict:
    """Y2's obligation: recompute P(k=0) vs P(k<=1) against the OBSERVED
    marginal, for the endpoint-cut reconsideration. Uses M1's marginal only,
    behind the X7 firewall -- no alignment quantity enters."""
    fr = np.asarray(marg["atom_fracs_by_k"], dtype=np.float64)
    # Implied mean p from the observed marginal.
    ks = np.arange(n_compare + 1)
    p_bar = float((fr * ks).sum() / n_compare)
    from math import comb
    p0 = (1 - p_bar) ** n_compare
    p1 = p0 + n_compare * p_bar * (1 - p_bar) ** (n_compare - 1)
    return {
        "implied_mean_p": p_bar,
        "binomial_P_k_eq_0_at_pbar": float(p0),
        "binomial_P_k_le_1_at_pbar": float(p1),
        "observed_frac_k_eq_0": marg["frac_r_eq_0"],
        "observed_frac_k_le_1": marg["frac_k_le_1"],
        "note": (
            "Y2's rationale used p = 0.02 as an illustration; these are the "
            "same quantities at the OBSERVED implied mean p. Cuts are not "
            "re-chosen here -- Y2 forbids re-choosing after the alignment join, "
            "and no alignment quantity is available to this job."
        ),
    }


# ---------------------------------------------------------------------------
# Per-anchor counts (amendment AR, route A)
# ---------------------------------------------------------------------------

def counts_at_theta(mnn_sim: np.ndarray, off: np.ndarray, n_anchors: int,
                    theta: float) -> np.ndarray:
    """Per-anchor MNN match counts at one theta. THE ONLY place this is
    computed, so the single-theta artifact and the all-theta artifact cannot
    disagree by construction.

    `off` is `anchor_off`: off-diagonal AND the row is an anchor. Slicing to
    `[:n_anchors]` afterwards is belt-and-braces -- a non-anchor row is
    already all-zero under that mask, and must never be mistaken for a
    feature with `r_f = 0`.
    """
    return ((mnn_sim >= theta) & off[:, :, None]
            ).sum(axis=1).astype(np.int16)[:n_anchors]


def compare_counts_to_reference(counts: np.ndarray, ref_path: Path,
                                theta: float, n_compare: int,
                                anchor_seeds: list[int]) -> dict:
    """Amendment AR's ACCEPTANCE CRITERION: bit-identical, not close.

    Declared before the edit that made it necessary. It is the same
    computation on the same decoders with the same pinned RNG, so anything
    other than equality is a defect in the refactor and not a tolerance
    question.

    THE COMPARISON IS ELEMENTWISE AND POSITIONAL. Shape, dtype and total all
    agree under a permutation of the rows or the features (gotcha 27), so
    none of them is the check -- they are reported alongside so a failure can
    be diagnosed, but `bit_identical` is `array_equal` on the raw buffers.

    The reference's IDENTITY is asserted first: an existence check is not an
    identity check (gotcha 15), and a reference built at a different
    `n_compare_seeds`, theta, similarity or anchor list is not a reference for
    this array at all. An ABSENT key is reported absent, never as agreement.
    """
    import h5py

    with h5py.File(ref_path, "r") as hf:
        if "match_count" not in hf:
            return {"verdict": "REFERENCE_MALFORMED",
                    "problem": f"{ref_path} has no `match_count` dataset"}
        ref = hf["match_count"][:]
        at = {k: (v.decode() if isinstance(v, bytes) else v)
              for k, v in hf.attrs.items()}

    problems = []
    identity = {}
    for key, want in (("theta", theta),
                      ("n_compare_seeds", n_compare),
                      ("similarity", "decoder_cosine"),
                      ("rule", "mnn"),
                      ("seed_labels", json.dumps(anchor_seeds))):
        if key not in at:
            identity[key] = "<ABSENT>"
            problems.append(f"{key}: ABSENT from {ref_path.name} "
                            f"(absent is not matching -- I3)")
            continue
        got = at[key]
        identity[key] = got.item() if hasattr(got, "item") else got
        if isinstance(want, float):
            same = abs(float(got) - want) < 1e-12
        elif isinstance(want, int):
            same = int(got) == want
        else:
            same = str(got) == str(want)
        if not same:
            problems.append(f"{key}: reference says {got!r}, this run has "
                            f"{want!r}")

    same_shape = tuple(ref.shape) == tuple(counts.shape)
    same_dtype = ref.dtype == counts.dtype
    # `array_equal` on integer arrays of the same dtype IS bit identity; the
    # byte comparison is kept beside it so the claim is not an inference.
    bit_identical = bool(
        same_shape and same_dtype
        and ref.tobytes() == np.ascontiguousarray(counts).tobytes())
    n_diff = (int((ref != counts).sum()) if same_shape else None)

    return {
        "reference": str(ref_path),
        "reference_identity": identity,
        "identity_problems": problems,
        "theta_compared": theta,
        "same_shape": same_shape,
        "same_dtype": bool(same_dtype),
        "n_cells_differing": n_diff,
        "total_reference": (int(ref.sum()) if same_shape else None),
        "total_this_run": int(counts.sum()),
        "bit_identical": bit_identical,
        # DERIVED, never an asserted sentence (gotcha 20).
        "verdict": ("BIT_IDENTICAL" if (bit_identical and not problems)
                    else "IDENTITY_MISMATCH" if problems
                    else "DIFFERS"),
    }


def verify_compare_denominator(n_compare: int, n_anchor_ckpts: int,
                               n_compare_ckpts: int) -> dict:
    """The comparison denominator, CHECKED against the pool it was formed from
    (amendment CQ, A3).

    `n_compare = len(decoders) - 1` is right because `ckpts = anchor_ckpts +
    compare_ckpts` and one decoder is loaded per checkpoint, so the pool is
    the whole ensemble and an anchor compares against everyone but itself
    (`n_compare = 39` at S = 40, AM7). That identity is asserted here rather
    than relied on: a skipped load, a filtered list or a second pool would
    otherwise write a plausible denominator. Prints the composition; raises
    `SystemExit(1)` naming both values on disagreement.
    """
    expected = n_anchor_ckpts + n_compare_ckpts - 1
    rep = {"n_compare": int(n_compare), "n_anchor_ckpts": int(n_anchor_ckpts),
           "n_compare_only_ckpts": int(n_compare_ckpts),
           "pool": int(n_anchor_ckpts + n_compare_ckpts),
           "expected_n_compare": int(expected),
           "agree": bool(n_compare == expected)}
    if n_compare != expected:
        raise SystemExit(
            f"PRE-FLIGHT FAIL: n_compare = len(decoders) - 1 = {n_compare}, "
            f"but the pool is {n_anchor_ckpts} anchors + {n_compare_ckpts} "
            f"comparison-only = {n_anchor_ckpts + n_compare_ckpts} "
            f"checkpoints, so n_compare must be {expected}. The denominator "
            f"of every r_f would be wrong by a plausible amount. Refusing.")
    print(f"[rf] n_compare = {n_compare} = ({n_anchor_ckpts} anchors + "
          f"{n_compare_ckpts} comparison-only) - 1  [checked]", flush=True)
    return rep


def assert_written_denominator(where: str, written, n_compare: int) -> None:
    """Read a written `n_compare_seeds` back and refuse unless it equals the
    denominator the counts were formed with (amendment CQ, A3)."""
    if int(written) != int(n_compare):
        raise SystemExit(
            f"FATAL: {where} records n_compare_seeds = {int(written)} but the "
            f"run's denominator is {int(n_compare)}. Refusing to leave an "
            f"artifact whose stated denominator is not the one used.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_decoder(ckpt: Path, device: str) -> tuple[torch.Tensor, dict]:
    meta = json.loads((ckpt / "meta.json").read_text())
    sd = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    W = sd["W_dec"].to(device=device, dtype=torch.float32)
    return W, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", type=Path, required=True,
                    help="ANCHOR checkpoints -- the seeds r_f is measured FOR. "
                         "AB4 fixes these at 0-19 (the seeds with AP).")
    ap.add_argument("--compare-ckpts", nargs="*", type=Path, default=[],
                    help="COMPARISON-ONLY checkpoints (AD5's "
                         "sae_ckpts_ext/seed_20..39). Dictionaries r_f is "
                         "measured AGAINST but never measured FOR. Empty "
                         "reproduces the N = 19 path exactly.")
    ap.add_argument("--expect-total-ckpts", type=int, default=None,
                    help="Refuse unless EXACTLY this many checkpoints were "
                         "loaded in total. An identity check, not an existence "
                         "check (gotcha 15): a self-consistency check on the "
                         "count passes at 20 and would let an N = 19 result be "
                         "labelled N = 39.")
    ap.add_argument("--preflight-only", action="store_true",
                    help="Load every checkpoint, run the identity / unit-norm "
                         "/ b_dec-distinctness checks, write the report and "
                         "exit. Computes no marginal.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rotation-draws", type=int, default=50)
    ap.add_argument("--hungarian-sizes", type=int, nargs="+",
                    default=[1024, 2048, 4096],
                    help="subsample sizes for the Hungarian/IoU scale curve "
                         "(amendment AB section 6)")
    ap.add_argument("--hungarian-repeats", type=int, default=5)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--emit-counts", type=Path, default=None,
                    help="write per-anchor MNN match counts at the reference "
                         "theta to this h5. Storing r_f is not a firewall "
                         "breach -- X7 forbids the JOIN, and this job performs "
                         "none. The join happens in m4_confound.py, which is "
                         "labelled as a deliberate crossing.")
    ap.add_argument("--emit-counts-all-theta", type=Path, default=None,
                    help="amendment AR route A: write per-anchor MNN match "
                         "counts at EVERY theta on the grid, one dataset per "
                         "grid point, to this h5. Does not touch "
                         "--emit-counts' output. Storing r_f is not a firewall "
                         "breach -- X7 forbids the JOIN.")
    ap.add_argument("--counts-reference", type=Path, default=None,
                    help="amendment AR's acceptance criterion: the reference-"
                         "theta slice of --emit-counts-all-theta must be "
                         "BIT-IDENTICAL to `match_count` in this existing "
                         "single-theta artifact. The job EXITS NON-ZERO if it "
                         "is not.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--accept-git-shas", nargs="+", default=None,
                    help="AMENDMENT CS, CS3c. The EXACT set of training git_sha "
                         "values the pool may carry (full 40-hex); a git_sha "
                         "disagreement equal to this set is recorded, not "
                         "refused. Superset/subset refused. See "
                         "verify_ensemble_identity.")
    args = ap.parse_args()

    install_firewall()

    from src.sae.train import git_dirty, git_sha  # provenance, same as elsewhere

    def _by_seed_num(p: Path) -> int:
        return int(str(p.name).rsplit("_", 1)[-1])

    anchor_ckpts = sorted(args.ckpts, key=_by_seed_num)
    compare_ckpts = sorted(args.compare_ckpts, key=_by_seed_num)
    # Anchors FIRST, so `decoders[:n_anchors]` are the anchors and the N = 19
    # path (no --compare-ckpts) indexes exactly as it did before.
    ckpts = anchor_ckpts + compare_ckpts
    n_anchors = len(anchor_ckpts)
    print(f"[rf] {len(ckpts)} checkpoints ({n_anchors} anchors, "
          f"{len(compare_ckpts)} comparison-only), device={args.device}",
          flush=True)

    # THE COUNT GUARD (AK2 A1 / AD5). Asserted on the TOTAL, before any load,
    # because the failure it prevents is a correct-looking N = 19 result wearing
    # an N = 39 label.
    if args.expect_total_ckpts is not None and len(ckpts) != args.expect_total_ckpts:
        print(f"PRE-FLIGHT FAIL: --expect-total-ckpts "
              f"{args.expect_total_ckpts} but {len(ckpts)} were supplied "
              f"({n_anchors} anchors + {len(compare_ckpts)} comparison-only). "
              f"Refusing: a count mismatch here silently changes which "
              f"ensemble every downstream number describes.", flush=True)
        raise SystemExit(1)

    decoders: list[torch.Tensor] = []
    meta_by_seed: dict[int, dict] = {}
    ckpt_hashes: dict[int, str] = {}
    unit_norm: dict[int, dict] = {}
    anchor_seeds: list[int] = []
    for i, p in enumerate(ckpts):
        W, meta = load_decoder(p, args.device)
        s = int(meta["seed"])
        if s in meta_by_seed:
            print(f"PRE-FLIGHT FAIL: seed {s} supplied twice "
                  f"({p}) -- a duplicated dictionary would be counted as an "
                  f"independent comparison seed.", flush=True)
            raise SystemExit(1)
        meta_by_seed[s] = meta
        ckpt_hashes[s] = sha256_16(p / "model.pt")
        unit_norm[s] = check_unit_norm(W)
        if i < n_anchors:
            anchor_seeds.append(s)
        decoders.append(W)

    identity = verify_ensemble_identity(meta_by_seed, args.accept_git_shas)
    print(json.dumps({"identity": identity}, indent=2)[:4000], flush=True)
    bad_norm = {s: v for s, v in unit_norm.items() if v["verdict"] != "PASS"}
    b_dec_distinct = verify_b_dec_distinctness(meta_by_seed)
    print(f"[rf] b_dec distinctness: {b_dec_distinct['verdict']} "
          f"({b_dec_distinct.get('n_distinct_final_b_dec')} distinct of "
          f"{b_dec_distinct.get('n_seeds')})", flush=True)

    # HARD FAIL, and the failure is readable to whatever gates on this job
    # (gotcha 13 extension 1). Do this BEFORE any expensive compute.
    if (identity["verdict"] != "PASS" or bad_norm
            or b_dec_distinct["verdict"] != "PASS"):
        print("PRE-FLIGHT FAIL", json.dumps(
            {"identity_problems": identity["problems"], "unit_norm": bad_norm,
             "b_dec_distinctness": b_dec_distinct},
            indent=2), flush=True)
        raise SystemExit(1)
    print("[rf] pre-flight PASS", flush=True)

    n_seeds = len(decoders)
    dict_size = decoders[0].shape[1]
    d_model = decoders[0].shape[0]
    n_compare = n_seeds - 1
    # amendment CQ, A3: the arithmetic does not move; it becomes checked.
    denominator_check = verify_compare_denominator(
        n_compare, len(anchor_ckpts), len(compare_ckpts))
    # amendment CQ, A5: the device the cosine matmul runs on, read off the
    # tensor rather than off the request (gotcha 36).
    compute_device_class = device_class(decoders[0])

    if args.preflight_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "identity": {
                "measurement": "rf_marginal_PREFLIGHT_ONLY",
                "n_seeds": n_seeds, "n_anchors": n_anchors,
                "n_compare_seeds": n_compare,
                "anchor_seed_labels": anchor_seeds,
                "compare_only_seed_labels": [
                    s for s in sorted(meta_by_seed) if s not in anchor_seeds],
                "dict_size": dict_size, "d_model": d_model,
                "ckpt_model_sha256_16": ckpt_hashes,
                "git_sha": git_sha(), "git_dirty": git_dirty(),
            },
            "preflight": {"ensemble_identity": identity,
                          "unit_norm": unit_norm,
                          "b_dec_distinctness": b_dec_distinct},
        }, indent=2))
        print(f"[rf] PREFLIGHT ONLY -- wrote {args.out}; no marginal computed",
              flush=True)
        return

    # ---- rotation null (one representative pair; X2 part 1) ----
    null = rotation_null(decoders[0], decoders[1], args.rotation_draws, args.rng_seed)
    print(f"[rf] rotation-null {FLOOR_QUANTILE}th pct = {null['floor']:.4f} "
          f"(MC SE {null['mc_se']:.5f})", flush=True)

    # ---- every unordered pair WITH AT LEAST ONE ANCHOR ----
    # An extension seed is never an anchor, so its row is never read and the
    # extension-extension pairs are pure waste. Skipping them is the difference
    # between 590 and 780 pairs at S = 40; with no --compare-ckpts nothing is
    # skipped and the loop is the original one.
    oneway_sim = np.full((n_seeds, n_seeds, dict_size), -np.inf, dtype=np.float32)
    mnn_sim = np.full((n_seeds, n_seeds, dict_size), -np.inf, dtype=np.float32)
    n_pairs = 0
    n_pairs_skipped = 0
    for a in range(n_seeds):
        for b in range(a + 1, n_seeds):
            if a >= n_anchors and b >= n_anchors:
                n_pairs_skipped += 1
                continue
            st = pair_statistics(decoders[a], decoders[b])
            oneway_sim[a, b] = st["a_oneway"].float().cpu().numpy()
            mnn_sim[a, b] = st["a_mnn"].float().cpu().numpy()
            oneway_sim[b, a] = st["b_oneway"].float().cpu().numpy()
            mnn_sim[b, a] = st["b_mnn"].float().cpu().numpy()
            n_pairs += 1
        print(f"[rf] pairs done through seed {a}", flush=True)
    print(f"[rf] {n_pairs} unordered pairs computed "
          f"({n_pairs_skipped} extension-extension pairs skipped)", flush=True)

    # `anchor_off`: off-diagonal AND the row is an anchor. Every reduction below
    # uses it, so a non-anchor's deliberately incomplete row cannot enter a
    # statistic. With no --compare-ckpts every row is an anchor and this is
    # elementwise equal to the old `off`, which is what keeps N = 19
    # bit-identical.
    off = ~np.eye(n_seeds, dtype=bool)
    is_anchor_row = np.zeros((n_seeds, 1), dtype=bool)
    is_anchor_row[:n_anchors] = True
    anchor_off = off & is_anchor_row
    off = anchor_off

    # ---- P1: MNN nested inside one-way. A violation is a BUG, not a result.
    viol = int((mnn_sim[off] > oneway_sim[off] + 1e-6).sum())
    nested_ok = viol == 0
    if not nested_ok:
        print(f"P1 CORRECTNESS ASSERTION FAILED: {viol} anchors have "
              f"mnn_sim > oneway_sim -- MNN is not nested inside one-way, "
              f"which amendment W1 proves it must be. This is an "
              f"implementation bug.", flush=True)
        raise SystemExit(1)
    print("[rf] P1 nesting assertion PASS", flush=True)

    results: dict = {}
    endpoint_by_theta: dict[float, dict] = {}
    for theta in THETA_GRID:
        per_rule = {}
        for rule, sim in (("mnn", mnn_sim), ("oneway", oneway_sim)):
            hit = (sim >= theta) & off[:, :, None]
            # Anchor rows only: a non-anchor row is all-zero by construction
            # (its pairs were never computed) and must not be mistaken for a
            # feature with r_f = 0.
            counts = hit.sum(axis=1).astype(np.int16)[:n_anchors]
            m = marginals(counts, n_compare)
            m["endpoint_purity"] = endpoint_purity(m, n_compare)
            m["per_seed"] = per_seed_atoms(counts, n_compare)
            per_rule[rule] = m
        results[f"{theta:.2f}"] = per_rule
        endpoint_by_theta[theta] = per_rule["mnn"]

    selection = select_theta(endpoint_by_theta, null["floor"])
    print(f"[rf] theta selection: {selection['verdict']} "
          f"-> {selection['selected_theta']}", flush=True)

    # ---- Hungarian arm on a common subsample, at SEVERAL subsample sizes ----
    # X4 established the like-for-like defence (all three rules on the SAME
    # subsampled matrix). Amendment AB adds the SCALE curve, because that
    # defence does not establish the IoU is scale-INVARIANT: with a smaller
    # candidate pool there is less opportunity for the competing-pair
    # configuration that makes Hungarian diverge from MNN, so a single n may
    # understate the full-scale divergence. Subsampling also perturbs the
    # other two rules -- it can only lower one-way's max cosine, and it can
    # MANUFACTURE mutuality by removing a feature's true best partner -- so
    # all three move and only their agreement is comparable.
    rng = np.random.default_rng(args.rng_seed)
    hung_rows = []
    for m_sub in args.hungarian_sizes:
        for rep in range(args.hungarian_repeats):
            sub = hungarian_subsample(decoders[0], decoders[1], m_sub, rng)
            for theta in THETA_GRID:
                i_mo = iou(sub["mnn"], sub["oneway"], theta)
                i_ho = iou(sub["hungarian"], sub["oneway"], theta)
                i_mh = iou(sub["mnn"], sub["hungarian"], theta)
                # ---- C3: an ANALYTIC identity, asserted rather than left as
                # an empty cell. If MNN and one-way select the identical set
                # (IoU = 1), then IoU(MNN, Hungarian) MUST equal
                # IoU(one-way, Hungarian). Gotcha 20's practice applied to a
                # table cell: compute what the value must be if the
                # dependence holds, and confirm the code produces it.
                if np.isfinite(i_mo) and abs(i_mo - 1.0) < 1e-12:
                    if not (np.isnan(i_mh) and np.isnan(i_ho)):
                        assert abs(i_mh - i_ho) < 1e-9, (
                            f"C3 identity violated at theta={theta}, m={m_sub}: "
                            f"IoU(MNN,1w)=1 implies IoU(MNN,H)==IoU(1w,H), "
                            f"got {i_mh} vs {i_ho}")
                hung_rows.append({
                    "repeat": rep,
                    "theta": theta,
                    "m": sub["m"],
                    "iou_hungarian_vs_oneway": i_ho,
                    "iou_mnn_vs_oneway": i_mo,
                    "iou_mnn_vs_hungarian": i_mh,
                    "n_hungarian": sum(1 for v in sub["hungarian"].values() if v >= theta),
                    "n_oneway": sum(1 for v in sub["oneway"].values() if v >= theta),
                    "n_mnn": sum(1 for v in sub["mnn"].values() if v >= theta),
                })

    # ---- C3 second half: is IoU(MNN, one-way) = 1 at high theta a REAL
    # result, or the MNN branch silently degenerating to one-way? The
    # evidence that it is real must be POSITIVE, not the absence of a
    # complaint (gotcha 13). Two derived checks on the FULL matrices:
    #   (a) the rules must DIFFER somewhere -- at some theta the MNN and
    #       one-way match sets are not identical;
    #   (b) at theta = 0.70 the pooled counts may agree only to the extent
    #       the full-matrix rates do, which is 0.4958 vs 0.4971, i.e. NOT
    #       exactly -- so an exact full-matrix tie would itself be the bug.
    n_mnn_full = int((mnn_sim[off] >= THETA_LITERATURE).sum())
    n_ow_full = int((oneway_sim[off] >= THETA_LITERATURE).sum())
    rules_differ_somewhere = bool(
        any(results[f"{t:.2f}"]["mnn"]["mean"]
            != results[f"{t:.2f}"]["oneway"]["mean"] for t in THETA_GRID))
    mnn_degeneracy_check = {
        "n_mnn_at_0.70_full_matrix": n_mnn_full,
        "n_oneway_at_0.70_full_matrix": n_ow_full,
        "full_matrix_sets_are_identical": n_mnn_full == n_ow_full,
        "rules_differ_at_some_theta": rules_differ_somewhere,
        "verdict": "MNN_IS_A_DISTINCT_RULE" if (
            rules_differ_somewhere and n_mnn_full != n_ow_full)
            else "SUSPECT_MNN_MAY_BE_ONEWAY",
    }

    # ---- AK3 B5: the MNN-to-one-way rate ratio across the whole theta grid.
    # The brief's "theta curve shape" prediction has two halves and only this
    # one is new: the SUBSAMPLED IoU is computed from the seed-0/seed-1 decoders
    # with a pinned RNG, so it is entailed by A2's bit-identity and carries no
    # information about the extension. This ratio is over the anchor rows and
    # therefore does move with N.
    ratio_rows = []
    for theta in THETA_GRID:
        m_rate = results[f"{theta:.2f}"]["mnn"]["pairwise_match_rate"]
        o_rate = results[f"{theta:.2f}"]["oneway"]["pairwise_match_rate"]
        ratio_rows.append({
            "theta": theta,
            "mnn_rate": m_rate,
            "oneway_rate": o_rate,
            "ratio": (m_rate / o_rate) if o_rate > 0 else float("nan"),
        })
    ratios = [r["ratio"] for r in ratio_rows]
    mnn_oneway_ratio_curve = {
        "rows": ratio_rows,
        "ratio_at_reference_theta": ratio_rows[
            [i for i, r in enumerate(ratio_rows)
             if abs(r["theta"] - THETA_LITERATURE) < 1e-9][0]]["ratio"],
        # DERIVED, never an asserted sentence (gotcha 20).
        "is_non_decreasing_in_theta": bool(
            all(ratios[i + 1] >= ratios[i] - 1e-12
                for i in range(len(ratios) - 1))),
        "first_decrease_at_theta": next(
            (ratio_rows[i + 1]["theta"] for i in range(len(ratios) - 1)
             if ratios[i + 1] < ratios[i] - 1e-12), None),
    }
    print(f"[rf] MNN/one-way ratio at {THETA_LITERATURE}: "
          f"{mnn_oneway_ratio_curve['ratio_at_reference_theta']:.6f}; "
          f"non-decreasing in theta: "
          f"{mnn_oneway_ratio_curve['is_non_decreasing_in_theta']}", flush=True)
    print(f"[rf] MNN-distinctness check: {mnn_degeneracy_check['verdict']} "
          f"({n_mnn_full} vs {n_ow_full} at 0.70)", flush=True)

    # ---- score the committed predictions, DERIVED not asserted (gotcha 20) --
    lit = f"{THETA_LITERATURE:.2f}"
    mnn_rate = results[lit]["mnn"]["pairwise_match_rate"]
    ow_rate = results[lit]["oneway"]["pairwise_match_rate"]
    h07 = [r for r in hung_rows if abs(r["theta"] - THETA_LITERATURE) < 1e-9]
    iou_h = float(np.mean([r["iou_hungarian_vs_oneway"] for r in h07]))
    iou_m = float(np.mean([r["iou_mnn_vs_oneway"] for r in h07]))
    scorecard = {
        "P1_mnn_nested_in_oneway": {
            "kind": "correctness assertion",
            "n_violations": viol,
            "outcome": "PASS" if nested_ok else "FAIL",
        },
        "P2_mnn_rate_at_0.70_in_[0.005,0.12]": {
            "measured": mnn_rate,
            "oneway_rate_same_theta": ow_rate,
            "borobia_band": [0.011, 0.039],
            "outcome": "HIT" if 0.005 <= mnn_rate <= 0.12 else "MISS",
        },
        "P3_rotation_floor_below_0.30": {
            "measured": null["floor"],
            "mc_se": null["mc_se"],
            "outcome": "HIT" if null["floor"] < 0.30 else "MISS",
        },
        "P4_no_theta_satisfies_usability": {
            "selected_theta": selection["selected_theta"],
            "outcome": "HIT" if selection["selected_theta"] is None else "MISS",
        },
        "P5_iou_hungarian_gt_iou_mnn_vs_oneway": {
            "iou_hungarian_vs_oneway": iou_h,
            "iou_mnn_vs_oneway": iou_m,
            "outcome": "HIT" if iou_h > iou_m else "MISS",
        },
    }
    print(json.dumps({"scorecard": scorecard}, indent=2), flush=True)

    out = {
        # ---- identity attrs (amendment X10 / I3) ----
        "identity": {
            "measurement": "rf_marginal_M1",
            "amendment": "X",
            "similarity": "decoder_cosine",   # X6: NOT min(corr, cos)
            "device_class": compute_device_class,           # CQ A5
            "denominator_check": denominator_check,         # CQ A3
            "rule_set": ["mnn", "oneway", "hungarian_subsampled"],
            "theta_grid": THETA_GRID,
            "floor_quantile": FLOOR_QUANTILE,
            "endpoint_max_share": ENDPOINT_MAX_SHARE,
            "seed_labels": sorted(meta_by_seed),
            "ckpt_model_sha256_16": ckpt_hashes,
            "n_seeds": n_seeds,
            "dict_size": dict_size,
            # ---- the anchor / comparison split (AB4, AD5, AK). ADDED KEYS at
            # N = 19; every pre-existing key's value is unchanged there, which
            # is the acceptance criterion AK1 states.
            "n_anchors": n_anchors,
            "n_compare_seeds": n_compare,
            "anchor_seed_labels": anchor_seeds,
            "compare_only_seed_labels": [s for s in sorted(meta_by_seed)
                                         if s not in anchor_seeds],
            "n_pairs_computed": n_pairs,
            "n_pairs_skipped_extension_extension": n_pairs_skipped,
            "d_model": d_model,
            "rotation_null_draws": args.rotation_draws,
            "hungarian_sizes": list(args.hungarian_sizes),
            "hungarian_repeats": args.hungarian_repeats,
            "rng_seed": args.rng_seed,
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "preflight": {"ensemble_identity": identity, "unit_norm": unit_norm,
                      "b_dec_distinctness": b_dec_distinct},
        "rotation_null": null,
        "theta_selection": selection,
        "marginals_by_theta": results,
        "mnn_distinctness_check": mnn_degeneracy_check,
        "mnn_oneway_ratio_curve": mnn_oneway_ratio_curve,
        "hungarian_arm": {
            "note": (
                "SUBSAMPLED, and labelled as such per amendment X4. A "
                "subsampled Hungarian is not a restriction of the full-matrix "
                "Hungarian; all three rules run on the SAME subsampled matrix "
                "so the IoU is like-for-like."
            ),
            "rows": hung_rows,
        },
        "prediction_scorecard": scorecard,
    }
    # amendment CQ, A3: the M1 identity's denominator is checked here, before
    # the JSON is written; each h5 copy is read back after it is set (`_stamp`,
    # the per-theta datasets). Not checked: the copies in the --preflight-only
    # output and in each `marginals()` block.
    assert_written_denominator("M1 identity.n_compare_seeds",
                               out["identity"]["n_compare_seeds"], n_compare)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[rf] wrote {args.out}", flush=True)

    def _stamp(hf) -> None:
        """The identity attrs both counts artifacts carry, written once."""
        hf.attrs["similarity"] = "decoder_cosine"
        hf.attrs["rule"] = "mnn"
        hf.attrs["n_compare_seeds"] = n_compare
        assert_written_denominator(f"{hf.filename} attrs",
                                   hf.attrs["n_compare_seeds"], n_compare)
        hf.attrs["device_class"] = compute_device_class          # CQ A5
        # `match_count` rows are ANCHORS, so `seed_labels` is the anchor
        # list -- which is what it always was, since every seed was an
        # anchor at N = 19. Consumers index rows by this.
        hf.attrs["seed_labels"] = json.dumps(anchor_seeds)
        hf.attrs["n_anchors"] = n_anchors
        hf.attrs["compare_only_seed_labels"] = json.dumps(
            [s for s in sorted(meta_by_seed) if s not in anchor_seeds])
        hf.attrs["dict_size"] = dict_size
        hf.attrs["git_sha"] = git_sha()
        hf.attrs["git_dirty"] = git_dirty()

    if args.emit_counts is not None:
        import h5py
        lit_counts = counts_at_theta(mnn_sim, off, n_anchors, THETA_LITERATURE)
        args.emit_counts.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.emit_counts, "w") as hf:
            hf.create_dataset("match_count", data=lit_counts,
                              compression="gzip", compression_opts=1)
            hf.attrs["measurement"] = "rf_per_anchor_counts"
            hf.attrs["theta"] = THETA_LITERATURE
            _stamp(hf)
        print(f"[rf] wrote per-anchor counts to {args.emit_counts} "
              f"(theta={THETA_LITERATURE})", flush=True)

    # ---- amendment AR route A: counts across the WHOLE theta grid ----------
    if args.emit_counts_all_theta is not None:
        import h5py
        by_theta = {t: counts_at_theta(mnn_sim, off, n_anchors, t)
                    for t in THETA_GRID}
        args.emit_counts_all_theta.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.emit_counts_all_theta, "w") as hf:
            for t, c in by_theta.items():
                d = hf.create_dataset(f"match_count_theta_{t:.2f}", data=c,
                                      compression="gzip", compression_opts=1)
                # Per-dataset attrs, so a slice carries its own theta and
                # denominator and cannot be read under the wrong one.
                d.attrs["theta"] = t
                d.attrs["n_compare_seeds"] = n_compare
                assert_written_denominator(f"{d.name} attrs",
                                           d.attrs["n_compare_seeds"], n_compare)
                d.attrs["similarity"] = "decoder_cosine"
                d.attrs["rule"] = "mnn"
            hf.attrs["measurement"] = "rf_per_anchor_counts_all_theta"
            hf.attrs["amendment"] = "AR"
            hf.attrs["theta_grid"] = json.dumps(THETA_GRID)
            hf.attrs["reference_theta"] = THETA_LITERATURE
            hf.attrs["dataset_name_template"] = "match_count_theta_{theta:.2f}"
            _stamp(hf)
        print(f"[rf] wrote per-anchor counts at {len(THETA_GRID)} thetas to "
              f"{args.emit_counts_all_theta}", flush=True)

        if args.counts_reference is not None:
            chk = compare_counts_to_reference(
                by_theta[THETA_LITERATURE], args.counts_reference,
                THETA_LITERATURE, n_compare, anchor_seeds)
            print("[rf] acceptance criterion (amendment AR):\n"
                  + json.dumps(chk, indent=2), flush=True)
            if chk["verdict"] != "BIT_IDENTICAL":
                # The verdict must be readable by whatever gates on this job
                # (gotcha 13 extension 1). A printed FAIL with exit 0 would be
                # worse than no check at all.
                print("ACCEPTANCE CRITERION FAILED -- the reference-theta "
                      "slice is not bit-identical to the existing "
                      "single-theta artifact. Refusing to report the "
                      "all-theta counts as usable.", flush=True)
                raise SystemExit(1)
            print("[rf] acceptance criterion PASS -- reference slice is "
                  "bit-identical", flush=True)
    elif args.counts_reference is not None:
        print("--counts-reference was given without "
              "--emit-counts-all-theta: there is nothing to compare, and a "
              "check that cannot run must not report success (gotcha 28).",
              flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
