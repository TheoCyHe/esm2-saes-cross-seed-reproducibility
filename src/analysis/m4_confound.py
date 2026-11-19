"""
M4 SECTION 2 -- THE CONFOUND CHECK. Amendment AC section 2.

*** THIS JOB DELIBERATELY CROSSES THE X7 FIREWALL. ***

X7 keeps `r_f` and alignment apart so that `theta` cannot be selected on the
relationship between them. This module joins them on purpose, because the
question it answers is a SPECIFICATION question and cannot be asked any other
way. The crossing is:

  * SCOPED -- it computes two quantities and nothing else;
  * LABELLED -- the output records `firewall_crossing: DELIBERATE` and the
    reason, so a later reader cannot mistake it for an ordinary job;
  * NON-AMBIENT -- it is its own script and its own SLURM job. Nothing about
    `theta`, the endpoint cuts, or the primary regression may be revised on
    the basis of what it shows, and the output says so.

THE PROBLEM. U3 makes `mu_s` a smooth function of FEATURE-SIDE covariates
(perplexity, concentration). The stable/unstable literature associates
reproducibility with activation frequency and decoder geometry, so feature-side
covariates plausibly predict `r_f`. Dividing the predictor by a quantity that
depends on feature properties **which themselves predict the outcome partially
regresses out the relationship being estimated.** GSEA's normalizer depends
only on the gene set; ours would depend on the feature too.

THE TWO QUANTITIES, both cheap:
  1. `corr(mu_hat, r_f)` directly.
  2. The coefficient on `log M` with feature-side covariates IN versus OUT of
     the `mu_s` model.

If (2) moves materially, the concept-side and feature-side halves of the trend
must be separated -- normalize on concept-side covariates only, and let
feature-side properties enter the regression as covariates where they belong.
**This module REPORTS. The decision to split is the author's** (amendment AC section
6), and no recommendation is emitted.

ON THE REGRESSION USED HERE. The primary regression's specification is
reserved for the author (within-seed dependence, dispersion, spline basis). This
module therefore uses a deliberately plain per-seed OLS of `r_f` on
standardised `log M` plus `log10` firing rate, and reports the coefficient on
`log M` under two `mu` models. **That is legitimate for a COMPARISON** -- the
same estimator is applied to both predictors, so the difference between the
two coefficients is what is being measured, not either coefficient's absolute
value. It is not the primary regression and is not a preview of one.

=============================================================================
PREDICTIONS, committed to git before the run
=============================================================================
  C1. `|Spearman(mu_hat, r_f)|` exceeds 0.05, i.e. the confound is present
      rather than nil. CATEGORY: within-system comparison.

  C2. The coefficient on `log M` changes by MORE than 10% in relative terms
      when feature-side covariates are dropped from the `mu_s` model.
      CATEGORY: within-system comparison. This is the quantity the decision
      turns on, so the threshold is stated before the number is seen.

  C3. Dropping the feature-side covariates makes the `log M` coefficient
      LARGER in magnitude -- i.e. the feature-side normalisation was
      absorbing part of the signal.
      CATEGORY: **DIRECTIONAL MECHANISM** (0/7 record). Filed and labelled
      per amendment AB17.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

from src.analysis.artifact_identity import argmax_pool_mask
from src.analysis.smooth_mu import (
    CONCEPT_SIDE,
    COVARIATES,
    _blocks,
    _order,
    assemble_covariates,
    fit_smooth_mu,
    predict_on_axis,
    resolve_row_index,
)
from src.analysis.artifact_identity import read_identity
from src.analysis.null_pooling import bin_index
from src.analysis.standardize_gain import masked_argmax


def per_seed_logM(real_gain: np.ndarray, mu_hat: np.ndarray,
                  pool_mask: np.ndarray) -> np.ndarray:
    """`log M_f` under a given `mu_hat`, exactly as the predictor defines it:
    ratio-normalise, then take the max over the argmax pool."""
    w = real_gain / mu_hat
    M, _ = masked_argmax(w, pool_mask)
    return np.log(np.maximum(M, 1e-300))


def ols_logM_coef(r_f: np.ndarray, logM: np.ndarray,
                  log_fire: np.ndarray) -> dict:
    """Plain OLS of r_f on standardised log M + log10 firing rate.

    Standardising log M makes the coefficient comparable across the two mu
    models even though their log M values live on different scales -- which is
    the whole point of the comparison and would otherwise be confounded by a
    units change.
    """
    ok = np.isfinite(logM) & np.isfinite(log_fire) & np.isfinite(r_f)
    x1 = logM[ok]
    x1 = (x1 - x1.mean()) / max(x1.std(), 1e-12)
    x2 = log_fire[ok]
    x2 = (x2 - x2.mean()) / max(x2.std(), 1e-12)
    X = np.column_stack([np.ones(x1.size), x1, x2])
    y = r_f[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(x1.size - 3, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return {"coef_logM": float(beta[1]),
            "se_logM": float(np.sqrt(cov[1, 1])),
            "coef_log_fire": float(beta[2]),
            "n": int(x1.size)}


def hard_bin_mu_per_feature(gtilde_path: Path,
                            fire_rate: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-feature HARD-BIN `mu_s`, the normaliser of the predictor of record.

    `stratum_mu` is indexed by (seed, concept cell, firing bin). `concept_cell`
    supplies the concept side; `feature_firing_bin` supplies the feature side --
    and the feature side was not stored until this session, which is why AC12
    reported C1 for hard-bin as not computable.

    THAT BLOCKER WAS WRONG, and this function is the demonstration: the
    assignment is a pure function of `fire_rate` (in the null-reductions
    artifact) and the `firing_rate_edges` attr (already written). Where
    `feature_firing_bin` is present it is used; where absent it is
    RECONSTRUCTED, and `tests/test_standardize_gain.py` proves the two agree
    exactly rather than approximately.

    Reduces over the concept axis by MEDIAN, matching how the smooth arm's C1
    reduces `mu_hat_full`, so the two C1 numbers are computed the same way and
    are comparable.
    """
    import json as _json

    with h5py.File(gtilde_path, "r") as hf:
        stratum_mu = hf["stratum_mu"][:]              # (seeds, cells, fbins)
        cell = hf["concept_cell"][:]                  # (concepts,)
        _fre = hf.attrs["firing_rate_edges"]
        if isinstance(_fre, bytes):
            _fre = _fre.decode()
        # json.loads, NOT list(): write_identity json.dumps lists, so this
        # round-trips as a STRING and list() of it yields CHARACTERS (gotcha 20)
        edges = _json.loads(_fre) if isinstance(_fre, str) else list(_fre)
        stored_fb = (hf["feature_firing_bin"][:]
                     if "feature_firing_bin" in hf else None)

    n_bins = len(edges) - 1
    if stored_fb is not None:
        fb = stored_fb.astype(np.int64)
        source = "stored feature_firing_bin"
    else:
        fb = np.stack([np.clip(bin_index(fire_rate[s], edges), 0, n_bins - 1)
                       for s in range(fire_rate.shape[0])]).astype(np.int64)
        source = ("reconstructed from fire_rate + firing_rate_edges; exactness "
                  "proven in tests/test_standardize_gain.py")

    n_seeds, n_feats = fb.shape
    mu_feat = np.empty((n_seeds, n_feats), dtype=np.float64)
    for sd in range(n_seeds):
        m = stratum_mu[sd][cell[:, None], fb[sd][None, :]]
        mu_feat[sd] = np.median(m, axis=0)
    return mu_feat, {"feature_bin_source": source,
                     "firing_rate_edges": edges,
                     "n_firing_bins": int(n_bins)}


def _c1_summary(rhos):
    a = np.abs(np.asarray(rhos, dtype=float))
    return {
        "rho_by_seed": [float(r) for r in rhos],
        "median_abs_rho": float(np.median(a)),
        "max_abs_rho": float(a.max()),
        "min_abs_rho": float(a.min()),
        "n_seeds": int(len(rhos)),
        # DERIVED against C1's pre-declared 0.05 threshold (gotcha 20).
        "C1_abs_rho_exceeds_0.05": "HIT" if float(np.median(a)) > 0.05 else "MISS",
    }


def c1_only(args) -> None:
    """C1 across every seed. No fit, no regression, no C2.

    Amendment AC11. C1 is a correlation between two vectors that already
    exist, so it needs neither a model fit nor a regression and previews no
    primary coefficient. That is precisely why it is the arm that can safely
    be taken to all 20 seeds while C2 waits for the freeze list.
    """
    print("*** X7 FIREWALL CROSSING -- DELIBERATE, SCOPED, C1 ONLY "
          "(amendment AC11) ***", flush=True)
    with h5py.File(args.rf_counts, "r") as hf:
        counts = hf["match_count"][:]
        n_compare = int(hf.attrs["n_compare_seeds"])
        theta = float(hf.attrs["theta"])
    r_f = counts.astype(np.float64) / n_compare
    n_seeds = r_f.shape[0]

    if args.mu_smooth is None:
        raise SystemExit("--c1-only requires --mu-smooth")

    # IDENTITY CHECK ON THE NORMALISER (I3). This path previously opened
    # `mu_hat_full` and correlated it without asserting anything about which
    # label store or population it was built against -- so a mu_hat from a
    # different store would have produced a plausible rho. An ABSENT key is
    # reported ABSENT and refused, never treated as agreement.
    # Two kinds of key, and they need different comparisons -- a label store
    # publishes a population and a concept count but NOT an argmax pool size,
    # so demanding equality on the latter would compare against an absent value
    # and fail every time (which is a gate that cannot pass, the mirror of one
    # that cannot fail). `git_sha` is the store's own building SHA and is what
    # `library_sha` must equal (the "derived" comparison of amendment Q).
    lab_id = read_identity(args.labels, extra=("git_sha",))
    with h5py.File(args.mu_smooth, "r") as hf:
        mu_id = {k: (v.decode() if isinstance(v, bytes) else v)
                 for k, v in hf.attrs.items()}
    mu_problems = []
    # (a) compared against the store
    for key, want in (("truncation_policy", lab_id.get("truncation_policy")),
                      ("n_concepts", lab_id.get("n_concepts")),
                      ("library_sha", lab_id.get("git_sha"))):
        got = mu_id.get(key, "<ABSENT>")
        if got == "<ABSENT>":
            mu_problems.append(f"{key}: ABSENT from {args.mu_smooth.name} "
                               f"(absent is not matching -- I3)")
        elif want is None:
            mu_problems.append(f"{key}: the label store does not publish it, "
                               f"so this comparison cannot be made")
        elif str(got) != str(want):
            mu_problems.append(f"{key}: mu_smooth={got!r} but label store "
                               f"says {want!r}")
    # (b) `argmax_pool_n`'s peer is the AP MATRIX, not the store -- a label
    # store publishes a library, not a pool. `--ap` was already a required
    # argument of this job and was read by nothing, so the pool went unchecked
    # while the path that needed it had the file in hand. A max over a
    # different pool is a normaliser for a DIFFERENT statistic (I4).
    if "argmax_pool_n" not in mu_id:
        mu_problems.append("argmax_pool_n: ABSENT from "
                           f"{args.mu_smooth.name} -- a max over an "
                           "unspecified pool is a normaliser for an "
                           "unspecified statistic (absent is not matching)")
    elif getattr(args, "ap", None) is not None:
        with h5py.File(args.ap, "r") as _ap:
            ap_pool = _ap.attrs.get("argmax_pool_n")
        if ap_pool is None:
            mu_problems.append("argmax_pool_n: ABSENT from the AP matrix, so "
                               "the pool comparison cannot be made")
        elif int(mu_id["argmax_pool_n"]) != int(ap_pool):
            mu_problems.append(
                f"argmax_pool_n: mu_smooth={int(mu_id['argmax_pool_n'])} but "
                f"the AP matrix says {int(ap_pool)} -- amendment I4")
    if mu_problems:
        raise SystemExit(
            "PRE-FLIGHT FAILED on the smooth normaliser -- refusing to "
            "compute C1.\n  * " + "\n  * ".join(mu_problems) +
            "\n(It feeds AC10's pre-registered sensitivity arm, so an "
            "unidentified mu_hat is not acceptable input.)")
    print(f"[c1] mu_smooth identity OK: "
          f"truncation_policy={mu_id.get('truncation_policy')} "
          f"n_concepts={mu_id.get('n_concepts')} "
          f"argmax_pool_n={mu_id.get('argmax_pool_n')}", flush=True)

    # ---- SEED ALIGNMENT (amendment AS; the AR9 blast radius) --------------
    # This path was one of the two defects AR9 names. `mu_hat_full`, `gtilde`
    # and `fire_rate` are LEXICOGRAPHIC; `rf_counts` is NUMERIC; only rows 0
    # and 1 coincide. Correlating `mu[s]` against `r_f[s]` therefore paired 18
    # of 20 seeds' normalisers with a DIFFERENT seed's reproducibility, and
    # cross-seed pairings are approximately null (gotcha 18).
    from src.analysis.seed_axis import (align_to_reference, seed_order,
                                        stored_seed_labels)

    anchor_labels = stored_seed_labels(args.rf_counts)
    if anchor_labels is None:
        raise SystemExit("FATAL: rf_counts carries no seed labels; refusing to "
                         "assume an order (gotcha 15: absent is not matching).")
    nr = getattr(args, "null_reductions", None)
    null_labels = stored_seed_labels(nr) if nr is not None else None
    # `mu_hat_full` carries its own labels from amendment AS onward; before
    # that it inherits the null's order, which is why the null is the fallback
    # and NOT `range(n_seeds)`. If neither is available the job refuses rather
    # than defaulting to position -- that default IS the AR9 defect.
    mu_labels = stored_seed_labels(args.mu_smooth) or null_labels
    if mu_labels is None:
        raise SystemExit(
            "FATAL: neither the smooth mu artifact nor the null reductions "
            "carry seed labels, so mu_hat_full's row order is unknown. "
            "Refusing to assume row s is seed s (amendment AR9).")
    print(f"[c1] seed order: anchors {anchor_labels}\n"
          f"[c1]             mu_smooth {mu_labels}", flush=True)

    with h5py.File(args.mu_smooth, "r") as hf:
        mu = hf["mu_hat_full"]
        mu_feat = np.stack([np.median(mu[s], axis=0) for s in range(n_seeds)])
    mu_feat = align_to_reference(mu_labels, mu_feat, anchor_labels)
    rhos = [spearmanr(mu_feat[s], r_f[s]).statistic for s in range(n_seeds)]

    # ---- THE HARD-BIN ARM (close-out item 2). Hard-bin is the predictor of
    # record (AC9), and its mu ALSO depends on a feature-side axis (firing
    # rate), so the confound C1 tests was never exclusive to the smooth trend.
    hard = None
    gt = None
    if getattr(args, "gtilde", None) is not None:
        with h5py.File(args.null_reductions, "r") as _hf:
            fire_rate_null = _hf["fire_rate"][:]
        gt = seed_order(args.gtilde, args.null_reductions)
        # BUILT in the gtilde's own order (it indexes `stratum_mu[s]` against
        # `fire_rate[s]`), then the RESULT aligned onward to the anchors.
        fire_gt = align_to_reference(null_labels, fire_rate_null, gt["labels"])
        mu_hb, prov = hard_bin_mu_per_feature(args.gtilde, fire_gt)
        mu_hb = align_to_reference(gt["labels"], mu_hb, anchor_labels)
        rhos_hb = [spearmanr(mu_hb[s], r_f[s]).statistic
                   for s in range(n_seeds)]
        hard = {"provenance": prov, "c1": _c1_summary(rhos_hb),
                "seed_order_gtilde": gt}
        print("[c1] HARD-BIN arm:\n" + json.dumps(hard["c1"], indent=2),
              flush=True)
    else:
        print("[c1] no --gtilde: HARD-BIN arm skipped", flush=True)

    out = {
        "firewall_crossing": "DELIBERATE",
        "firewall_scope": (
            "C1 ONLY -- a correlation between two already-computed vectors. "
            "No model is fitted and no regression is run, so nothing here "
            "previews the primary coefficient. C2 is deferred until after the "
            "freeze list, and when it runs its relative-change metric is "
            "replaced by change in SE units (amendment AC11)."),
        "theta": theta,
        "normaliser": "smooth_trend",
        "note_on_hard_bin_RESOLVED": (
            "AC12 reported C1 for hard-bin as not computable because the "
            "per-feature firing-bin assignment was not on disk. That was "
            "WRONG: the assignment is a pure function of fire_rate and the "
            "firing_rate_edges attr, both already stored, and the two routes "
            "are proven to agree exactly. It is computed as c1_hard_bin above. "
            "The superseded note is kept verbatim below."),
        "note_on_hard_bin_SUPERSEDED": (
            "C1 is NOT computed for the hard-bin normaliser. Its stratum_mu is "
            "indexed by (concept cell, firing bin), and the per-feature firing "
            "bin assignment is not stored in the gtilde artifact, so a "
            "per-feature normaliser vector cannot be reconstructed from what "
            "is on disk. Reported as absent rather than approximated "
            "(gotcha 15). NOTE this leaves a real question open: hard-bin mu "
            "also depends on a FEATURE-SIDE axis -- firing rate -- so the "
            "confound C1 tests is not exclusive to the smooth trend."),
        "c1": _c1_summary(rhos),
        "c1_hard_bin": hard,
        "seed_alignment": {
            "anchor_order": anchor_labels,
            "null_order": null_labels,
            "mu_smooth_order": mu_labels,
            "gtilde_order": (gt["labels"] if gt else None),
            "n_rows_mispaired_by_a_positional_join": int(sum(
                a != b for a, b in zip(anchor_labels, mu_labels))),
            "note": ("AMENDMENT AS. Every normaliser is aligned to the ANCHOR "
                     "order before correlating. The pre-AS numbers correlated "
                     "18 of 20 seeds across AR9's lexicographic/numeric split."),
        },
    }
    print(json.dumps(out["c1"], indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[c1] wrote {args.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ap", type=Path, required=True)
    ap.add_argument("--null-raw", type=Path, required=True)
    ap.add_argument("--null-reductions", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--labels-json", type=Path, default=None)
    ap.add_argument("--row-index-cache", type=Path, required=True)
    ap.add_argument("--ckpts", type=Path, required=True)
    ap.add_argument("--eval-cache", type=Path, required=True)
    ap.add_argument("--scored-row-set", default="label_store")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--rf-counts", type=Path, required=True,
                    help="per-anchor MNN counts from rf_marginal --emit-counts")
    ap.add_argument("--arm", default="b", choices=["a", "b"])
    ap.add_argument("--stratum", default="pooled")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 7])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c1-only", action="store_true",
                    help="Amendment AC11. Compute C1 ONLY -- the correlation "
                         "between the normaliser and r_f -- across every seed, "
                         "with no model fitting and no regression. C2 is "
                         "deliberately unavailable here: fitting with "
                         "covariates in and out across 20 seeds would produce "
                         "a full-ensemble preview of the primary coefficient "
                         "before the specification is frozen, which is a "
                         "LARGER firewall crossing than was authorised and is "
                         "avoidable at no cost.")
    ap.add_argument("--gtilde", type=Path, default=None,
                    help="c1-only: the gtilde h5 whose stratum_mu IS the "
                         "hard-bin normaliser. Enables the hard-bin C1 arm.")
    ap.add_argument("--mu-smooth", type=Path, default=None,
                    help="c1-only: smooth mu_hat h5 from smooth_mu.py")
    args = ap.parse_args()

    if args.c1_only:
        c1_only(args)
        return

    print("*** X7 FIREWALL CROSSING -- DELIBERATE AND SCOPED (amendment AC2) "
          "***", flush=True)

    labels_json = args.labels_json or args.labels.with_suffix(".json")

    with h5py.File(args.rf_counts, "r") as hf:
        counts = hf["match_count"][:]                      # (seeds, features)
        n_compare = int(hf.attrs["n_compare_seeds"])
        theta = float(hf.attrs["theta"])
        if str(hf.attrs.get("similarity")) != "decoder_cosine":
            raise SystemExit("FATAL: r_f counts were not built on "
                             "decoder_cosine (amendment AB decision B).")
    r_f_all = counts.astype(np.float64) / n_compare
    print(f"r_f: {r_f_all.shape} at theta={theta}", flush=True)

    with h5py.File(args.ap, "r") as hf:
        g = hf[args.stratum]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in g["concepts"][:]]
        prevalence = g["prevalence"][:]
        ap_all = g["ap"][:]
        n_rows = int(hf.attrs["n_rows"])
    n_features = ap_all.shape[1]
    pool_mask = np.array(argmax_pool_mask(concepts))
    pool_idx = np.flatnonzero(pool_mask)
    pool_names = [concepts[i] for i in pool_idx]

    def _cov(seed: int) -> dict:
        ri = resolve_row_index(args.row_index_cache,
                               args.ckpts / f"seed_{seed}", args.eval_cache,
                               args.scored_row_set, args.k, n_rows)
        return assemble_covariates(args.labels, labels_json, ri, concepts,
                                   n_rows, n_features)
    con = _cov(args.seeds[0])["concept"]

    with h5py.File(args.null_reductions, "r") as rf:
        fire_rate = rf["fire_rate"][:]
    log_fire = np.log10(np.maximum(fire_rate, 1e-12))

    report: dict = {
        "firewall_crossing": "DELIBERATE",
        "firewall_scope": (
            "Joins r_f to alignment to answer a SPECIFICATION question "
            "(amendment AC section 2). Nothing about theta, the endpoint cuts, "
            "or the primary regression may be revised on the basis of this "
            "output. The crossing is not ambient: it is confined to this job."),
        "theta": theta, "seeds": list(args.seeds), "per_seed": {},
    }

    with h5py.File(args.null_raw, "r") as hf:
        null_concepts = [c.decode() if isinstance(c, bytes) else c
                         for c in hf["concepts"][:]]
        if list(null_concepts) != pool_names:
            raise SystemExit("FATAL: null pool axis != AP argmax pool (I3).")
        raw = hf[f"gain_{args.arm}"]

        for s in args.seeds:
            feat = _cov(s)["feature"]        # PER SEED (gotcha 18)
            y = np.maximum(raw[s].astype(np.float64).mean(axis=0), 0.0)
            real = ((ap_all[s].T.astype(np.float64) - prevalence[:, None])
                    / (1.0 - prevalence[:, None]))

            row = {}
            for label, covs in (("all_four", COVARIATES),
                                ("concept_side_only", CONCEPT_SIDE)):
                blocks, knots = _blocks(feat, con, pool_idx, tuple(covs))
                res = fit_smooth_mu(y, blocks)
                beta = np.asarray(res["fit"]["beta"])
                mu_full = predict_on_axis(beta, feat, con, knots, covs,
                                          n_concepts=len(concepts))
                logM = per_seed_logM(real, mu_full, pool_mask)
                fit = ols_logM_coef(r_f_all[s], logM, log_fire[s])
                # quantity 1: correlation of the NORMALISER with the outcome,
                # taken per feature (mu_hat varies by concept too, so the
                # per-feature median is the feature-level summary).
                mu_feat = np.median(mu_full, axis=0)
                rho = spearmanr(mu_feat, r_f_all[s]).statistic
                row[label] = {
                    "edf": res["fit"]["edf"], "lambda": res["fit"]["lambda"],
                    "n_covariates": len(covs),
                    "spearman_mu_hat_vs_r_f": float(rho),
                    **fit,
                }
            a, b = row["all_four"]["coef_logM"], row["concept_side_only"]["coef_logM"]
            row["comparison"] = {
                "coef_logM_all_four": a,
                "coef_logM_concept_side_only": b,
                "abs_relative_change": abs(b - a) / max(abs(a), 1e-30),
                "concept_only_is_larger_in_magnitude": bool(abs(b) > abs(a)),
            }
            report["per_seed"][str(s)] = row
            print(f"  seed {s}: coef(logM) all-four {a:+.5f} -> "
                  f"concept-only {b:+.5f} "
                  f"({100*row['comparison']['abs_relative_change']:.1f}% change) | "
                  f"rho(mu_hat, r_f) {row['all_four']['spearman_mu_hat_vs_r_f']:+.4f}",
                  flush=True)

    rhos = [report["per_seed"][str(s)]["all_four"]["spearman_mu_hat_vs_r_f"]
            for s in args.seeds]
    chg = [report["per_seed"][str(s)]["comparison"]["abs_relative_change"]
           for s in args.seeds]
    larger = [report["per_seed"][str(s)]["comparison"]
              ["concept_only_is_larger_in_magnitude"] for s in args.seeds]
    report["scorecard"] = {
        "C1_abs_rho_exceeds_0.05": {
            "rho_by_seed": rhos, "median_abs": float(np.median(np.abs(rhos))),
            "outcome": "HIT" if np.median(np.abs(rhos)) > 0.05 else "MISS",
            "category": "within-system comparison",
        },
        "C2_logM_coef_moves_more_than_10pct": {
            "relative_change_by_seed": chg,
            "median": float(np.median(chg)),
            "outcome": "HIT" if np.median(chg) > 0.10 else "MISS",
            "category": "within-system comparison",
        },
        "C3_concept_only_coef_larger_in_magnitude": {
            "by_seed": larger,
            "outcome": "HIT" if all(larger) else "MISS",
            "category": "DIRECTIONAL MECHANISM (0/7 record; filed per AB17)",
        },
    }
    report["decision"] = (
        "REPORT ONLY. Whether to split the trend into concept-side-only "
        "normalization is the author's (amendment AC section 6). No recommendation "
        "is made here.")
    print(json.dumps({"scorecard": report["scorecard"]}, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[m4-confound] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
