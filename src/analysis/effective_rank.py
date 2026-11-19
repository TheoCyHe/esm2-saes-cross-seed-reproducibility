"""
M2 -- EFFECTIVE RANK of decoder submatrices as a function of `r_f`.
PREREGISTRATION amendment AB15 (the author's RECORD), scoped and cleared at CF3a.

WHAT AB15 FIXES, and this module implements exactly that and nothing more:

  * PRIMARY: QUINTILES of `r_f`, 2,048 features per bin. Quintiles rather than
    deciles because `ER <= min(d, m)` and a decile is 1,024 features, BELOW
    d = 1,280 -- which would cap `ER/d <= 0.80` for every decile MECHANICALLY,
    just under Gerasimov's headline 0.80-0.81, so a reader could mistake an
    arithmetic ceiling for agreement. Bins are equal-sized by construction, so
    size-matching is automatic.
  * SECONDARY: DECILES, normalised by `min(d, m)` rather than by `d`, with BOTH
    normalisations reported side by side and THE CAP ARITHMETIC STATED IN THE
    OUTPUT so it cannot be misread later.
  * RETAINED: the ENDPOINT CONTRAST for Gerasimov comparability, reported WITH
    THE PURITY FIGURES ATTACHED so attenuation is visible rather than mistaken
    for a weak effect.
  * THE SIZE-MATCHED NULL CURVE: required for the endpoint arm (unequal sizes),
    and run for the quintiles too as the reference for what `ER/d` does under
    RANDOM selection at m = 2,048.
  * One SVD per submatrix, deriving ER under BOTH the l1 and energy
    normalisations, plus stable rank, participation ratio and EV_SVD(r).
    **l1 is the pre-declared primary, PROVISIONAL pending Appendix F.1, which
    remains unread.**

WHY A TREND AND NOT A TWO-GROUP CONTRAST. U6 committed ER on stable/unstable
submatrices, which presupposes SEPARATED groups. AB9's purity figures say ours
are 33.6% genuine at `p == 0` and 91.3% at `p < 0.10` -- ALWAYS STATE WHICH
DEFINITION A PURITY FIGURE USES -- and contamination attenuates a real
difference toward zero, stacking on the size-dependence artifact the null curve
exists to catch. Gerasimov could use endpoint sets because their `r_f` was
genuinely bimodal; ours is a populated continuum (AB10).

WHICH `r_f`. AB15 predates X2's theta grid and AK's move to n_compare = 39, so
it could not name them. The project's REFERENCE CONVENTION supplies both
(theta = 0.70, n_compare = 39) -- an inheritance, not a new declaration, the
same way AB13's "the eval split" supplied the FVU population (CF3a).

THE SEED AXIS IS NAMED, NEVER POSITIONAL. `--seeds` is explicit and the
`match_count` rows are addressed by the artifact's own `anchor_seeds` labels;
`glob`-order and numeric order disagree on 18 of 20 rows in this repo
(gotcha 29).
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import torch

# AB9's purity figures, QUOTED WITH THEIR DEFINITIONS because AB15 requires the
# definition to travel with the number. Read from PREREGISTRATION, not
# recomputed here -- this module does not own them.
PURITY = {
    "genuinely_unstable_at_p_equals_0": 0.336,
    "genuinely_unstable_at_p_lt_0.10": 0.913,
    "note": ("AB9, fitted by NPMLE against the observed marginal. The 'genuinely "
             "unstable' definition moves the k<=1 figure over a 2.7x range: "
             "33.6% at p == 0, 91.3% at p < 0.10. ALWAYS state which definition "
             "a purity figure uses (AB15). Contamination ATTENUATES a real ER "
             "difference toward zero, so the endpoint contrast understates."),
}


def spectrum_statistics(W: np.ndarray, d_model: int) -> dict:
    """Every spectral statistic AB15 names, from ONE SVD of one submatrix.

    `W` is (d_model, m) -- decoder COLUMNS for the selected features, which are
    unit-norm by construction.

    ER is Roy & Vetterli's effective rank, exp(H(p)), under two normalisations
    of the singular values:
      * l1     p_i = s_i / sum(s)          -- AB15's pre-declared PRIMARY
      * energy p_i = s_i^2 / sum(s^2)
    Both are computed because AB15 requires it and because Appendix F.1 is
    unread: if F.1 uses the energy form, the primary switches to a number
    already computed rather than needing a re-run.
    """
    m = W.shape[1]
    s_all = np.linalg.svd(W, compute_uv=False).astype(np.float64)
    # RELATIVE threshold, the same one `numpy.linalg.matrix_rank` uses. A bare
    # `s > 0` keeps floating-point denormal noise: a mathematically rank-1
    # submatrix came back with FOUR "non-zero" singular values at ~1e-17, which
    # is not a tolerance question but a wrong rank. The tiny values barely move
    # ER (p log p -> 0), so this was invisible in every entropy statistic and
    # visible only in the count -- which is why the count is asserted in the
    # tests (amendment CF3c).
    tol = max(W.shape) * np.finfo(np.float64).eps * (s_all.max() if s_all.size else 0.0)
    s = s_all[s_all > tol]
    out = {"m": int(m), "d_model": int(d_model), "n_nonzero_sv": int(s.size),
           "n_singular_values_total": int(s_all.size),
           "rank_tolerance": float(tol),
           "cap_min_d_m": int(min(d_model, m))}

    for name, p in (("l1", s / s.sum()),
                    ("energy", (s ** 2) / (s ** 2).sum())):
        # exp(Shannon entropy). 0*log0 is 0; p > 0 already by construction.
        er = float(np.exp(-(p * np.log(p)).sum()))
        out[f"ER_{name}"] = er
        out[f"ER_{name}_over_d"] = er / d_model
        out[f"ER_{name}_over_min_d_m"] = er / min(d_model, m)

    s2 = s ** 2
    out["stable_rank"] = float(s2.sum() / s2.max())
    out["participation_ratio"] = float((s2.sum() ** 2) / (s2 ** 2).sum())
    # EV_SVD(r): cumulative energy at r = 0.5 * min(d, m), a fixed fraction so
    # it is comparable across submatrices of different width.
    r = max(1, int(round(0.5 * min(d_model, m))))
    out["EV_SVD_r"] = float(s2[:r].sum() / s2.sum())
    out["EV_SVD_r_at"] = int(r)
    return out


def null_curve(W: np.ndarray, m: int, n_draws: int, rng: np.random.Generator,
               d_model: int) -> dict:
    """What ER/d does under RANDOM selection of `m` features.

    AB15 requires this for the endpoint arm (unequal sizes) and asks for it on
    the quintiles too, as the reference for what the statistic does when the
    selection carries no information. Without it a size-dependent statistic
    reads as an effect.
    """
    vals = {"ER_l1_over_d": [], "ER_energy_over_d": [], "stable_rank": []}
    n_feat = W.shape[1]
    for _ in range(n_draws):
        idx = rng.choice(n_feat, size=m, replace=False)
        st = spectrum_statistics(W[:, idx], d_model)
        for k in vals:
            vals[k].append(st[k])
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "n_draws": n_draws, "m": m}
            for k, v in vals.items()}


def bin_by_rf(rf: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Equal-sized bins of feature indices, ascending in `r_f`.

    Ties are broken by a STABLE sort on the index, so the partition is a
    deterministic function of (rf, n_bins) and does not depend on any RNG --
    `r_f` is heavily tied (it takes 40 values) and an unstable sort would make
    the bins irreproducible run to run.
    """
    order = np.argsort(rf, kind="stable")
    return [np.sort(b) for b in np.array_split(order, n_bins)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rf-counts", type=Path, required=True,
                   help="rf_counts_theta070_n39.h5 -- the REFERENCE theta")
    p.add_argument("--ckpts", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--n-compare", type=int, required=True)
    p.add_argument("--null-draws", type=int, default=20)
    # NO DEFAULTS (amendment CT, §CT8 item 3; corrections entry 271). The
    # defaults were Y2's N = 19 cuts (1 / 18), and the N = 39 run of record
    # applied them to N = 39 counts because the launcher passed neither. The
    # cut depends on n_compare (AG1: k <= 2 / k >= 37 at N = 39), so it must be
    # stated by the caller.
    p.add_argument("--endpoint-low", type=int, required=True,
                   help="k <= this is the UNSTABLE endpoint (AG1 at N = 39: 2; "
                        "Y2 at N = 19: 1)")
    p.add_argument("--endpoint-high", type=int, required=True,
                   help="k >= this is the STABLE endpoint (AG1 at N = 39: 37; "
                        "Y2 at N = 19: 18)")
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--out", type=Path, required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()

    with h5py.File(args.rf_counts, "r") as hf:
        counts = hf["match_count"][:]
        attrs = {k: str(v) for k, v in hf.attrs.items()}
        # `seed_labels` is what rf_marginal actually writes; the other two are
        # accepted because other artifacts in this repo use them. THE LIST WAS
        # INCOMPLETE ON THE FIRST RUN and the guard below refused rather than
        # joining positionally, which is the outcome it exists for (CF3d).
        #
        # An attr round-trips through json.dumps for lists, so it is PARSED
        # back rather than iterated -- list() of a JSON string is a list of
        # characters (gotcha 20).
        labels = None
        for key in ("seed_labels", "anchor_seeds", "seeds"):
            if key in hf:
                labels = [int(x) for x in hf[key][:]]
                break
            if key in hf.attrs:
                labels = [int(x) for x in json.loads(str(hf.attrs[key]))]
                break
    if labels is None:
        raise SystemExit(
            "the rf-counts artifact carries no seed labels; a positional join "
            "would mis-pair 18 of 20 rows (gotcha 29). Refusing.")
    print(f"[m2] counts {counts.shape}, seed labels {labels}", flush=True)

    rows = []
    d_model = None
    for s in args.seeds:
        if s not in labels:
            raise SystemExit(f"seed {s} absent from the artifact's labels")
        r = labels.index(s)                       # LABEL join, never positional
        rf = counts[r].astype(np.float64) / args.n_compare

        sd = torch.load(args.ckpts / f"seed_{s}" / "model.pt", map_location="cpu")
        sd = sd.get("model", sd)
        W = sd["W_dec"].numpy().astype(np.float64)    # (d_model, dict_size)
        d_model = W.shape[0]
        rng = np.random.default_rng(args.seed + s)

        rec = {"seed": s, "anchor_row": r, "n_features": int(W.shape[1]),
               "mean_r_f": float(rf.mean())}

        # ---- PRIMARY: quintiles, ER/d as a TREND ---------------------------
        rec["quintiles"] = []
        for q, idx in enumerate(bin_by_rf(rf, 5)):
            st = spectrum_statistics(W[:, idx], d_model)
            st.update({"bin": q, "mean_r_f": float(rf[idx].mean()),
                       "min_r_f": float(rf[idx].min()),
                       "max_r_f": float(rf[idx].max())})
            rec["quintiles"].append(st)
        rec["quintile_null"] = null_curve(W, len(bin_by_rf(rf, 5)[0]),
                                          args.null_draws, rng, d_model)

        # ---- SECONDARY: deciles, normalised by min(d, m) -------------------
        rec["deciles"] = []
        for q, idx in enumerate(bin_by_rf(rf, 10)):
            st = spectrum_statistics(W[:, idx], d_model)
            st.update({"bin": q, "mean_r_f": float(rf[idx].mean())})
            rec["deciles"].append(st)

        # ---- RETAINED: the endpoint contrast -------------------------------
        k = counts[r]
        lo = np.nonzero(k <= args.endpoint_low)[0]
        hi = np.nonzero(k >= args.endpoint_high)[0]
        ep = {"unstable_cut": f"k <= {args.endpoint_low}",
              "stable_cut": f"k >= {args.endpoint_high}",
              "n_unstable": int(lo.size), "n_stable": int(hi.size)}
        for nm, idx in (("unstable", lo), ("stable", hi)):
            if idx.size < 2:
                ep[nm] = {"status": "TOO_FEW_FEATURES", "n": int(idx.size)}
                continue
            ep[nm] = spectrum_statistics(W[:, idx], d_model)
            # UNEQUAL sizes here, so each endpoint needs its OWN size-matched
            # null -- this is the arm AB15 says the null curve is required for.
            ep[f"{nm}_size_matched_null"] = null_curve(
                W, int(idx.size), args.null_draws, rng, d_model)
        ep["purity"] = PURITY
        rec["endpoint_contrast"] = ep
        rows.append(rec)
        print(f"[m2] seed {s:2d}: ER_l1/d by quintile "
              + " ".join(f"{q['ER_l1_over_d']:.4f}" for q in rec["quintiles"]),
              flush=True)

    # ---- THE CAP ARITHMETIC, STATED IN THE OUTPUT (AB15) -------------------
    n_feat = rows[0]["n_features"]
    cap = {
        "why_quintiles_are_primary": (
            "ER <= min(d, m). A DECILE is "
            f"{n_feat // 10} features, BELOW d_model = {d_model}, so ER/d is "
            f"capped at {(n_feat // 10) / d_model:.4f} for EVERY decile, "
            "MECHANICALLY and regardless of the data. Gerasimov's stable-set "
            "value is ER/d ~ 0.80-0.81, so a decile analysis normalised by d "
            "would produce numbers pinned just under their headline figure for "
            "purely arithmetic reasons and could be misread as agreement. "
            f"A QUINTILE is {n_feat // 5} features, comfortably above "
            f"d_model = {d_model}, so the cap does not bind."),
        "decile_cap_ER_over_d": (n_feat // 10) / d_model,
        "quintile_cap_binds": bool((n_feat // 5) < d_model),
        "decile_cap_binds": bool((n_feat // 10) < d_model),
    }

    out = {
        "measurement": "effective_rank_M2_CF",
        "declaration": "PREREGISTRATION amendment AB15; scoped at CF3a",
        "primary": ("QUINTILE TREND in ER_l1/d over r_f. l1 is the "
                    "PRE-DECLARED primary and is PROVISIONAL pending Appendix "
                    "F.1, which remains UNREAD; the energy normalisation is "
                    "computed so a switch needs no re-run."),
        "r_f_convention": {
            "theta": 0.70, "n_compare": args.n_compare,
            "note": ("AB15 predates X2's theta grid and AK's n_compare = 39, "
                     "so it names neither. The project's reference convention "
                     "supplies both -- an inheritance, not a new declaration "
                     "(CF3a)."),
        },
        "cap_arithmetic": cap,
        "purity": PURITY,
        "identity": {
            "rf_counts": str(args.rf_counts), "rf_counts_attrs": attrs,
            "ckpts": str(args.ckpts), "anchor_seeds": list(args.seeds),
            "null_draws": args.null_draws, "rng_seed": args.seed,
            "python": platform.python_version(), "torch": torch.__version__,
        },
        "per_seed": rows,
    }

    # Cross-seed summary of the PRIMARY trend, per seed then combined -- never
    # a reduction over a feature-indexed axis (gotcha 18); the bins are
    # r_f-defined, so they are comparable across seeds by construction.
    prim = np.array([[q["ER_l1_over_d"] for q in r["quintiles"]] for r in rows])
    out["quintile_trend_ER_l1_over_d"] = {
        "by_bin_mean": prim.mean(axis=0).tolist(),
        "by_bin_sd": prim.std(axis=0, ddof=1).tolist(),
        "by_bin_min": prim.min(axis=0).tolist(),
        "by_bin_max": prim.max(axis=0).tolist(),
        "per_seed": prim.tolist(),
        # DERIVED, not asserted (gotcha 20).
        "n_seeds_monotone_increasing": int(
            sum(bool(np.all(np.diff(row) > 0)) for row in prim)),
        "n_seeds_monotone_decreasing": int(
            sum(bool(np.all(np.diff(row) < 0)) for row in prim)),
        "n_seeds": int(prim.shape[0]),
        "top_minus_bottom_by_seed": (prim[:, -1] - prim[:, 0]).tolist(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[m2] wrote {args.out}", flush=True)
    t = out["quintile_trend_ER_l1_over_d"]
    print(f"[m2] ER_l1/d by quintile (mean over seeds): "
          + " ".join(f"{v:.4f}" for v in t["by_bin_mean"]), flush=True)
    print(f"[m2] monotone increasing in {t['n_seeds_monotone_increasing']}"
          f"/{t['n_seeds']} seeds; decreasing in "
          f"{t['n_seeds_monotone_decreasing']}", flush=True)


if __name__ == "__main__":
    main()
