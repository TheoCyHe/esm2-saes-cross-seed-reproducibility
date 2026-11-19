"""D5 -- breadth of the winning concepts (PREREGISTRATION amendments D5, I5d).

THE OBLIGATION. D5's caveat is that pooled `g_f` may be close to a two-concept
detector: on the OLD pooled axis, `REGION` was the arg-max behind `g_f` for
most features clearing the effect-size floor. Amendment I5a **retracted** the
87.9%/84.8% figures (`ap_diagnostics.py` argmaxed RAW AP, whose baseline is
prevalence, so it systematically over-attributes to the broadest concepts --
precisely the quantity D5 reports). I5c then established that the obligation
attaches to the PRIMARY analysis, so recomputation is not optional.

THE CRITERION IS PRE-DECLARED IN I5d, committed before `main_ap` ran:

  (a) BREADTH -- median prevalence of the argmax concept across aligned
      features. Old axis: ~0.13. SUCCESS = a drop of at least one order of
      magnitude, i.e. **median winner prevalence <= 0.013**. (Library median
      concept prevalence is 0.000085, so there is headroom.)

  (b) CONCENTRATION -- share of aligned features whose argmax falls on the
      TEN BROADEST CONCEPTS IN THE ARGMAX POOL (computed over the pool, so
      the excluded pooled parents are not counted), compared against the
      SHUFFLED-LABEL NULL: under shuffled labels, what concentration does the
      library's own prevalence distribution produce by itself? SUCCESS =
      observed concentration **not meaningfully above** that null.

  If both pass, the obligation is DISCHARGED EXPLICITLY with the numbers -- a
  standing obligation nobody follows reads as non-compliance, not resolution.
  If either fails, the pre-recorded reading is that sub-typing RELOCATED the
  granularity problem rather than fixing it, most likely onto generic
  sub-concepts such as `REGION:Disordered` (p = 0.049, the broadest concept
  remaining in the pool).

WHAT "ALIGNED" MEANS HERE. The pre-registered classification rule (FDR + the
`G >= 0.05` floor) needs a threshold design that does not exist yet (P2). This
script uses the **pre-declared effect-size floor `G >= 0.05` alone**, which is
the same G-proxy every prior D-numbered diagnostic used, and says so in its
output. It does NOT invent a threshold.

THE NULL IS THE REAL ONE. (b)'s shuffled-label reference is computed from the
null's own raw per-(seed, shift, concept, feature) gain array (amendment K3),
taking the argmax concept per feature per replicate. That is exactly "what
concentration the prevalence distribution produces by itself", measured rather
than modelled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask
from src.sae.train import git_dirty, git_sha

G_FLOOR = 0.05          # pre-declared effect-size floor (PREREGISTRATION §2)
BREADTH_SUCCESS = 0.013  # I5d(a): one order of magnitude below ~0.13
N_BROADEST = 10          # I5d(b)

# Concept TYPES the pre-registration excludes for PROVENANCE (§1, gotcha 9):
#   HELIX/STRAND/TURN -- from PDB cross-references, so they exist only for the
#     ~10% of Swiss-Prot with solved structures. "Absent" means NOT MEASURED,
#     while AP scores every unlabelled row as a negative.
#   COMPBIAS -- zero experimental residues out of 405,798.
# They are NOT filtered anywhere in the label build, so they are present in the
# library and can win `g_f`'s argmax. Reported here so the contamination is
# visible in the number rather than discovered later.
PROVENANCE_EXCLUDED_TYPES = ("HELIX", "STRAND", "TURN", "COMPBIAS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--raw-null", type=Path, default=None,
                   help="null_gain_raw_*.h5 (K3). Needed for criterion (b)'s "
                        "shuffled-label reference; without it (b) is skipped "
                        "rather than approximated.")
    p.add_argument("--null-arm", default="b", choices=["a", "b"],
                   help="Null B is primary (amendment G4).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with h5py.File(args.ap, "r") as hf:
        g = hf["pooled"]
        concepts = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in g["concepts"][:]])
        ap = g["ap"][:]                      # (seeds, features, concepts)
        prev = g["prevalence"][:]
        pool_attr = int(hf.attrs["argmax_pool_n"])

    pool = np.array(argmax_pool_mask(concepts))
    assert pool.sum() == pool_attr, (pool.sum(), pool_attr)
    pool_idx = np.flatnonzero(pool)
    pool_concepts, pool_prev = concepts[pool], prev[pool]

    # Normalised gain, then mask to the pool. G1: normalize BEFORE the max.
    gain = (ap - prev[None, None, :]) / (1.0 - prev[None, None, :])
    gain_pool = gain[:, :, pool_idx]                       # (seeds, feat, 707)

    winner = gain_pool.argmax(axis=2)                      # (seeds, features)
    g_f = gain_pool.max(axis=2)
    aligned = g_f >= G_FLOOR

    n_seeds, n_feat = g_f.shape
    win_prev = pool_prev[winner]

    # ---- (a) breadth --------------------------------------------------
    per_seed_median = [float(np.median(win_prev[s][aligned[s]]))
                       if aligned[s].any() else float("nan")
                       for s in range(n_seeds)]
    all_aligned_prev = win_prev[aligned]
    median_winner_prev = float(np.median(all_aligned_prev)) if all_aligned_prev.size else float("nan")
    passes_a = bool(median_winner_prev <= BREADTH_SUCCESS)

    # ---- (b) concentration --------------------------------------------
    broadest = np.argsort(pool_prev)[::-1][:N_BROADEST]
    broad_set = set(broadest.tolist())
    obs_share = (float(np.isin(winner[aligned], broadest).mean())
                 if aligned.any() else float("nan"))

    null_share = None
    null_detail = None
    if args.raw_null is not None:
        key = f"gain_{args.null_arm}"
        with h5py.File(args.raw_null, "r") as hf:
            null_concepts = np.array([c.decode() if isinstance(c, bytes) else c
                                      for c in hf["concepts"][:]])
            assert list(null_concepts) == list(pool_concepts), (
                "null concept axis != AP argmax pool -- refusing to compare "
                "(the identity failure amendment I4 is about)")
            d = hf[key]                      # (seeds, shifts, concepts, feat)
            ns, nsh = d.shape[0], d.shape[1]
            shares = []
            for s in range(ns):
                for r in range(nsh):
                    w = np.asarray(d[s, r]).argmax(axis=0)   # per feature
                    shares.append(float(np.isin(w, broadest).mean()))
        null_share = float(np.mean(shares))
        sh = np.asarray(shares)
        # ---- N4: THREE SEPARATE QUESTIONS, NOT ONE VERDICT --------------
        # M1's ">= 50% relative excess" rule is WITHDRAWN: it was declared
        # after observing +13%, i.e. chosen to pass, and disclosing that does
        # not repair it. No threshold replaces it -- the single-verdict
        # framing was the error, not the number.
        #
        # (1) ABOVE CHANCE? -> a permutation p-value straight from the
        #     replicates. The standard (r+1)/(n+1) form, which is the
        #     unbiased estimator and never returns exactly 0 for a finite
        #     permutation set -- reporting p=0 off 400 draws would overstate
        #     what 400 draws can establish.
        n_ge = int((sh >= obs_share).sum())
        p_perm = (n_ge + 1) / (len(sh) + 1)
        null_detail = {"mean": null_share, "sd": float(sh.std()),
                       "min": float(sh.min()), "max": float(sh.max()),
                       "n_replicates": len(sh), "arm": args.null_arm,
                       "n_replicates_ge_observed": n_ge,
                       "permutation_p": p_perm,
                       "relative_excess": (obs_share / null_share - 1.0
                                           if null_share else float("nan")),
                       "absolute_excess_pp": (obs_share - null_share) * 100}
        passes_b = None      # deliberately no verdict; see the three answers
    else:
        passes_b = None

    print(f"AP artifact  : {args.ap}")
    print(f"argmax pool  : {pool.sum()} concepts "
          f"(library {len(concepts)}, {len(concepts)-pool.sum()} parents excluded)")
    print(f"aligned rule : G >= {G_FLOOR} (pre-declared effect-size floor; NOT "
          f"the full FDR rule, which needs P2)")
    print(f"aligned      : {int(aligned.sum()):,} of {n_seeds*n_feat:,} "
          f"feature-seeds ({aligned.sum()/(n_seeds*n_feat):.3%}), "
          f"{aligned.sum()/n_seeds:.1f}/seed")
    print()
    print("--- I5d(a) BREADTH -------------------------------------------")
    print(f"  median winner prevalence : {median_winner_prev:.6g}")
    print(f"  success threshold        : <= {BREADTH_SUCCESS} "
          f"(one order of magnitude below the old axis' ~0.13)")
    print(f"  library median prevalence: 0.000085")
    print(f"  per-seed medians         : min {np.nanmin(per_seed_median):.6g} "
          f"max {np.nanmax(per_seed_median):.6g}")
    print(f"  VERDICT                  : {'PASS' if passes_a else 'FAIL'}")
    print()
    print("--- I5d(b) CONCENTRATION -------------------------------------")
    print(f"  ten broadest pool concepts:")
    for i in broadest:
        print(f"     {pool_concepts[i]:<44} p={pool_prev[i]:.4f}")
    print(f"  observed share of aligned winners in the top 10: {obs_share:.4%}")
    if null_share is not None:
        nd = null_detail
        print(f"  shuffled-label null share                     : "
              f"{null_share:.4%} (sd {nd['sd']:.4%}, "
              f"{nd['n_replicates']} replicates, Null {args.null_arm.upper()})")
        print()
        print("  N4: reported as THREE SEPARATE QUESTIONS. M1's >=50% "
              "relative-excess rule is WITHDRAWN (declared after seeing +13%,")
        print("      i.e. chosen to pass). No threshold replaces it -- the "
              "single-verdict framing was the error, not the number.")
        print(f"    1. ABOVE CHANCE?  permutation p = {nd['permutation_p']:.4g}  "
              f"({nd['n_replicates_ge_observed']} of {nd['n_replicates']} "
              f"replicates >= observed)")
        print(f"    2. LARGE?         +{nd['absolute_excess_pp']:.2f} pp "
              f"absolute, {nd['relative_excess']:+.1%} relative")
        print(f"    3. THE PRE-RECORDED FAILURE MODE?  see the winner "
              f"attribution below -- I5d predicted breadth-dominance,")
        print(f"       most likely REGION:Disordered.")
    else:
        print("  shuffled-label null      : SKIPPED (--raw-null not given)")

    # ---- PROVENANCE CONTAMINATION of the argmax pool --------------------
    prov_mask = np.array([c.split(":")[0] in PROVENANCE_EXCLUDED_TYPES
                          for c in pool_concepts])
    prov_names = pool_concepts[prov_mask].tolist()
    prov_win = np.isin(winner[aligned], np.flatnonzero(prov_mask))
    prov_share = float(prov_win.mean()) if aligned.any() else float("nan")
    n_prov_in_broadest = int(prov_mask[broadest].sum())
    print()
    print("--- PROVENANCE CONTAMINATION (not part of I5d; found while running it)")
    print(f"  concepts the pre-registration excludes for provenance that are "
          f"nonetheless IN the argmax pool: {len(prov_names)}")
    for c in prov_names:
        i = int(np.flatnonzero(pool_concepts == c)[0])
        print(f"     {c:<44} p={pool_prev[i]:.4f}")
    print(f"  of them in the ten broadest: {n_prov_in_broadest}")
    print(f"  share of aligned features whose ARGMAX is one of them: "
          f"{prov_share:.4%}")

    # ---- SENSITIVITY: both criteria on the PRE-REGISTRATION-COMPLIANT pool
    # Dropping the provenance-excluded concepts changes BOTH the argmax (for
    # the features they were winning) and the identity of the ten broadest.
    # Reported as a sensitivity arm and NOT swapped in as primary: the primary
    # is the pool `feature_ap_main.h5` and the null were actually built on.
    # Which of the two SHOULD be primary is a pre-registration question, not a
    # choice to make by seeing which one passes -- flagged for the author.
    clean = ~prov_mask
    clean_idx = np.flatnonzero(clean)
    gain_clean = gain_pool[:, :, clean_idx]
    winner_c = gain_clean.argmax(axis=2)
    aligned_c = gain_clean.max(axis=2) >= G_FLOOR
    cprev = pool_prev[clean]
    med_c = float(np.median(cprev[winner_c][aligned_c])) if aligned_c.any() else float("nan")
    broad_c = np.argsort(cprev)[::-1][:N_BROADEST]
    obs_c = float(np.isin(winner_c[aligned_c], broad_c).mean())
    null_c = None
    if args.raw_null is not None:
        with h5py.File(args.raw_null, "r") as hf:
            d = hf[f"gain_{args.null_arm}"]
            sh = []
            for s in range(d.shape[0]):
                for r in range(d.shape[1]):
                    w = np.asarray(d[s, r])[clean_idx].argmax(axis=0)
                    sh.append(float(np.isin(w, broad_c).mean()))
        null_c = {"mean": float(np.mean(sh)), "sd": float(np.std(sh))}
    print()
    print("--- SENSITIVITY: pre-registration-compliant pool "
          f"({int(clean.sum())} concepts, 4 provenance-excluded dropped) ---")
    print(f"  (a) median winner prevalence : {med_c:.6g}  "
          f"{'PASS' if med_c <= BREADTH_SUCCESS else 'FAIL'}")
    print(f"      ten broadest now         : "
          f"{[str(x) for x in pool_concepts[clean][broad_c][:4]]} ...")
    print(f"  (b) observed top-10 share    : {obs_c:.4%}")
    if null_c:
        ok_c = obs_c <= null_c["mean"] + 2 * null_c["sd"]
        print(f"      shuffled-label null      : {null_c['mean']:.4%} "
              f"(sd {null_c['sd']:.4%})   {'PASS' if ok_c else 'FAIL'}")

    # Winner attribution, the descriptive half D5 has always reported.
    from collections import Counter
    cnt = Counter(pool_concepts[winner[aligned]].tolist())
    print()
    print("--- winner attribution (top 12) ------------------------------")
    for name, n in cnt.most_common(12):
        i = int(np.flatnonzero(pool_concepts == name)[0])
        print(f"  {name:<46} {n:>7} ({n/aligned.sum():6.2%})  p={pool_prev[i]:.5f}")

    payload = {
        "ap_artifact": str(args.ap), "raw_null": str(args.raw_null),
        "aligned_rule": f"G >= {G_FLOOR} (effect-size floor only; no FDR, P2 absent)",
        "argmax_pool_n": int(pool.sum()), "n_concepts_library": len(concepts),
        "n_aligned_feature_seeds": int(aligned.sum()),
        "aligned_per_seed": float(aligned.sum() / n_seeds),
        "criterion_a": {"median_winner_prevalence": median_winner_prev,
                        "success_threshold": BREADTH_SUCCESS,
                        "per_seed_medians": per_seed_median,
                        "passes": passes_a},
        "criterion_b_note": ("N4: reported as three separate questions, not a "
                             "verdict. M1's >=50% relative-excess rule is "
                             "WITHDRAWN -- declared after observing +13%."),
        "criterion_b": {"observed_top10_share": obs_share,
                        "ten_broadest": [[str(pool_concepts[i]), float(pool_prev[i])]
                                         for i in broadest],
                        "shuffled_label_null": null_detail,
                        "passes": passes_b},
        "provenance_contamination": {
            "excluded_types": list(PROVENANCE_EXCLUDED_TYPES),
            "present_in_argmax_pool": prov_names,
            "n_in_ten_broadest": n_prov_in_broadest,
            "share_of_aligned_winners": prov_share,
            "note": ("PREREGISTRATION section 1 excludes these for provenance "
                     "but nothing in build_residue_labels.py enforces it, so "
                     "they are in the library and can win g_f's argmax."),
        },
        "sensitivity_compliant_pool": {
            "n_concepts": int(clean.sum()),
            "median_winner_prevalence": med_c,
            "passes_a": bool(med_c <= BREADTH_SUCCESS),
            "observed_top10_share": obs_c,
            "shuffled_label_null": null_c,
            "passes_b": (None if null_c is None else
                         bool(obs_c <= null_c["mean"] + 2 * null_c["sd"])),
            "note": ("Sensitivity only. Primary is the 707 pool that "
                     "feature_ap_main.h5 and the null were actually built on. "
                     "Which SHOULD be primary is a pre-registration question."),
        },
        "winner_attribution_top12": [
            [name, n, float(n / aligned.sum())] for name, n in cnt.most_common(12)],
        "prereg": "amendments D5 + I5d",
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
