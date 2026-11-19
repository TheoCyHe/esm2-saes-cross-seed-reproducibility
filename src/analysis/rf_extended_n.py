"""The EXTENDED-N comparison. Governed by PREREGISTRATION amendments AG and AK.

TWO JOBS, AND THEY HAVE OPPOSITE ACCEPTANCE CRITERIA. Collapsing them is the
failure this module exists to prevent (AK1):

  * the **N = 19 arm** is the sensitivity arm of record and must reproduce the
    canary baseline **BIT-IDENTICALLY** -- a movement there is a refactor
    defect, whatever its magnitude;
  * the **N = 39 arm** is new and is judged against AK's **pre-committed
    predictions**. An N = 39 artifact matching N = 19 exactly would be evidence
    the extension was never read.

ADDED KEYS ARE ALLOWLISTED EXPLICITLY, NEVER FILTERED HEURISTICALLY. The
anchor/comparison split adds keys to the M1 artifact, so byte equality of the
whole document is not the criterion and pretending otherwise would force either
a false failure or a loosened comparison. Instead: every key present in the
BASELINE must be present in the candidate with an identical value, and the set
of ADDED keys must equal the allowlist below exactly. An addition nobody
declared is a FAILURE, not a pass -- the same discipline AJ4 applied to the
retired-literal audit, where a heuristic filter is how an instrument gets
switched off rather than fixed.

REPORTED AS A DIFF, NOT AS A VERDICT (AD2, gotcha 20). Every outcome word in
the output is computed from the data it just compared. The process exit code
carries the verdict so an `afterok` chain or a `&&` can read it -- a validator
whose failure is unreadable by the thing gating on it is worse than none
(gotcha 13 extension 1).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# The declared additions. Paths are `/`-joined key paths; `*` matches one level.
# Declared in PREREGISTRATION amendment AK1 BEFORE the run, so a key appearing
# that is not here means the refactor did something that was not described.
# ---------------------------------------------------------------------------
DECLARED_ADDED_JSON_KEYS = (
    "identity/n_anchors",
    "identity/n_compare_seeds",
    "identity/anchor_seed_labels",
    "identity/compare_only_seed_labels",
    "identity/n_pairs_computed",
    "identity/n_pairs_skipped_extension_extension",
    "preflight/b_dec_distinctness",
    "mnn_oneway_ratio_curve",
    "marginals_by_theta/*/*/ag_endpoint_groups",
)

DECLARED_ADDED_H5_ATTRS = ("n_anchors", "compare_only_seed_labels")

# ---------------------------------------------------------------------------
# A BIT-IDENTITY CRITERION APPLIED TO A PROVENANCE KEY IS A CHECK THAT CAN ONLY
# FAIL -- the mirror of one that can never fail, and equally useless.
# Amendment AL7, and it was found by RUNNING the comparison, not by reading it.
#
# `identity/git_sha` records the commit that PERFORMED the run. The canary
# baseline was written at a0b5fbb and any re-run is necessarily at a later
# commit, so demanding bit-identity here guarantees a DIFFERS verdict forever,
# no matter what the statistics do. The first run reported exactly that: one
# disagreement, `identity/git_sha`, with `match_count` bit-identical.
#
# HONESTY ABOUT THE ORDERING, because AD2 forbids naming a tolerance change
# after seeing the comparison. This exemption WAS added after seeing the
# failure. It is defensible only because of what it is and is not:
#   * it is NOT a tolerance on any statistic -- no threshold anywhere is
#     loosened, and every remaining key is still compared for exact equality;
#   * the excluded set is TWO keys, named by FULL PATH, both of which are
#     definitionally properties of the comparing run rather than of the result;
#   * the evidence that nothing hides behind it is the failing run's own
#     output: it reported ONE disagreement and named it.
#
# EXEMPTION IS BY FULL PATH, NEVER BY LEAF NAME, and this distinction is
# load-bearing. `preflight/ensemble_identity/keys/git_sha/value` is the FORTY
# CHECKPOINTS' shared SHA -- the design's central invariant, which must never
# move. A leaf-name match would exempt it too and silently stop checking the one
# `git_sha` that matters.
# ---------------------------------------------------------------------------
PROVENANCE_PATHS_EXPECTED_TO_DIFFER = ("identity/git_sha", "identity/git_dirty")
PROVENANCE_H5_ATTRS_EXPECTED_TO_DIFFER = ("git_sha", "git_dirty")


def _split_provenance(dis: list[dict], paths: tuple[str, ...],
                      key: str = "path") -> tuple[list[dict], list[dict]]:
    """Partition disagreements into (statistic, provenance).

    Exact full-path membership, so a nested `git_sha` elsewhere in the document
    stays a statistic disagreement.
    """
    prov = [d for d in dis if d.get(key) in paths]
    stat = [d for d in dis if d.get(key) not in paths]
    return stat, prov


def _matches_declared(path: str, patterns: tuple[str, ...]) -> bool:
    """Does `path` match a declared pattern, treating `*` as one path level?

    A declared PREFIX covers everything beneath it: `preflight/b_dec_distinctness`
    licenses the whole subtree it introduces, because the subtree arrives as one
    addition rather than as a family of independent ones.
    """
    parts = path.split("/")
    for pat in patterns:
        pp = pat.split("/")
        if len(pp) > len(parts):
            continue
        if all(p == "*" or p == q for p, q in zip(pp, parts)):
            return True
    return False


def json_diff(baseline, candidate, path: str = "") -> tuple[list[dict], list[str]]:
    """Recursive comparison. Returns (disagreements, added_paths).

    A disagreement is any baseline leaf whose value differs, or any baseline key
    absent from the candidate. Floats are compared with `==` deliberately:
    AD2 fixes the tolerance at bit-identical for these quantities, and a
    tolerance introduced here is exactly the move that lets a real change be
    waved through as noise.
    """
    dis: list[dict] = []
    add: list[str] = []

    if isinstance(baseline, dict):
        if not isinstance(candidate, dict):
            return ([{"path": path, "problem": "TYPE_CHANGED",
                      "baseline_type": "dict",
                      "candidate_type": type(candidate).__name__}], [])
        for k, v in baseline.items():
            sub = f"{path}/{k}" if path else str(k)
            if k not in candidate:
                dis.append({"path": sub, "problem": "MISSING_IN_CANDIDATE"})
                continue
            d, a = json_diff(v, candidate[k], sub)
            dis += d
            add += a
        for k in candidate:
            if k not in baseline:
                add.append(f"{path}/{k}" if path else str(k))
        return dis, add

    if isinstance(baseline, list):
        if not isinstance(candidate, list):
            return ([{"path": path, "problem": "TYPE_CHANGED",
                      "baseline_type": "list",
                      "candidate_type": type(candidate).__name__}], [])
        if len(baseline) != len(candidate):
            return ([{"path": path, "problem": "LENGTH_CHANGED",
                      "baseline": len(baseline),
                      "candidate": len(candidate)}], [])
        for i, (b, c) in enumerate(zip(baseline, candidate)):
            d, a = json_diff(b, c, f"{path}[{i}]")
            dis += d
            add += a
        return dis, add

    # ---- leaf ----
    if isinstance(baseline, float) and isinstance(candidate, float):
        same = (baseline == candidate) or (math.isnan(baseline)
                                           and math.isnan(candidate))
    else:
        same = baseline == candidate and type(baseline) is type(candidate)
    if not same:
        dis.append({"path": path, "problem": "VALUE_CHANGED",
                    "baseline": baseline, "candidate": candidate})
    return dis, add


def compare_n19(baseline_path: Path, candidate_path: Path) -> dict:
    """The canary comparison for the N = 19 sensitivity arm."""
    base = json.loads(baseline_path.read_text())
    cand = json.loads(candidate_path.read_text())
    dis, add = json_diff(base, cand)
    stat, prov = _split_provenance(dis, PROVENANCE_PATHS_EXPECTED_TO_DIFFER)
    undeclared = [p for p in add
                  if not _matches_declared(p, DECLARED_ADDED_JSON_KEYS)]
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "n_baseline_disagreements": len(dis),
        "n_statistic_disagreements": len(stat),
        "statistic_disagreements": stat[:40],
        # Reported, NOT gating, and enumerated so the exemption is visible in
        # the artifact rather than only in the source (AL7).
        "provenance_paths_expected_to_differ":
            list(PROVENANCE_PATHS_EXPECTED_TO_DIFFER),
        "n_provenance_disagreements": len(prov),
        "provenance_disagreements": prov,
        "n_added_keys": len(add),
        "added_keys": sorted(add),
        "undeclared_added_keys": sorted(undeclared),
        # DERIVED from the lists just built, never asserted (gotcha 20). The
        # verdict turns on STATISTIC disagreements; a provenance difference is
        # expected and is reported beside it.
        "verdict": ("REPRODUCES" if not stat and not undeclared
                    else "DIFFERS"),
    }


def compare_counts_h5(baseline_path: Path, candidate_path: Path) -> dict:
    """`match_count` must be bit-identical; baseline attrs must be unchanged.

    The h5 FILE cannot be byte-identical -- two attrs are added -- so the
    comparison is on the dataset and on every baseline attr's value. Stating
    that here rather than quietly comparing something weaker.
    """
    import h5py
    with h5py.File(baseline_path, "r") as b, h5py.File(candidate_path, "r") as c:
        bd = b["match_count"][:]
        cd = c["match_count"][:]
        battrs = {k: b.attrs[k] for k in b.attrs}
        cattrs = {k: c.attrs[k] for k in c.attrs}

    shape_same = bd.shape == cd.shape
    identical = bool(shape_same and np.array_equal(bd, cd))
    attr_dis = []
    for k, v in battrs.items():
        if k not in cattrs:
            attr_dis.append({"attr": k, "problem": "MISSING_IN_CANDIDATE"})
            continue
        bv = v.decode() if isinstance(v, bytes) else v
        cv = cattrs[k].decode() if isinstance(cattrs[k], bytes) else cattrs[k]
        if np.ndim(bv) == 0 and np.ndim(cv) == 0:
            same = bv == cv
        else:
            same = np.array_equal(np.asarray(bv), np.asarray(cv))
        if not same:
            attr_dis.append({"attr": k, "problem": "VALUE_CHANGED",
                             "baseline": str(bv), "candidate": str(cv)})
    added = [k for k in cattrs if k not in battrs]
    undeclared = [k for k in added if k not in DECLARED_ADDED_H5_ATTRS]
    stat_attr, prov_attr = _split_provenance(
        attr_dis, PROVENANCE_H5_ATTRS_EXPECTED_TO_DIFFER, key="attr")
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_shape": list(bd.shape),
        "candidate_shape": list(cd.shape),
        "match_count_bit_identical": identical,
        "n_differing_cells": (int((bd != cd).sum()) if shape_same else None),
        "statistic_attr_disagreements": stat_attr,
        "provenance_attrs_expected_to_differ":
            list(PROVENANCE_H5_ATTRS_EXPECTED_TO_DIFFER),
        "provenance_attr_disagreements": prov_attr,
        "added_attrs": sorted(added),
        "undeclared_added_attrs": sorted(undeclared),
        "verdict": ("REPRODUCES" if identical and not stat_attr and not undeclared
                    else "DIFFERS"),
    }


# ---------------------------------------------------------------------------
# AK's prediction scorecard. Every outcome is DERIVED (gotcha 20), and every
# predicate is total over the statistic's range (gotcha 24) -- there is no
# outcome for which a prediction has no branch.
# ---------------------------------------------------------------------------

REF_THETA = "0.70"


def _mnn(doc: dict, theta: str = REF_THETA) -> dict:
    return doc["marginals_by_theta"][theta]["mnn"]


def score_predictions(n19: dict, n39: dict,
                      mix19: dict | None, mix39: dict | None) -> dict:
    """Score AK2-AK4 from the two M1 documents and the two mixture documents."""
    m19, m39 = _mnn(n19), _mnn(n39)
    i19, i39 = n19["identity"], n39["identity"]
    out: dict = {}

    # ---- A1: the run consumed 40 dictionaries -------------------------------
    out["A1_consumed_40_dictionaries"] = {
        "kind": "correctness assertion",
        "n_seeds": i39.get("n_seeds"),
        "n_anchors": i39.get("n_anchors"),
        "n_compare_seeds": i39.get("n_compare_seeds"),
        "n_pairs_computed": i39.get("n_pairs_computed"),
        "expected_pairs": 590,
        "outcome": "PASS" if (i39.get("n_seeds") == 40
                              and i39.get("n_anchors") == 20
                              and i39.get("n_compare_seeds") == 39
                              and i39.get("n_pairs_computed") == 590)
                   else "FAIL",
    }

    # ---- A2: rotation null and Hungarian arm bit-identical ------------------
    rot_same = (n19["rotation_null"]["floor"] == n39["rotation_null"]["floor"]
                and n19["rotation_null"]["mc_se"] == n39["rotation_null"]["mc_se"])
    hung_dis, _ = json_diff(n19["hungarian_arm"], n39["hungarian_arm"])
    out["A2_rotation_null_and_hungarian_bit_identical"] = {
        "kind": "correctness assertion",
        "rotation_floor_n19": n19["rotation_null"]["floor"],
        "rotation_floor_n39": n39["rotation_null"]["floor"],
        "rotation_bit_identical": bool(rot_same),
        "n_hungarian_disagreements": len(hung_dis),
        "hungarian_disagreements": hung_dis[:8],
        "outcome": "PASS" if rot_same and not hung_dis else "FAIL",
        "note": ("Both are computed from the seed-0/seed-1 decoders with a "
                 "pinned --rng-seed, so bit-identity is the right tolerance "
                 "and Monte Carlo error is not (AK2 A2)."),
    }

    # ---- A3: MNN nested inside one-way --------------------------------------
    out["A3_mnn_nested_in_oneway"] = {
        "kind": "correctness assertion",
        "n_violations_n39": n39["prediction_scorecard"][
            "P1_mnn_nested_in_oneway"]["n_violations"],
        "outcome": n39["prediction_scorecard"][
            "P1_mnn_nested_in_oneway"]["outcome"],
    }

    # ---- A4: P(k=0) and P(k=N) strictly decrease ---------------------------
    out["A4_endpoint_atoms_decrease_with_N"] = {
        "kind": "correctness assertion (near-analytic)",
        "P_k_eq_0": {"n19": m19["frac_r_eq_0"], "n39": m39["frac_r_eq_0"],
                     "decreased": bool(m39["frac_r_eq_0"] < m19["frac_r_eq_0"])},
        "P_k_eq_N": {"n19": m19["frac_r_eq_1"], "n39": m39["frac_r_eq_1"],
                     "decreased": bool(m39["frac_r_eq_1"] < m19["frac_r_eq_1"])},
        "outcome": "PASS" if (m39["frac_r_eq_0"] < m19["frac_r_eq_0"]
                              and m39["frac_r_eq_1"] < m19["frac_r_eq_1"])
                   else "FAIL",
    }

    # ---- B1: mean r_f agrees within 0.01 -----------------------------------
    d_mean = abs(m39["pairwise_match_rate"] - m19["pairwise_match_rate"])
    out["B1_mean_rf_within_0.01"] = {
        "kind": "within-system comparison",
        "n19": m19["pairwise_match_rate"], "n39": m39["pairwise_match_rate"],
        "abs_diff": d_mean, "threshold": 0.01,
        "outcome": "HIT" if d_mean < 0.01 else "MISS",
    }

    # ---- B2 / B3: the w0 interval ------------------------------------------
    if mix19 is not None and mix39 is not None:
        p19 = mix19["by_theta"][REF_THETA]["w0_profile"]
        p39 = mix39["by_theta"][REF_THETA]["w0_profile"]
        a, b = p19["w0_interval"]
        c, d = p39["w0_interval"]
        intersects = bool(max(a, c) <= min(b, d))
        out["B2_w0_interval_intersects"] = {
            "kind": "within-system comparison",
            "n19_interval": [a, b], "n39_interval": [c, d],
            "intersects": intersects,
            "outcome": "HIT" if intersects else "MISS",
            "consequence_if_MISS": (
                "the fit is partly fitting N, which undercuts every purity "
                "number derived from the mixture at EITHER N (AK5)"),
        }
        out["B3_w0_interval_narrows"] = {
            "kind": "within-system comparison",
            "n19_width": p19["w0_interval_width"],
            "n39_width": p39["w0_interval_width"],
            "grid_step": 1.0 / (p19["n_points"] - 1),
            "outcome": "HIT" if p39["w0_interval_width"] < p19["w0_interval_width"]
                       else "MISS",
            "note": "a tie is a MISS, declared in AK3 B3 before the run",
        }
        t19 = mix19["by_theta"][REF_THETA]["bimodality"]["mass_in_thirds"]
        t39 = mix39["by_theta"][REF_THETA]["bimodality"]["mass_in_thirds"]
        moves = [abs(x - y) for x, y in zip(t19, t39)]
        out["B4_coarse_thirds_move_less_than_0.05"] = {
            "kind": "within-system comparison",
            "n19": t19, "n39": t39, "abs_moves": moves, "threshold": 0.05,
            "outcome": "HIT" if max(moves) < 0.05 else "MISS",
        }
        s39 = mix39["by_theta"][REF_THETA]["bimodality"]["shape"]
        out["C2_shape_verdict_stays_BROAD"] = {
            "kind": "distributional shape",
            "n19": mix19["by_theta"][REF_THETA]["bimodality"]["shape"],
            "n39": s39,
            "outcome": "HIT" if s39 == "BROAD" else "MISS",
        }
    else:
        for key in ("B2_w0_interval_intersects", "B3_w0_interval_narrows",
                    "B4_coarse_thirds_move_less_than_0.05",
                    "C2_shape_verdict_stays_BROAD"):
            out[key] = {"outcome": "NOT_SCORED",
                        "reason": "a mixture document was not supplied"}

    # ---- B5: MNN/one-way ratio curve ---------------------------------------
    curve = n39.get("mnn_oneway_ratio_curve")
    if curve is None:
        out["B5_mnn_oneway_ratio"] = {
            "outcome": "NOT_SCORED",
            "reason": "the N = 39 document carries no mnn_oneway_ratio_curve"}
    else:
        r = curve["ratio_at_reference_theta"]
        in_band = bool(0.995 <= r <= 1.000)
        mono = bool(curve["is_non_decreasing_in_theta"])
        out["B5_mnn_oneway_ratio"] = {
            "kind": "within-system comparison",
            "ratio_at_0.70_n39": r,
            "ratio_at_0.70_n19": (m19["pairwise_match_rate"]
                                  / n19["marginals_by_theta"][REF_THETA]
                                  ["oneway"]["pairwise_match_rate"]),
            "band": [0.995, 1.000], "in_band": in_band,
            "non_decreasing_in_theta": mono,
            "first_decrease_at_theta": curve["first_decrease_at_theta"],
            "outcome": "HIT" if in_band and mono else "MISS",
        }

    # ---- B6 / B7: the AG endpoint group sizes ------------------------------
    g19 = m19.get("ag_endpoint_groups", {})
    g39 = m39.get("ag_endpoint_groups", {})
    if "primary_unstable_frac" not in g39:
        out["B6_primary_unstable_shrinks"] = {
            "outcome": "NOT_SCORED",
            "reason": "ag_endpoint_groups absent from the N = 39 document"}
        out["B7_primary_stable_shrinks"] = dict(
            out["B6_primary_unstable_shrinks"])
    else:
        u19 = g19.get("primary_unstable_frac", m19["frac_k_le_1"])
        u39 = g39["primary_unstable_frac"]
        out["B6_primary_unstable_shrinks"] = {
            "kind": "within-system comparison",
            "n19_k_le_1": u19, "n39_k_le_2": u39,
            "strictly_less": bool(u39 < u19),
            "band": [0.08, 0.17], "in_band": bool(0.08 <= u39 <= 0.17),
            "outcome": "HIT" if (u39 < u19 and 0.08 <= u39 <= 0.17) else "MISS",
        }
        s19 = g19.get("primary_stable_frac", m19["frac_k_ge_nminus1"])
        s39 = g39["primary_stable_frac"]
        out["B7_primary_stable_shrinks"] = {
            "kind": "within-system comparison",
            "n19_k_ge_18": s19, "n39_k_ge_37": s39,
            "strictly_less": bool(s39 < s19),
            "band": [0.03, 0.12], "in_band": bool(0.03 <= s39 <= 0.12),
            "outcome": "HIT" if (s39 < s19 and 0.03 <= s39 <= 0.12) else "MISS",
        }

    # ---- C1: interior mass stays above 0.70 --------------------------------
    out["C1_interior_mass_above_0.70"] = {
        "kind": "distributional shape",
        "n19": m19["frac_interior"], "n39": m39["frac_interior"],
        "threshold": 0.70,
        "outcome": "HIT" if m39["frac_interior"] > 0.70 else "MISS",
    }

    tally: dict[str, int] = {}
    for v in out.values():
        tally[v["outcome"]] = tally.get(v["outcome"], 0) + 1
    return {"predictions": out, "tally": tally}


def side_by_side(n19: dict, n39: dict) -> list[dict]:
    """The N = 19 / N = 39 comparison table across the whole theta grid.

    AK6's substitute for multiplicity control is reporting the WHOLE set rather
    than a selected member, so the table spans every theta and both rules
    instead of the reference theta alone.
    """
    rows = []
    for theta in sorted(n19["marginals_by_theta"]):
        for rule in ("mnn", "oneway"):
            a = n19["marginals_by_theta"][theta][rule]
            b = n39["marginals_by_theta"][theta][rule]
            rows.append({
                "theta": float(theta), "rule": rule,
                "mean_rf_n19": a["pairwise_match_rate"],
                "mean_rf_n39": b["pairwise_match_rate"],
                "frac_k_eq_0_n19": a["frac_r_eq_0"],
                "frac_k_eq_0_n39": b["frac_r_eq_0"],
                "frac_k_eq_N_n19": a["frac_r_eq_1"],
                "frac_k_eq_N_n39": b["frac_r_eq_1"],
                "frac_interior_n19": a["frac_interior"],
                "frac_interior_n39": b["frac_interior"],
                "ag_primary_unstable_n19": a.get(
                    "ag_endpoint_groups", {}).get("primary_unstable_frac"),
                "ag_primary_unstable_n39": b.get(
                    "ag_endpoint_groups", {}).get("primary_unstable_frac"),
                "ag_primary_stable_n19": a.get(
                    "ag_endpoint_groups", {}).get("primary_stable_frac"),
                "ag_primary_stable_n39": b.get(
                    "ag_endpoint_groups", {}).get("primary_stable_frac"),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-m1", type=Path, default=None,
                    help="the canary baseline rf_marginal_m1.json")
    ap.add_argument("--candidate-m1-n19", type=Path, default=None,
                    help="the re-run N = 19 artifact; compared BIT-IDENTICALLY")
    ap.add_argument("--baseline-counts", type=Path, default=None)
    ap.add_argument("--candidate-counts-n19", type=Path, default=None)
    ap.add_argument("--m1-n39", type=Path, default=None)
    ap.add_argument("--mixture-n19", type=Path, default=None)
    ap.add_argument("--mixture-n39", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # PROVENANCE, added in AM11. This artifact had NONE -- a comparison of record
    # with no record of what produced it, found by grepping git_dirty across
    # today's outputs and getting `None` back. An artifact that cannot say which
    # commit wrote it cannot participate in the invariant the read-only deploy
    # key exists to create.
    from src.sae.train import git_dirty, git_sha

    report: dict = {"identity": {"measurement": "rf_extended_n_comparison",
                                "amendments": ["AG", "AK", "AM"],
                                "git_sha": git_sha(),
                                "git_dirty": git_dirty(),
                                "declared_added_json_keys":
                                    list(DECLARED_ADDED_JSON_KEYS),
                                "declared_added_h5_attrs":
                                    list(DECLARED_ADDED_H5_ATTRS)}}
    failures: list[str] = []

    if args.baseline_m1 and args.candidate_m1_n19:
        c = compare_n19(args.baseline_m1, args.candidate_m1_n19)
        report["canary_m1"] = c
        print(f"[cmp] N=19 M1 vs baseline: {c['verdict']} "
              f"({c['n_statistic_disagreements']} STATISTIC disagreements, "
              f"{c['n_provenance_disagreements']} provenance (expected), "
              f"{c['n_added_keys']} added keys, "
              f"{len(c['undeclared_added_keys'])} undeclared)", flush=True)
        for d in c["statistic_disagreements"][:10]:
            print(f"      STAT {d}", flush=True)
        for d in c["provenance_disagreements"]:
            print(f"      prov {d['path']}: {d.get('baseline')} -> "
                  f"{d.get('candidate')}", flush=True)
        if c["verdict"] != "REPRODUCES":
            failures.append("canary_m1")

    if args.baseline_counts and args.candidate_counts_n19:
        c = compare_counts_h5(args.baseline_counts, args.candidate_counts_n19)
        report["canary_counts"] = c
        print(f"[cmp] N=19 counts vs baseline: {c['verdict']} "
              f"(match_count bit-identical: "
              f"{c['match_count_bit_identical']})", flush=True)
        if c["verdict"] != "REPRODUCES":
            failures.append("canary_counts")

    if args.candidate_m1_n19 and args.m1_n39:
        n19 = json.loads(args.candidate_m1_n19.read_text())
        n39 = json.loads(args.m1_n39.read_text())
        mix19 = (json.loads(args.mixture_n19.read_text())
                 if args.mixture_n19 else None)
        mix39 = (json.loads(args.mixture_n39.read_text())
                 if args.mixture_n39 else None)
        sc = score_predictions(n19, n39, mix19, mix39)
        report["ak_scorecard"] = sc
        report["side_by_side"] = side_by_side(n19, n39)
        print("[cmp] AK scorecard:", json.dumps(sc["tally"]), flush=True)
        for name, v in sc["predictions"].items():
            print(f"      {v['outcome']:<11} {name}", flush=True)
        # A correctness ASSERTION failing is a defect and must gate; a
        # prediction MISSING is a legitimate outcome and must not.
        for name, v in sc["predictions"].items():
            if v.get("kind", "").startswith("correctness") and v["outcome"] != "PASS":
                failures.append(f"assertion:{name}")

    report["failures"] = failures
    report["verdict"] = "OK" if not failures else "FAILED"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[cmp] wrote {args.out} -- verdict {report['verdict']}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
