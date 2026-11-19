"""
Seed-axis alignment. `glob("seed_*")` (used by every sbatch script that
loops over checkpoints) sorts LEXICOGRAPHICALLY, not numerically -- seed_0,
seed_1, seed_10, seed_11, ..., seed_19, seed_2, seed_3, .... Every h5 output
in this project with a seed axis already stores its own `seeds` label array
for exactly this reason (feature_ap.py, null_pilot.py). The gap this module
closes: nothing enforces that those labels actually get CHECKED before two
seed-indexed arrays from different files are combined by position.

This is this project's specific failure signature, not a hypothetical: it
already produced one silent-until-caught bug this session (the null-tail
investigation script hardcoded "seed_7" for array index 7, which is
actually "seed_15" -- caught only because that script happened to include
its own sanity check against a stored reference value). P2 will standardize
a real g_f array against null_pilot_g.h5's null distribution, seed by seed;
if the two arrays' seed orders differ, every feature is standardized
against the WRONG seed's null, and the result looks entirely plausible
(every seed's null is statistically similar) while being wrong.
"""

from __future__ import annotations

import numpy as np


def assert_seed_axes_match(names_a: list[str], names_b: list[str],
                           label_a: str = "A", label_b: str = "B") -> None:
    """Raise if two seed-labelled arrays are not in the SAME order. Call this
    before combining any two seed-indexed arrays positionally (e.g. a
    per-feature real g_f array against a per-feature null array)."""
    la, lb = list(names_a), list(names_b)
    if la != lb:
        raise ValueError(
            f"seed axis mismatch between {label_a} and {label_b}: "
            f"{label_a}={la} vs {label_b}={lb}. Combining these arrays "
            "positionally would silently standardize each feature against "
            "the WRONG seed's data."
        )


def align_to_reference(names: list[str], array: np.ndarray,
                       reference_names: list[str], axis: int = 0) -> np.ndarray:
    """Reorder `array` along `axis` (currently ordered per `names`) so its
    seed order matches `reference_names` exactly. Raises if the seed SETS
    differ, not just the order -- a genuinely missing seed must never be
    silently dropped by an index lookup that skips it."""
    if set(names) != set(reference_names):
        missing_in_names = set(reference_names) - set(names)
        missing_in_ref = set(names) - set(reference_names)
        raise ValueError(
            f"seed sets differ, not just order: missing from source "
            f"{missing_in_names}, missing from reference {missing_in_ref}"
        )
    idx = [names.index(n) for n in reference_names]
    return np.take(array, idx, axis=axis)


# ---------------------------------------------------------------------------
# AMENDMENT AS: seed ORDER, not just seed alignment.
#
# AR9 found this module's whole subject matter shipped as a live defect --
# `feature_ap`, `null_pilot` and both `gtilde` artifacts are in LEXICOGRAPHIC
# seed order (`glob("seed_*")` sorts that way), `rf_counts` and `d_fs` are
# NUMERIC, only rows 0 and 1 coincide, and 18 of 20 rows disagree. The cause of
# the miss was not that the hazard was unknown: it is documented in
# CODEBASE 7.3, this module was written for it, and NO CONSUMER CALLED IT.
#
# So the machinery lives HERE now rather than inside one analysis script,
# and every consumer that joins two seed-indexed arrays imports it.
# ---------------------------------------------------------------------------

def normalise_seed_labels(labels) -> list[str]:
    """`seed_7`, `7`, `b'seed_7'`, `numpy.int32(7)` -> `"7"`.

    Three different writers use three different spellings for the same axis,
    which is how the collision stayed invisible: two lists that name the same
    seeds cannot be compared until they are spelled the same way.
    """
    out = []
    for x in labels:
        if isinstance(x, bytes):
            x = x.decode()
        s = str(x)
        out.append(str(int(s.rsplit("_", 1)[-1])))
    return out


def stored_seed_labels(path, dataset: str = "seeds",
                       attr: str = "seed_labels") -> list[str] | None:
    """The artifact's OWN labels, or `None` if it does not carry any.

    `None` means ABSENT and must be handled as absent -- never as agreement
    with whatever the caller expected (gotcha 15).
    """
    import json as _json

    import h5py

    with h5py.File(path, "r") as hf:
        if dataset in hf:
            return normalise_seed_labels(hf[dataset][:])
        if attr in hf.attrs:
            v = hf.attrs[attr]
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, str):
                v = _json.loads(v)
            return normalise_seed_labels(v)
    return None


def resolve_unlabelled_seed_order(gtilde, null_reductions, n_sub: int = 800,
                                  rng_seed: int = 0,
                                  min_margin: float = 3.0) -> dict:
    """WHICH SEED IS ROW i OF AN ARTIFACT THAT CARRIES NO LABELS? From DATA.

    `standardize_gain.py` and `degenerate_flag.py` historically wrote no
    `seeds` dataset, so their order cannot be read off the file at all. It is
    recovered here: `gtilde_null[i]` is a per-stratum monotone transform of
    `null_g_b[j]`, so the rank correlation across (shift, feature) peaks
    sharply at the matching seed. Measured on the real artifacts: **0.45 at
    the match against 0.018 at the next best**, a 25x margin, agreeing exactly
    with an independent handle (the smooth arm's stored `feature_firing_bin`
    matches `bin_index(fire_rate[j])` at agreement 1.000000 against 0.37).

    IT REFUSES RATHER THAN GUESSES. The permutation must be a BIJECTION and
    every row's margin must clear `min_margin`; either failure raises. A
    resolver that always returns something is a resolver that cannot fail
    (gotcha 25).
    """
    import h5py
    from scipy.stats import rankdata

    with h5py.File(null_reductions, "r") as hf:
        labels = normalise_seed_labels(hf["seeds"][:])
        n_seeds, n_shifts, n_feats = hf["null_g_b"].shape
        rng = np.random.default_rng(rng_seed)
        sub = np.sort(rng.choice(n_feats, min(n_sub, n_feats), replace=False))
        A = hf["null_g_b"][:, :, sub].reshape(n_seeds, -1).astype(np.float64)
    with h5py.File(gtilde, "r") as hf:
        B = hf["gtilde_null"][:, :, sub].reshape(n_seeds, -1).astype(np.float64)
    if B.shape != A.shape:
        raise SystemExit(f"FATAL: gtilde_null {B.shape} vs null_g_b {A.shape}")

    def _z(x):
        r = rankdata(x, axis=1)
        r = r - r.mean(axis=1, keepdims=True)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-300)

    C = np.abs(_z(B) @ _z(A).T)                    # (gtilde row, null row)
    perm, rows = [], []
    for i in range(n_seeds):
        order = np.argsort(C[i])[::-1]
        j, best, nxt = int(order[0]), float(C[i, order[0]]), float(C[i, order[1]])
        perm.append(j)
        rows.append({"gtilde_row": i, "null_row": j, "seed": labels[j],
                     "rho": best, "next_best_rho": nxt,
                     "margin": best / max(nxt, 1e-300)})
    problems = []
    if sorted(perm) != list(range(n_seeds)):
        problems.append(f"not a bijection: {perm}")
    weak = [r for r in rows if r["margin"] < min_margin]
    if weak:
        problems.append(f"{len(weak)} rows resolve with margin < {min_margin}")
    if problems:
        raise SystemExit(
            f"FATAL: cannot resolve {getattr(gtilde, 'name', gtilde)}'s seed "
            "order from the data.\n  * " + "\n  * ".join(problems) +
            "\nRefusing to assume an order for an unlabelled seed axis -- a "
            "wrong one mis-joins 18 of 20 rows and produces entirely plausible "
            "numbers (gotcha 18).")
    return {"labels": [r["seed"] for r in rows], "rows": rows,
            "null_labels": labels, "min_margin_required": min_margin,
            "min_margin_observed": float(min(r["margin"] for r in rows)),
            "method": ("rank correlation of gtilde_null against null_g_b over "
                       f"{len(sub)} features x {n_shifts} shifts, rng {rng_seed}"),
            "verdict": "RESOLVED"}


def seed_order(gtilde, null_reductions, **kw) -> dict:
    """The artifact's seed order, PREFERRING its own labels and CHECKING them.

    Three cases, and the third is the point:

      * labels stored, resolver available -> **both are computed and compared**.
        A disagreement RAISES. This is what keeps the AS migration honest: once
        the labels are on disk a consumer would normally stop looking, and then
        a wrong label would be undetectable. Here it is not.
      * labels stored, no `gtilde_null` to resolve against -> labels used, and
        the report says the cross-check did not run.
      * no labels -> resolved from data, as before AS.
    """
    stored = stored_seed_labels(gtilde)
    try:
        res = resolve_unlabelled_seed_order(gtilde, null_reductions, **kw)
    except (KeyError, OSError):
        res = None
    if stored is None:
        if res is None:
            raise SystemExit(
                f"FATAL: {getattr(gtilde, 'name', gtilde)} carries no seed "
                "labels AND cannot be resolved from data. Refusing to assume "
                "an order (gotcha 15: absent is not matching).")
        return {**res, "source": "RESOLVED_FROM_DATA", "stored_labels": None,
                "cross_check": "NOT_APPLICABLE_NO_STORED_LABELS"}
    if res is None:
        return {"labels": stored, "source": "STORED_LABELS",
                "stored_labels": stored,
                "cross_check": "NOT_RUN_NO_GTILDE_NULL"}
    if list(res["labels"]) != list(stored):
        raise SystemExit(
            f"FATAL: {getattr(gtilde, 'name', gtilde)}'s STORED seed labels "
            f"{stored} disagree with the order recovered from its own data "
            f"{res['labels']}. One of them is wrong and this job cannot tell "
            "which; refusing to proceed.")
    return {**res, "source": "STORED_LABELS_CONFIRMED_BY_RESOLVER",
            "stored_labels": stored, "cross_check": "AGREE"}
