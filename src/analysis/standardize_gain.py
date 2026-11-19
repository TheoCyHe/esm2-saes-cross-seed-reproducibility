"""K2's THREE-STAGE STANDARDIZATION: `g(f,c)` -> `z(f,c)` -> `M_f` -> `g~_f`.

PREREGISTRATION amendment K2, on amendment O2's four pooling axes
(prevalence x firing rate x surviving protein count x cluster count).

WHY THREE STAGES AND NOT ONE. Amendments F1.4 and G5 read as alternatives --
F says standardize `g_f` against *the feature's own null*, G5 says pool the
null on *concept properties* -- and as alternatives they are incompatible.
K2 resolves it: they are consecutive stages of one pipeline, correcting two
different things, and neither substitutes for the other.

  Stage 1 (INNER)  z(f,c) = (g(f,c) - mu_s) / sigma_s,  s = the K1 stratum.
        Equalizes CONCEPT DIFFICULTY, so a rare family competes on the same
        scale as a broad one. Moments are per stratum, pooled over features,
        concepts and shifts within the cell, because 20 shifts cannot
        estimate a per-pair sigma -- which is the entire reason the pooling
        axes are concept properties.

  Stage 2 (MAX)    M_f = max over the argmax pool of z(f,c).

  Stage 3 (OUTER)  g~_f = M_f standardized against the null distribution of
        M_f itself, obtained by computing z for all concepts within each
        shift and taking the max: 20 shifts -> 20 null draws of M.
        Corrects LIBRARY-SIZE INFLATION, which stage 1 does nothing about --
        a max over 703 standardized values still has a positive expectation
        that grows with concept count and with inter-concept correlation.

READ ALL 712, MASK TO 703 AT MAX TIME. The 9 pooled parents are kept in the
matrix so the I4 diagnostic arm has their `z` values; they simply cannot win
the max. Subsetting the matrix on load would discard them (K2 says so
explicitly).

LEAVE-ONE-OUT ON THE SHIFT AXIS, FOR NULL DRAWS ONLY. A null draw must not be
standardized against moments computed from a set containing itself. The real
`g` is not in the null and cannot contaminate its own moments, so LOO is not
applied to it. K2 states this at stage 3; the same structure exists at stage 1
(whose moments also pool over shifts), so it is applied at BOTH and the
difference between LOO and non-LOO is reported rather than assumed
negligible -- at stage 1 the pooled cell holds thousands of values so removing
one shift of twenty is a small correction, while at stage 3 the moments come
from exactly 20 draws and it is not.

THE AXIS SPEC IS A PARAMETER, NOT A HARDCODED SCHEME. O2 committed four axes,
but the cluster-gate question (O1) is explicitly conditional on a check this
module performs (O5), so the scheme must be changeable by configuration
rather than by editing the arithmetic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import (argmax_pool_mask, device_class,
                                            write_identity)
from src.analysis.classification_2x2 import (SeedOrderUnrecoverable,
                                             recover_seed_position)
from src.analysis.seed_axis import stored_seed_labels
from src.analysis.null_pooling import (
    CLUSTER_COUNT_EDGES,
    FIRING_RATE_EDGES,
    MIN_FEATURES_PER_STRATUM,
    PREVALENCE_EDGES,
    PROTEIN_COUNT_EDGES,
    bin_index,
    collapse_firing_bins,
)
from src.sae.train import git_dirty, git_sha

# Concept-side axes available to the spec, each with its pre-declared edges.
# Edges are imported, never restated: K1 and N2 fixed them BEFORE any of these
# counts were computed, and a second copy is a second thing to drift.
CONCEPT_AXES: dict[str, list] = {
    "prevalence": PREVALENCE_EDGES,
    "protein_count": PROTEIN_COUNT_EDGES,
    "cluster_count": CLUSTER_COUNT_EDGES,
}

# Amendment O2's committed scheme.
DEFAULT_CONCEPT_AXES = ("prevalence", "protein_count", "cluster_count")


@dataclass(frozen=True)
class AxisSpec:
    """The pooling scheme. `concept_axes` names a subset of `CONCEPT_AXES`;
    the firing-rate axis is always present and always binned PER SEED (L1)."""

    concept_axes: tuple[str, ...] = DEFAULT_CONCEPT_AXES
    firing_rate: bool = True

    def __post_init__(self) -> None:
        bad = [a for a in self.concept_axes if a not in CONCEPT_AXES]
        if bad:
            raise ValueError(f"unknown concept axes {bad}; "
                             f"available: {sorted(CONCEPT_AXES)}")
        if not self.firing_rate:
            # K1's remedy list forbids collapsing the firing axis to fix a
            # count, and dropping it entirely is the limiting case of that.
            raise ValueError(
                "the firing-rate axis cannot be dropped. Firing rate is the "
                "confound the entire design exists to control (K1's remedy "
                "list: 'never firing rate'); removing it from the pooling "
                "scheme reintroduces the confound into mu_s and sigma_s.")

    @property
    def name(self) -> str:
        return " x ".join(self.concept_axes) + " x firing_rate"


def concept_cell_index(values: dict[str, np.ndarray],
                       spec: AxisSpec) -> tuple[np.ndarray, int]:
    """Flatten the concept-side axes into a single cell index per concept.

    Returns (index, n_cells). A concept whose value falls below an axis's
    first edge gets -1 from `bin_index`; that is propagated as -1 (OUT OF
    SCOPE) rather than clipped into bin 0, because a concept below the >=5
    protein gate or with an absent cluster count is not a member of the
    lowest stratum -- it is a concept this scheme cannot place, and silently
    placing it is how a sentinel becomes data.
    """
    idx = np.zeros(len(next(iter(values.values()))), dtype=np.int64)
    n_cells = 1
    out_of_scope = np.zeros_like(idx, dtype=bool)
    for axis in spec.concept_axes:
        edges = CONCEPT_AXES[axis]
        b = bin_index(values[axis], edges)
        out_of_scope |= (b < 0)
        idx = idx * (len(edges) - 1) + np.maximum(b, 0)
        n_cells *= len(edges) - 1
    idx = np.where(out_of_scope, -1, idx)
    return idx, n_cells


def merge_lowest_firing_bin(edges: list) -> list:
    """Amendment CT §CT14: merge the LOWEST firing-rate bin into the next one,
    after K1's collapse -- the edge list loses its second edge and nothing else
    changes. Declared before running (on the shared set's collapsed edges
    [0, 1e-4, 3e-4, ...] it gives [0, 3e-4, ...]). Refuses when there is no
    second bin to merge into rather than returning the list unchanged: an
    option that can silently do nothing is not a verified operation (gotcha 13,
    extension 4)."""
    e = [float(x) for x in edges]
    if len(e) < 3:
        raise ValueError(f"cannot merge the lowest firing bin of {e}: "
                         "fewer than two bins")
    return [e[0]] + e[2:]


def merge_top_firing_bin(edges: list) -> list:
    """Amendment CT §CT29-D2 item 2: merge the TOP firing-rate bin into the
    one below it, after K1's collapse -- §CT14's merge mirrored. The edge list
    loses its second-to-last edge and nothing else changes (arm (iii)'s
    [..., 0.01, 0.03, 1.000000001] gives [..., 0.01, 1.000000001]). Refuses
    when there is no second bin, rather than returning the list unchanged
    (gotcha 13, extension 4)."""
    e = [float(x) for x in edges]
    if len(e) < 3:
        raise ValueError(f"cannot merge the top firing bin of {e}: "
                         "fewer than two bins")
    return e[:-2] + [e[-1]]


def pooled_moments(slab: np.ndarray, cell: np.ndarray, n_cells: int,
                   fbin: np.ndarray, n_fbins: int):
    """Per-shift sums and sums-of-squares per (concept cell, firing bin).

    `slab` is (shifts, concepts, features) for ONE seed. Returned arrays are
    (shifts, n_cells, n_fbins), which is what makes leave-one-out on the
    shift axis a subtraction rather than a recomputation.

    The counts are per-shift too, so a cell that is empty in one shift cannot
    silently inflate another's denominator.
    """
    n_r = slab.shape[0]
    s1 = np.zeros((n_r, n_cells, n_fbins), dtype=np.float64)
    s2 = np.zeros((n_r, n_cells, n_fbins), dtype=np.float64)
    cnt = np.zeros((n_cells, n_fbins), dtype=np.int64)

    cmasks = [np.flatnonzero(cell == cc) for cc in range(n_cells)]
    fmasks = [np.flatnonzero(fbin == fb) for fb in range(n_fbins)]
    for cc, ci in enumerate(cmasks):
        if ci.size == 0:
            continue
        for fb, fi in enumerate(fmasks):
            if fi.size == 0:
                continue
            cnt[cc, fb] = ci.size * fi.size
            sub = slab[:, ci][:, :, fi]              # (shifts, |ci|, |fi|)
            s1[:, cc, fb] = sub.sum(axis=(1, 2))
            s2[:, cc, fb] = np.square(sub, dtype=np.float64).sum(axis=(1, 2))
    return s1, s2, cnt


def moments_from_sums(s1: np.ndarray, s2: np.ndarray, cnt: np.ndarray,
                      drop_shift: int | None = None):
    """(mu, sigma) per (cell, firing bin), optionally leaving one shift out.

    `sigma` is NaN wherever it cannot be estimated -- fewer than two
    independent shifts, or a zero/undefined spread. NaN propagates into `z`
    and is reported, never silently replaced by a small number: dividing by a
    fabricated sigma is how a stratum with no information produces a
    confident-looking extreme value.
    """
    n_r = s1.shape[0]
    if drop_shift is None:
        t1, t2, n_used = s1.sum(0), s2.sum(0), n_r
    else:
        t1 = s1.sum(0) - s1[drop_shift]
        t2 = s2.sum(0) - s2[drop_shift]
        n_used = n_r - 1
    n = cnt.astype(np.float64) * n_used
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(n > 0, t1 / n, np.nan)
        var = np.where(n > 1, t2 / n - mu ** 2, np.nan)
        var = np.where(var > 0, var, np.nan)
        sigma = np.sqrt(var)
    return mu, sigma


def _grid(table: np.ndarray, cell: np.ndarray, fbin: np.ndarray) -> np.ndarray:
    """Broadcast a (cells, fbins) table to (concepts, features)."""
    return table[np.maximum(cell, 0)[:, None], fbin[None, :]]


def assert_seed_order_agrees(reference_name: str, reference: list,
                             other_name: str, other) -> None:
    """Refuse unless `other` equals `reference` as an ORDERED list of seed
    labels (amendment CQ, A2).

    This module indexes the AP matrix, the raw null and `fire_rate` by
    POSITION on their seed axes, so every one of them must be in the reference
    order -- the null reductions' `seeds`, which this module copies onto its
    output. `other is None` means the artifact carries no labels, and that is
    a refusal too: an unlabelled axis is not agreement (gotcha 15; the AS
    convention this module already applies to the null reductions). The
    message prints BOTH orders.
    """
    ref = [str(x) for x in reference]
    if other is None:
        raise SystemExit(
            f"FATAL: {other_name} carries no seed labels, so its seed order "
            f"cannot be checked against {reference_name} "
            f"({ref}). Refusing to index an unlabelled seed axis positionally "
            f"(gotcha 29).")
    oth = [str(x) for x in other]
    if oth != ref:
        n_bad = (sum(a != b for a, b in zip(ref, oth))
                 + abs(len(ref) - len(oth)))
        raise SystemExit(
            f"FATAL: seed order of {other_name} differs from "
            f"{reference_name}:\n  {reference_name}: {ref}\n"
            f"  {other_name}: {oth}\n  A positional join would mis-pair "
            f"{n_bad} of {max(len(ref), len(oth))} rows (gotcha 29). Refusing.")


def assert_raw_slab_is_reference_row(slab: np.ndarray, null_g: np.ndarray,
                                     s: int, labels: list, arm: str,
                                     stored_dtype=None) -> None:
    """Refuse unless raw-null slab `s` IS the null reductions' row `s`
    (amendment CQ, A2-iii).

    The raw null carries no seed labels, so its order is RECOVERED FROM THE
    DATA rather than read (gotcha 29's own rule, by the method amendment BS
    part C3 recorded): `null_g_<arm>` in the reductions is the maximum over
    CONCEPTS of this very array, so the slab's max over its concept axis must
    equal exactly one row of `null_g_<arm>` -- exact equality, no tolerance --
    and that row must be `s`. `slab` is (n_shifts, n_pool, n_features);
    `null_g` is (n_seeds, n_shifts, n_features).
    """
    # NaN-tolerant, as classification_2x2's recovery is: a degenerate Null-B
    # draw is NaN (BH2) and a NaN under a plain max would break the exact
    # equality. On a raw null with no NaN -- every null this module consumed
    # before amendment CT -- nanmax IS max, so their recovery is unchanged.
    with np.errstate(invalid="ignore"):
        m = np.nanmax(np.asarray(slab, dtype=np.float64), axis=1)
    g = np.asarray(null_g, dtype=np.float64)
    # IN THE RAW ARRAY'S STORED PRECISION (amendment CT §CT5.1-R). The raw
    # array holds float32; on the sparse GPU path `null_g` is a max over
    # float32 values, so the two agree in float64. On the segmented CPU path
    # (masked negatives) the gain is accumulated in float64 and `null_g` keeps
    # the float64 max while the raw array keeps its float32 cast, so they
    # differ in the last bits in nearly every cell. Rounding to nearest is
    # monotone, so max(f32(x)) == f32(max(x)) EXACTLY: comparing both sides
    # in the stored dtype is still an identity, not a tolerance. For a null
    # whose `null_g` is already float32-representable (every pooled null of
    # record) the cast is lossless and the verdict cannot change.
    if stored_dtype is not None and np.dtype(stored_dtype) != np.float64:
        m, g = m.astype(stored_dtype), g.astype(stored_dtype)
    try:
        pos = recover_seed_position(m, g)
    except SeedOrderUnrecoverable as e:
        raise SystemExit(
            f"FATAL: raw null slab {s} (gain_{arm}) matches no row of the "
            f"reductions' null_g_{arm} exactly, or more than one: {e}\n"
            f"  reductions seed labels: {[str(x) for x in labels]}\n"
            f"  The raw null's seed order cannot be established, so it is not "
            f"assumed (gotcha 29). Refusing.") from e
    if pos != s:
        raise SystemExit(
            f"FATAL: raw null slab {s} (gain_{arm}) IS the reductions' row "
            f"{pos} (seed {labels[pos]!s}), not row {s} (seed "
            f"{labels[s]!s}). The raw null and the reductions are in "
            f"different seed orders; indexing them positionally would pair "
            f"slab {s} with seed {labels[s]!s}'s AP and firing rate. Refusing.")


def masked_argmax(z: np.ndarray, pool_mask: np.ndarray):
    """Max over POOL concepts only, with lowest-index tie-breaking.

    `z` is (concepts, features). Non-pool concepts are set to -inf rather
    than removed, so the returned argmax indexes the FULL concept axis and
    stays comparable with the 712-concept matrix the diagnostic arm reads.

    Tie-breaking is `np.argmax`'s lowest-index rule, asserted here rather
    than assumed: with ties broken differently by two code paths, the same
    `g~` would attribute to different concepts and D5's breadth accounting
    would silently disagree with the classification's.
    """
    masked = np.where(pool_mask[:, None], z, -np.inf)
    am = np.argmax(masked, axis=0)               # lowest index wins ties
    return masked[am, np.arange(masked.shape[1])], am


def _inner(g, mu, sig, mode):
    """Stage 1 in three variants, so the CENTRING and the SCALING can be
    separated (amendment S measurement 1).

    Q3 measured that scaling alone reorders essentially nothing
    (Spearman vs raw G = 0.9961) while centring alone does nearly all of it
    (0.2582, against 0.2738 for the full transform). So stage 1 is a
    centring operation with a cosmetic scale attached, and the real question
    is not standardize-or-not but whether to subtract a per-stratum
    empirical mean. `mu` spans 21.7x across cells at one firing bin, which
    is chance-level structure the analytic (AP - p_c)/(1 - p_c) cannot see:
    p_c knows the concept's prevalence and nothing about firing rate or how
    clustered the feature's activations are.
    """
    if mode == "none":
        return g
    if mode == "centre":
        return g - mu
    if mode == "ratio":
        # GSEA's normalization (Subramanian et al. 2005 PNAS): divide by the
        # null MEAN, not by its SD, to factor out intrinsic size dependence.
        # Adopted for the regression predictor in amendment U1.
        #
        # DEVIATION FROM THE PRECEDENT, recorded rather than silent: GSEA
        # normalizes each gene set by its OWN null mean. We pool over the
        # stratum, because 20 draws per (feature, concept) cannot estimate a
        # per-pair mean.
        #
        # NO SIGN PARTITIONING. GSEA partitions by sign because it reports an
        # ES per set. We take a MAXIMUM, and the pool max is positive for all
        # 10,240 features in every seed -- measured, U1 decision 2 -- so the
        # 50.4% of negative g values never participate in `M_f`. They are
        # retained here (not floored) because the per-(f,c) values feed the
        # I4 diagnostic arm.
        return g / mu
    return (g - mu) / sig


def standardize_seed(real_gain: np.ndarray, null_slab: np.ndarray,
                     cell: np.ndarray, n_cells: int, fbin: np.ndarray,
                     n_fbins: int, pool_mask: np.ndarray, loo_inner: bool,
                     inner_mode: str = "full",
                     mu_grid_real: np.ndarray | None = None,
                     mu_grid_pool: np.ndarray | None = None):
    """All three stages for one seed.

    `real_gain` is (concepts_all, features) over the FULL library (712);
    `null_slab` is (shifts, concepts_pool, features). The two concept axes
    differ by exactly the pooled parents, so the pool concepts' positions in
    the full axis are passed via `pool_mask`.
    """
    n_r = null_slab.shape[0]
    pool_idx = np.flatnonzero(pool_mask)
    cell_pool, fb = cell[pool_idx], fbin

    s1, s2, cnt = pooled_moments(null_slab, cell_pool, n_cells, fb, n_fbins)

    # ---- stage 1 on the REAL gain: full-sample moments, no LOO (the real
    # value is not in the null, so it cannot contaminate its own moments).
    mu_full, sig_full = moments_from_sums(s1, s2, cnt)
    if inner_mode == "ratio":
        # U1 decision 4, re-checked PER SEED rather than taken from the one
        # seed it was measured on. Pooling makes a near-zero mu_s unlikely
        # (measured min 3.358e-07 over 240 strata) but "unlikely on seed 0"
        # is not a guarantee, and dividing by a near-zero denominator is the
        # failure mode this whole line of work exists to avoid.
        finite = mu_full[np.isfinite(mu_full)]
        if finite.size and finite.min() <= 0:
            raise SystemExit(
                f"FATAL: {int((finite <= 0).sum())} strata have mu_s <= 0; "
                "ratio normalization is undefined there. Refusing rather "
                "than flooring silently.")
    # M4 / amendment U3: when a SMOOTH mu is supplied it replaces the
    # hard-stratum broadcast entirely. The stratum moments are still computed
    # above because `sigma` is unaffected and the `mu_s <= 0` guard above is
    # still worth running on them.
    mu_real_grid = (mu_grid_real if mu_grid_real is not None
                    else _grid(mu_full, cell, fb))
    z_real = _inner(real_gain, mu_real_grid,
                    _grid(sig_full, cell, fb), inner_mode)

    # ---- stage 2 on the real value
    M, argc = masked_argmax(z_real, pool_mask)

    # ---- stages 1+2 on each null draw
    M_null = np.empty((n_r, real_gain.shape[1]), dtype=np.float64)
    for r in range(n_r):
        mu, sig = moments_from_sums(s1, s2, cnt,
                                    drop_shift=r if loo_inner else None)
        # A SMOOTH mu is fitted once per seed on all shifts, so there is no
        # leave-one-out version of it. Stated rather than hidden: the null
        # draws contributed to their own normaliser here, which makes M_null
        # slightly optimistic. It does NOT affect the cross-seed CV
        # comparison, which is computed on the REAL M values only -- so the
        # arm this deviation appears in is not the arm the kill criterion
        # reads.
        mu_null_grid = (mu_grid_pool if mu_grid_pool is not None
                        else _grid(mu, cell_pool, fb))
        zr = _inner(null_slab[r], mu_null_grid,
                    _grid(sig, cell_pool, fb), inner_mode)
        M_null[r] = np.nanmax(np.where(np.isfinite(zr), zr, -np.inf), axis=0)

    # ---- stage 3: standardize M against the null distribution of M.
    # LOO here is NOT a refinement -- the moments come from exactly 20 draws,
    # so a draw standardized against a set containing itself is pulled toward
    # zero by 1/20 of its own value plus its own contribution to sigma.
    m_mean, m_sd = M_null.mean(axis=0), M_null.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gtilde = (M - m_mean) / m_sd
        tot = M_null.sum(axis=0)
        tot2 = np.square(M_null).sum(axis=0)
        loo_mean = (tot - M_null) / (n_r - 1)
        loo_var = ((tot2 - np.square(M_null)) / (n_r - 1)
                   - np.square(loo_mean)) * (n_r - 1) / (n_r - 2)
        gtilde_null = (M_null - loo_mean) / np.sqrt(np.where(loo_var > 0,
                                                             loo_var, np.nan))
    return dict(z_real=z_real, M=M, argmax_concept=argc, M_null=M_null,
                gtilde=gtilde, gtilde_null=gtilde_null,
                mu=mu_full, sigma=sig_full, cell_counts=cnt)


def load_axis_values(labels_h5: Path, labels_json: Path, concepts: list[str],
                     n_rows: int) -> dict[str, np.ndarray]:
    """Per-concept values for every concept-side axis, over the FULL library.

    `null_pooling.load_concepts` returns pool concepts only; stage 1 needs
    the pooled parents too, because the I4 diagnostic arm reads their `z`.
    """
    diag = json.loads(labels_json.read_text())["retention_floor_diagnostics"]
    missing = [c for c in concepts if c not in diag]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} concepts absent from the "
                         f"manifest diagnostics, e.g. {missing[:3]}")
    vals = {
        "prevalence": np.array([diag[c]["n_residues"] / n_rows
                                for c in concepts], dtype=float),
        "protein_count": np.array([diag[c]["n_proteins"] for c in concepts],
                                  dtype=float),
        "cluster_count": np.array([diag[c].get("n_clusters", -1)
                                   for c in concepts], dtype=float),
    }
    # A -1 means the store was built WITHOUT --clusters. Absent is not zero,
    # and stratifying on a sentinel is exactly the class of defect the
    # identity work in this session exists to stop.
    if (vals["cluster_count"] < 0).any():
        raise SystemExit(
            "FATAL: the label manifest carries no `n_clusters` for "
            f"{int((vals['cluster_count'] < 0).sum())} concepts. The store was "
            "built without --clusters, so the O2 cluster axis cannot be "
            "formed. Refusing rather than binning a -1 sentinel as a count.")
    return vals


def stratum_axis_values(concept_rows: dict[str, np.ndarray],
                        row_protein: np.ndarray, protein_cluster: list,
                        concepts: list[str], prevalence: np.ndarray
                        ) -> dict[str, np.ndarray]:
    """Amendment CT, §CT12.1: a stratum's OWN concept-axis values on its row
    set. `concept_rows[c]` are the stratum's positive rows for `c` already
    restricted to the row set, `row_protein` maps a cache row to its protein
    index, `protein_cluster[i]` is protein i's 30%-identity cluster.
    `prevalence` is the stratum AP group's own (what G is computed against).
    """
    prot, clu = [], []
    for c in concepts:
        r = concept_rows.get(c, np.zeros(0, dtype=np.int64))
        pr = np.unique(row_protein[r]) if r.size else np.zeros(0, dtype=np.int64)
        prot.append(pr.size)
        clu.append(len({protein_cluster[i] for i in pr}))
    return {"prevalence": np.asarray(prevalence, dtype=float),
            "protein_count": np.asarray(prot, dtype=float),
            "cluster_count": np.asarray(clu, dtype=float)}


def _stratum_counts(args, ap_attrs: dict, concepts: list[str],
                    strata: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per stratum, on the AP's own scored row set: each concept's protein
    count, cluster count and positive-row count. Loads the row set, the cache
    index and the cluster map once for every stratum named."""
    from src.analysis.feature_ap import load_labels, load_scored_row_set
    from src.annotations.build_residue_labels import header_to_acc, load_cluster_map
    if args.eval_cache is None or args.clusters_tsv is None:
        raise SystemExit("FATAL: --axis-values stratum and --admission-strata "
                         "need --eval-cache and --clusters-tsv")
    row_set = str(ap_attrs["scored_row_set"])
    sub = load_scored_row_set(args.labels, row_set,
                              getattr(args, "row_set_file", None))
    with h5py.File(args.eval_cache, "r") as hf:
        raw = hf["index_json"][()]
    index = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    n_cache = sum(e["n_rows"] for e in index)
    in_set = np.zeros(n_cache, dtype=bool)
    if sub is None:
        in_set[:] = True
    else:
        in_set[sub] = True
    row_protein = np.full(n_cache, -1, dtype=np.int64)
    for i, e in enumerate(index):
        row_protein[e["start"]:e["start"] + e["n_rows"]] = i
    cluster_of = load_cluster_map(args.clusters_tsv)
    protein_cluster = [cluster_of.get(header_to_acc(e["accession"])) for e in index]
    out = {}
    for t in strata:
        lab, _ = load_labels(args.labels, t)
        rows = {}
        for c in concepts:
            r = lab.get(c, np.zeros(0, dtype=np.int64)).astype(np.int64)
            r = r[r < n_cache]
            rows[c] = r[in_set[r]]
        v = stratum_axis_values(rows, row_protein, protein_cluster, concepts,
                                np.zeros(len(concepts)))
        out[t] = {"protein_count": v["protein_count"],
                  "cluster_count": v["cluster_count"],
                  "n_positive_rows": np.array([rows[c].size for c in concepts],
                                              dtype=np.int64)}
        print(f"[stratum counts] {t} on row set {row_set}: protein_count min "
              f"{int(v['protein_count'].min())}, cluster_count min "
              f"{int(v['cluster_count'].min())}", flush=True)
    return out


def admission_mask(protein_counts: dict[str, np.ndarray],
                   min_proteins: int) -> np.ndarray:
    """Amendment CT §CT5.1: a concept is ADMITTED when it carries at least
    `min_proteins` proteins in EVERY stratum named, so both strata are scored on
    one concept set (the pools re-matched). `min_proteins` is the label store's
    own pre-declared gate, not a new number."""
    if not protein_counts:
        raise ValueError("admission needs at least one stratum")
    keep = None
    for pc in protein_counts.values():
        k = np.asarray(pc) >= min_proteins
        keep = k if keep is None else (keep & k)
    return keep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null-raw", type=Path, required=True)
    p.add_argument("--null-reductions", type=Path, required=True,
                   help="for `fire_rate` (a property of the SAEs)")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", default="b", choices=["a", "b"],
                   help="Null B is primary (amendment G4)")
    p.add_argument("--stratum", default="pooled")
    p.add_argument("--concept-axes", default=",".join(DEFAULT_CONCEPT_AXES),
                   help="comma-separated subset of "
                        f"{sorted(CONCEPT_AXES)}; default is amendment O2's "
                        "committed four-axis scheme")
    p.add_argument("--inner-mode", default="full",
                   choices=["full", "centre", "none", "ratio"],
                   help="stage 1 variant. `full` = (g-mu)/sigma as K2 "
                        "specifies; `centre` = g-mu only; `none` = raw g, "
                        "stage 1 disabled. Separating the two halves is "
                        "measurement 1: scaling is inert-to-harmful while "
                        "centring may be doing real work. `ratio` = g/mu, "
                        "GSEA's normalization, adopted for the regression "
                        "predictor in amendment U1.")
    p.add_argument("--no-loo-inner", action="store_true",
                   help="disable leave-one-out at STAGE 1 only (stage 3 LOO "
                        "is not optional). For measuring what LOO buys.")
    p.add_argument("--mu-smooth", type=Path, default=None,
                   help="M4 smooth-trend mu_hat h5 (dataset mu_hat_full, "
                        "(seeds, concepts_full, features)). When given, it "
                        "REPLACES the hard-stratum mu for inner-mode ratio.")
    p.add_argument("--axis-values", default="labels", choices=["labels", "stratum"],
                   help="amendment CT §CT12.1. `labels` (default, every run of "
                        "record): the label store's full-population diagnostics. "
                        "`stratum`: the stratum's OWN values on the AP's scored "
                        "row set -- its AP prevalence, and protein and cluster "
                        "counts over its own positives. Needs --eval-cache and "
                        "--clusters-tsv.")
    p.add_argument("--eval-cache", type=Path, default=None,
                   help="for its index_json (row -> protein), --axis-values stratum")
    p.add_argument("--clusters-tsv", type=Path, default=None,
                   help="mmseqs 30%% clusters, --axis-values stratum")
    p.add_argument("--admission-strata", nargs="+", default=None,
                   help="amendment CT §CT5.1 (the author, option 1 with the pools "
                        "re-matched): keep only concepts with at least the label "
                        "store's `min_protein_count` proteins on the AP's row set "
                        "in EVERY stratum named (the scored --stratum among them), "
                        "in the AP axis and the null pool alike. Needs "
                        "--axis-values stratum.")
    p.add_argument("--row-set-file", type=Path, default=None,
                   help="amendment CT §CT10: the halves file, when the AP's "
                        "scored_row_set is ct10_half_a / ct10_half_b")
    p.add_argument("--merge-lowest-firing-bin", action="store_true",
                   help="amendment CT §CT14 (a declared sensitivity, off by "
                        "default): after K1's collapse, merge the lowest "
                        "firing-rate bin into the next (merge_lowest_firing_bin). "
                        "Recorded in the output's attrs.")
    p.add_argument("--merge-top-firing-bin", action="store_true",
                   help="amendment CT §CT29-D2 item 2 (a declared sensitivity, "
                        "off by default): after K1's collapse, merge the top "
                        "firing-rate bin into the one below it "
                        "(merge_top_firing_bin). Recorded in the output's attrs.")
    p.add_argument("--accept-ap-arith-split", action="store_true",
                   help="amendment CT (the author, 2026-10-24): accept an AP whose "
                        "ap_arith_device_class differs from the null's, ONLY when "
                        "their gather_device_class agrees; recorded in identity")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--skip-preflight", action="store_true",
                   help="local debugging ONLY -- never for a kept run")
    args = p.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    labels_json = args.labels_json or args.labels.with_suffix(".json")
    spec = AxisSpec(tuple(a.strip() for a in args.concept_axes.split(",") if a.strip()))

    # The population and the AP implementation the gate must EXPECT, read from
    # the AP before anything else (amendment CT §CT12.1). `scored_row_set` is
    # the AP's own; the implementation is derived from a DIFFERENT attr --
    # masked negatives force the segmented path -- so the check is not the
    # artifact compared with itself. Every run of record read an unmasked
    # label_store AP: label_store and the sparse default, unchanged.
    with h5py.File(args.ap, "r") as hf:
        _rs = hf.attrs.get("scored_row_set", "label_store")
        _neg = hf.attrs.get("negatives", "all_rows")
    expect_row_set = _rs.decode() if isinstance(_rs, bytes) else str(_rs)
    _neg = _neg.decode() if isinstance(_neg, bytes) else str(_neg)
    from src.analysis.artifact_identity import AP_IMPL_SEGMENTED
    from src.analysis.preflight import AP_IMPL_SPARSE
    expect_impl = AP_IMPL_SEGMENTED if _neg == "unannotated" else AP_IMPL_SPARSE

    # ---- PRE-FLIGHT ON THE INPUTS, before a single number is computed.
    if not args.skip_preflight:
        from src.analysis.preflight import check
        excluded = None
        if args.config is not None:
            import yaml
            excluded = tuple(yaml.safe_load(args.config.read_text())
                             .get("annotations", {})
                             .get("provenance_excluded_types", []))
        pf = check(args.labels, args.null_reductions, (args.ap,),
                   config_excluded_types=excluded, expect_ap_impl=expect_impl,
                   expect_scored_row_set=expect_row_set,
                   accept_ap_arith_split=args.accept_ap_arith_split)
        accepted_splits = pf.get("ap_arith_splits_accepted", [])
        print("PRE-FLIGHT PASSED on inputs"
              + (f" (AP-arithmetic device split ACCEPTED, gather agreeing: {accepted_splits})"
                 if accepted_splits else ""))
    else:
        accepted_splits = None

    with h5py.File(args.ap, "r") as hf:
        g = hf[args.stratum]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in g["concepts"][:]]
        prevalence = g["prevalence"][:]
        ap = g["ap"]                                   # (seeds, feats, concepts)
        n_seeds, n_feats, n_c_all = ap.shape
        ap_all = ap[:]                                 # ~583 MB float32
        n_rows = int(hf.attrs["n_rows"])
        ap_attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                    for k, v in hf.attrs.items()}
    print(f"AP: {n_seeds} seeds x {n_feats} features x {n_c_all} concepts")

    # ---- amendment CT §CT5.1: the admission gate, BEFORE any arithmetic.
    # The counts are each stratum's own on the AP's row set (§CT12.1); the
    # scored stratum's double as its concept-axis values.
    counts = admission = None
    axis_concepts = list(concepts)                 # the AP's axis, before admission
    if args.admission_strata and args.axis_values != "stratum":
        raise SystemExit("FATAL: --admission-strata needs --axis-values stratum")
    if args.axis_values == "stratum":
        strata = [args.stratum] + [t for t in (args.admission_strata or [])
                                   if t != args.stratum]
        counts = _stratum_counts(args, ap_attrs, concepts, strata)
    if args.admission_strata:
        if args.stratum not in args.admission_strata:
            raise SystemExit("FATAL: the scored --stratum must be one of the "
                             "--admission-strata")
        with h5py.File(args.labels, "r") as lf:
            gate = lf.attrs.get("min_protein_count")
        if gate is None:
            raise SystemExit("FATAL: the label store carries no "
                             "`min_protein_count`; the admission gate is its "
                             "own pre-declared gate, and an absent gate is not "
                             "a gate (gotcha 15). Refusing.")
        gate = int(gate)
        if gate != PROTEIN_COUNT_EDGES[0]:
            raise SystemExit(
                f"FATAL: the label store's min_protein_count {gate} is not the "
                f"protein-count axis's first edge {PROTEIN_COUNT_EDGES[0]}; an "
                "admitted concept could still fall outside the axis. Refusing.")
        keep = admission_mask({t: counts[t]["protein_count"]
                               for t in args.admission_strata}, gate)
        pool_before = np.array(argmax_pool_mask(axis_concepts))
        n_admitted_pool = int((keep & pool_before).sum())
        admission = {"strata": list(args.admission_strata), "min_proteins": gate,
                     "keep": keep, "n_pool_before": int(pool_before.sum()),
                     "n_admitted_pool": n_admitted_pool,
                     "n_admitted_axis": int(keep.sum())}
        for t in args.admission_strata:
            print(f"[admission] {t}: {int((pool_before & (counts[t]['protein_count'] < gate)).sum())} "
                  f"of {int(pool_before.sum())} pool concepts below {gate} proteins")
        print(f"[admission] admitted in all of {args.admission_strata}: "
              f"{n_admitted_pool} of {int(pool_before.sum())} pool concepts "
              f"({int(keep.sum())} of {len(keep)} on the axis)", flush=True)
        if n_admitted_pool == 0:
            raise SystemExit("UNMEASURABLE: no pool concept is admitted in every "
                             "stratum; the score's max is over an empty pool "
                             "(§CT5.1: the minimum is one concept).")
        concepts = [c for c, k in zip(axis_concepts, keep) if k]
        ap_all = ap_all[:, :, keep]
        prevalence = prevalence[keep]
        n_c_all = len(concepts)

    with h5py.File(args.null_raw, "r") as hf:
        null_concepts = [c.decode() if isinstance(c, bytes) else c
                         for c in hf["concepts"][:]]
        raw = hf[f"gain_{args.arm}"]
        n_shifts = raw.shape[1]
        # THE POOL AXES MUST BE THE SAME OBJECT, not merely the same size.
        # I4's defect was a null maxed over a different concept set than the
        # statistic it is a null for; a length check cannot see that. Under
        # admission it is checked TWICE: the null against the AP's full pool
        # (the files are each other's), then the selected null columns against
        # the admitted pool (the restriction is the same on both sides).
        if admission is not None:
            pool_full = [c for c, k in zip(axis_concepts,
                                           argmax_pool_mask(axis_concepts)) if k]
            if pool_full != list(null_concepts):
                raise SystemExit(
                    "FATAL: the null's concept axis is not the AP matrix's "
                    f"argmax pool before admission. null={len(null_concepts)}, "
                    f"AP pool={len(pool_full)} (amendment I4).")
            admitted = set(concepts)
            null_cols = np.array([i for i, c in enumerate(null_concepts)
                                  if c in admitted], dtype=np.int64)
            null_concepts = [null_concepts[i] for i in null_cols]
        else:
            null_cols = None
        # Under admission the pool is the ORIGINAL axis's pool, restricted --
        # never recomputed on the restricted axis: a pooled parent whose
        # subtypes were all refused would otherwise rejoin the pool and win
        # maxes it may not win (amendment I4; found by this module's test).
        pool_mask = (np.array(argmax_pool_mask(axis_concepts))[admission["keep"]]
                     if admission is not None
                     else np.array(argmax_pool_mask(concepts)))
        pool_idx = np.flatnonzero(pool_mask)
        smooth_mu = None
        if args.mu_smooth is not None:
            if args.inner_mode != "ratio":
                raise SystemExit(
                    "--mu-smooth is only meaningful for --inner-mode ratio "
                    f"(got {args.inner_mode!r})")
            with h5py.File(args.mu_smooth, "r") as sf:
                if int(sf.attrs["argmax_pool_n"]) != int(pool_mask.sum()):
                    raise SystemExit(
                        "FATAL: smooth mu_hat pool "
                        f"{int(sf.attrs['argmax_pool_n'])} != AP pool "
                        f"{int(pool_mask.sum())} -- identity mismatch (I3).")
                smooth_mu = sf["mu_hat_full"][:]
            print(f"smooth mu_hat loaded: {smooth_mu.shape} "
                  f"(replaces hard-stratum mu)")
        pool_from_ap = [c for c, k in zip(concepts, pool_mask) if k]
        if pool_from_ap != list(null_concepts):
            raise SystemExit(
                "FATAL: the null's concept axis is not the AP matrix's argmax "
                f"pool. null={len(null_concepts)}, AP pool={len(pool_from_ap)}. "
                "Standardizing against a null drawn on a different pool is "
                "the amendment-I4 defect.")

        with h5py.File(args.null_reductions, "r") as rf:
            fire_rate = rf["fire_rate"][:]             # (seeds, features)
            # ---- amendment CQ, A2: the reference order and the raw null's
            # recovery target, read BEFORE any positional indexing.
            null_g_key = f"null_g_{args.arm}"
            if null_g_key not in rf:
                raise SystemExit(
                    f"FATAL: {args.null_reductions} has no {null_g_key}, so "
                    f"the raw null's seed order cannot be recovered and "
                    f"checked (CQ A2-iii). Refusing.")
            null_g_ref = rf[null_g_key][:]             # (seeds, shifts, feats)
        ref_labels = stored_seed_labels(args.null_reductions)
        if ref_labels is None:
            raise SystemExit(
                "FATAL: the null reductions carry no seed labels; there is "
                "no reference order to check the other inputs against (CQ A2).")
        if len(ref_labels) != n_seeds or fire_rate.shape[0] != n_seeds:
            raise SystemExit(
                f"FATAL: {len(ref_labels)} seed labels, fire_rate has "
                f"{fire_rate.shape[0]} rows, AP has {n_seeds} seeds.")
        # (i) the AP matrix's own labels, as an ordered list.
        assert_seed_order_agrees("null_reductions.seeds", ref_labels,
                                 f"{args.ap}:seeds", stored_seed_labels(args.ap))
        # (ii) fire_rate comes from the reference artifact itself: its order IS
        # the reference by construction; the row count was asserted above.
        # (iv) the smooth mu_hat, when supplied, is indexed positionally too.
        if args.mu_smooth is not None:
            assert_seed_order_agrees("null_reductions.seeds", ref_labels,
                                     f"{args.mu_smooth}:seeds",
                                     stored_seed_labels(args.mu_smooth))
        print(f"[seed order] reference (null reductions): {ref_labels}; AP "
              f"{'and mu_smooth ' if args.mu_smooth is not None else ''}"
              f"agree as ordered lists; raw null checked per slab below",
              flush=True)

        fr_edges = collapse_firing_bins(fire_rate, FIRING_RATE_EDGES,
                                        MIN_FEATURES_PER_STRATUM)
        if args.merge_lowest_firing_bin:
            print(f"§CT14: merging the lowest firing bin, {fr_edges} -> "
                  f"{merge_lowest_firing_bin(fr_edges)}", flush=True)
            fr_edges = merge_lowest_firing_bin(fr_edges)
        if args.merge_top_firing_bin:
            print(f"§CT29-D2: merging the top firing bin, {fr_edges} -> "
                  f"{merge_top_firing_bin(fr_edges)}", flush=True)
            fr_edges = merge_top_firing_bin(fr_edges)
        n_fbins = len(fr_edges) - 1
        print(f"firing bins: {len(FIRING_RATE_EDGES)-1} base -> {n_fbins} "
              f"retained (L1 outward-in), edges {fr_edges}")

        if args.axis_values == "labels":
            vals = load_axis_values(args.labels, labels_json, concepts, n_rows)
        else:
            sel = (admission["keep"] if admission is not None
                   else np.ones(len(axis_concepts), dtype=bool))
            vals = {"prevalence": np.asarray(prevalence, dtype=float),
                    "protein_count": counts[args.stratum]["protein_count"][sel],
                    "cluster_count": counts[args.stratum]["cluster_count"][sel]}
        cell, n_cells = concept_cell_index(vals, spec)
        print(f"axis spec: {spec.name}  -> {n_cells} concept cells x "
              f"{n_fbins} firing bins")
        if (cell < 0).any():
            raise SystemExit(
                f"FATAL: {int((cell < 0).sum())} concepts fall outside the "
                "concept-axis edges. They would be silently pooled into bin 0.")

        prev_bcast = prevalence[None, None, :]
        out_z = np.empty((n_seeds, n_feats, n_c_all), dtype=np.float32)
        out_M = np.empty((n_seeds, n_feats), dtype=np.float64)
        out_am = np.empty((n_seeds, n_feats), dtype=np.int32)
        out_gt = np.empty((n_seeds, n_feats), dtype=np.float64)
        out_Mn = np.empty((n_seeds, n_shifts, n_feats), dtype=np.float32)
        out_gtn = np.empty((n_seeds, n_shifts, n_feats), dtype=np.float32)
        mus = np.empty((n_seeds, n_cells, n_fbins))
        sigs = np.empty((n_seeds, n_cells, n_fbins))
        # `stratum_mu` is indexed by (concept cell, FIRING BIN), and the concept
        # side is persisted as `concept_cell` while the feature side was not --
        # so a per-feature normaliser vector could not be reconstructed from
        # what was on disk, and C1 for the HARD-BIN mu was not computable at all
        # (m4_confound reported it absent rather than approximated, gotcha 15).
        # Hard-bin is now the predictor of record (AC9) and its mu also depends
        # on a feature-side axis, so the confound question has to be closable
        # for it. Persisting the assignment is additive: it changes no statistic.
        # int8 is sufficient -- there are at most 9 base bins before collapse.
        fbins = np.empty((n_seeds, n_feats), dtype=np.int8)

        for s in range(n_seeds):
            # G = (AP - p) / (1 - p): normalise the gain BEFORE any max (G1).
            real = ((ap_all[s].T.astype(np.float64) - prevalence[:, None])
                    / (1.0 - prevalence[:, None]))       # (concepts, features)
            slab = raw[s].astype(np.float64)             # (shifts, pool, feats)
            # (iii) amendment CQ, A2: the raw null has no labels -- slab s must
            # BE the reductions' row s, established from the data, exactly.
            # On the FULL slab: the reductions are its max over every concept.
            assert_raw_slab_is_reference_row(slab, null_g_ref, s, ref_labels,
                                             args.arm, stored_dtype=raw.dtype)
            if null_cols is not None:
                slab = slab[:, null_cols, :]
            # PER SEED, never on a cross-seed mean: firing rate by feature index
            # has cross-seed correlation 0.0009 (gotcha 18), so a mean would bin
            # 20 unrelated variables.
            fb = bin_index(fire_rate[s], fr_edges)
            fb = np.clip(fb, 0, n_fbins - 1)
            fbins[s] = fb
            mgr = mgp = None
            if smooth_mu is not None:
                mgr = smooth_mu[s].astype(np.float64)           # (concepts, feats)
                mgp = mgr[pool_idx]
                if not np.all(mgr > 0):
                    raise SystemExit(
                        "FATAL: smooth mu_hat has non-positive entries; it is "
                        "a denominator. Refusing rather than flooring.")
            res = standardize_seed(real, slab, cell, n_cells, fb, n_fbins,
                                   pool_mask, loo_inner=not args.no_loo_inner,
                                   inner_mode=args.inner_mode,
                                   mu_grid_real=mgr, mu_grid_pool=mgp)
            out_z[s] = res["z_real"].T.astype(np.float32)
            out_M[s], out_am[s] = res["M"], res["argmax_concept"]
            out_gt[s] = res["gtilde"]
            out_Mn[s] = res["M_null"].astype(np.float32)
            out_gtn[s] = res["gtilde_null"].astype(np.float32)
            mus[s], sigs[s] = res["mu"], res["sigma"]
            print(f"  seed {s:2d}: g~ median {np.nanmedian(res['gtilde']):.3f} "
                  f"max {np.nanmax(res['gtilde']):.3f}  "
                  f"M median {np.nanmedian(res['M']):.3f}")
        del prev_bcast

    with h5py.File(args.out, "w") as out:
        out.create_dataset("concepts",
                           data=np.array(concepts, dtype=h5py.string_dtype()))
        out.create_dataset("in_argmax_pool", data=pool_mask)
        out.create_dataset("z", data=out_z, compression="gzip")
        out.create_dataset("M", data=out_M, compression="gzip")
        out.create_dataset("argmax_concept", data=out_am, compression="gzip")
        out.create_dataset("gtilde", data=out_gt, compression="gzip")
        out.create_dataset("M_null", data=out_Mn, compression="gzip")
        out.create_dataset("gtilde_null", data=out_gtn, compression="gzip")
        out.create_dataset("stratum_mu", data=mus, compression="gzip")
        out.create_dataset("stratum_sigma", data=sigs, compression="gzip")
        out.create_dataset("concept_cell", data=cell)
        # THE SEED AXIS IS LABELLED (amendment AS decision 5). Its
        # absence is what let AR9 stay invisible: a consumer could not
        # read this artifact's order off the file, so it defaulted to
        # position -- and this axis is LEXICOGRAPHIC while rf_counts and
        # d_fs are NUMERIC. The labels come from the null reductions,
        # which is the axis every array here was indexed against.
        _seed_labels = stored_seed_labels(args.null_reductions)   # module import (CQ)
        if _seed_labels is None:
            raise SystemExit(
                "FATAL: the null reductions carry no seed labels, so "
                "this artifact's seed order cannot be recorded. "
                "Refusing to write an unlabelled seed axis (AS).")
        if len(_seed_labels) != n_seeds:
            raise SystemExit(
                f"FATAL: {len(_seed_labels)} seed labels for {n_seeds} "
                "rows.")
        out.create_dataset("seeds", data=[x.encode() for x in _seed_labels],
                           dtype=h5py.string_dtype())
        out.attrs["seed_label_source"] = str(args.null_reductions)
        # The feature-side index into `stratum_mu`, the companion of
        # `concept_cell`. Interpret it against the `firing_rate_edges` attr --
        # which round-trips as a JSON STRING, so `json.loads` it; `list()` of it
        # yields CHARACTERS and silently produces plausible wrong bins
        # (gotcha 20). Values are post-collapse and post-clip: a feature below
        # the lowest retained edge is pooled into bin 0, not excluded.
        out.create_dataset("feature_firing_bin", data=fbins)
        for k in spec.concept_axes:
            out.create_dataset(f"axis_{k}", data=vals[k])
        if counts is not None:
            # §CT5.1: every stratum's own counts on the AP's axis BEFORE
            # admission, and which concepts were admitted -- the refusals are a
            # result in their own right, so they are on disk, not only printed.
            grp = out.create_group("stratum_counts")
            grp.create_dataset("concepts", data=np.array(
                axis_concepts, dtype=h5py.string_dtype()))
            grp.create_dataset("in_argmax_pool",
                               data=np.array(argmax_pool_mask(axis_concepts)))
            if admission is not None:
                grp.create_dataset("admitted", data=admission["keep"])
            for t, ct in counts.items():
                for k, v in ct.items():
                    grp.create_dataset(f"{t}/{k}", data=v)
        with h5py.File(args.labels, "r") as lab:
            lab_attrs = {k: v for k, v in lab.attrs.items()}
        for k in ("truncation_policy", "n_rows", "retention_floor_mode",
                  "retention_min_residues", "retention_min_instances",
                  "min_protein_count"):
            if k in lab_attrs:
                out.attrs[k] = lab_attrs[k]
        write_identity(
            # LIBRARY size (gotcha 16), from the AP's own attrs: a stratum AP's
            # concept axis is the strata's common set, not the library. For a
            # pooled AP the two are equal, so runs of record are unchanged.
            out, n_concepts=int(ap_attrs.get("n_concepts", len(concepts))),
            n_concepts_library=int(ap_attrs.get("n_concepts", len(concepts))),
            # LIBRARY-scoped (gotcha 16, as feature_ap and null_pilot write it):
            # the AP's own `argmax_pool_n`. The pool this run maxed over is
            # `argmax_pool_used_n`; on a pooled run of record the three agree.
            argmax_pool_n=int(ap_attrs.get("argmax_pool_n", pool_mask.sum())),
            argmax_pool_scored_n=int(np.sum(argmax_pool_mask(axis_concepts))),
            argmax_pool_used_n=int(pool_mask.sum()),
            admission_strata=(admission["strata"] if admission else None),
            admission_min_protein_count=(admission["min_proteins"]
                                         if admission else None),
            stratum=args.stratum,
            library_sha=(lab_attrs["git_sha"].decode()
                         if isinstance(lab_attrs.get("git_sha"), bytes)
                         else lab_attrs.get("git_sha")),
            labels=str(args.labels),
            # The AP's own population, not a constant (amendment CT §CT12.1);
            # every run of record read a label_store AP, so this is unchanged
            # for them.
            scored_row_set=str(ap_attrs.get("scored_row_set", "label_store")),
            axis_values=args.axis_values,
            exclude_pooled_parents=True,
            null_arm=args.arm, n_shifts=int(n_shifts),
            concept_axes=list(spec.concept_axes),
            firing_rate_edges=[float(x) for x in fr_edges],
            lowest_firing_bin_merged=bool(args.merge_lowest_firing_bin),
            # written only when applied (None is skipped), so every other
            # run's identity set is unchanged
            top_firing_bin_merged=(True if args.merge_top_firing_bin else None),
            loo_inner=not args.no_loo_inner, loo_outer=True,
            inner_mode=args.inner_mode,
            ap_source=str(args.ap), null_raw_source=str(args.null_raw),
            # amendment CQ, A5: numpy on the host, so CPU by construction;
            # recorded in the shared vocabulary rather than asserted.
            device_class=device_class("cpu"),
            # amendment CT, §CT24-F-C3 / §CT25-F-C1: the flag and what it let
            # through, in the output's identity (None = pre-flight skipped)
            accept_ap_arith_split=bool(args.accept_ap_arith_split),
            ap_arith_splits_accepted=json.dumps(accepted_splits),
            git_sha=git_sha(), git_dirty=bool(git_dirty()))
    print(f"wrote {args.out}")

    # ---- PRE-FLIGHT ON THE OUTPUT. An output that cannot be asserted
    # against the store it was derived from is the stale-identity problem in
    # a new file (gotcha 15).
    if not args.skip_preflight:
        from src.analysis.preflight import check
        check(args.labels, args.out, (), expect_scored_row_set=expect_row_set)
        print("PRE-FLIGHT PASSED on output")


if __name__ == "__main__":
    main()
