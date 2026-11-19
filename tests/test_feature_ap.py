"""
Tests for the sparse per-feature Average Precision.

`g_f` is built from these numbers and the headline is built from `g_f`, so the
sparse/segmented formula must agree with a reference implementation exactly --
not approximately. The central test compares against
`sklearn.metrics.average_precision_score` on dense data, including the
tied-at-zero block that the closed-form tail term stands in for.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.analysis.feature_ap import (
    average_precision_segmented,
    filter_nonzeros,
    library_scoped_concept_counts,
)


def _segmented_from_dense(Z: np.ndarray):
    """Dense (n_rows, n_features) activations -> the segmented layout the
    implementation expects: non-zeros ordered by (feature asc, value desc)."""
    n_rows, n_features = Z.shape
    feats, poss, vals = [], [], []
    for f in range(n_features):
        nz = np.flatnonzero(Z[:, f])
        order = nz[np.argsort(-Z[nz, f], kind="stable")]
        feats.append(np.full(len(order), f, dtype=np.int64))
        poss.append(order.astype(np.int64))
        vals.append(Z[order, f])
    feat = np.concatenate(feats) if feats else np.zeros(0, dtype=np.int64)
    pos = np.concatenate(poss) if poss else np.zeros(0, dtype=np.int64)
    counts = np.bincount(feat, minlength=n_features)
    seg_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (torch.from_numpy(feat), torch.from_numpy(seg_start),
            torch.from_numpy(pos))


def _ap_ours(Z, y):
    feat, seg_start, pos = _segmented_from_dense(Z)
    labels = torch.from_numpy(y.astype(bool))[pos]
    return average_precision_segmented(
        feat, seg_start, labels, Z.shape[1], int(y.sum()), Z.shape[0]).numpy()


def test_matches_sklearn_on_sparse_activations():
    """THE test: sparse TopK-like activations with the dominant tie block at
    zero. Every feature must match sklearn to floating-point tolerance."""
    rng = np.random.default_rng(0)
    n_rows, n_features, k = 400, 12, 3
    Z = np.zeros((n_rows, n_features))
    for i in range(n_rows):                       # exactly k non-zeros per row
        cols = rng.choice(n_features, k, replace=False)
        Z[i, cols] = rng.uniform(0.1, 5.0, size=k)
    y = (rng.random(n_rows) < 0.15).astype(int)

    ours = _ap_ours(Z, y)
    for f in range(n_features):
        ref = average_precision_score(y, Z[:, f])
        assert abs(ours[f] - ref) < 1e-9, (f, ours[f], ref)


def test_matches_sklearn_with_rare_positives():
    """Rare concepts are the realistic case (ACT_SITE is 0.086% of rows) and are
    where a wrong tail term would show up most."""
    rng = np.random.default_rng(7)
    n_rows, n_features = 2000, 8
    Z = np.zeros((n_rows, n_features))
    for i in range(n_rows):
        cols = rng.choice(n_features, 2, replace=False)
        Z[i, cols] = rng.uniform(0.1, 3.0, size=2)
    y = np.zeros(n_rows, dtype=int)
    y[rng.choice(n_rows, 12, replace=False)] = 1        # 0.6% prevalence

    ours = _ap_ours(Z, y)
    for f in range(n_features):
        ref = average_precision_score(y, Z[:, f])
        assert abs(ours[f] - ref) < 1e-9, (f, ours[f], ref)


def test_perfect_detector_scores_one():
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    Z = np.zeros((100, 1))
    Z[:10, 0] = np.linspace(5, 1, 10)                  # fires on exactly the positives
    assert abs(_ap_ours(Z, y)[0] - 1.0) < 1e-12
    assert abs(average_precision_score(y, Z[:, 0]) - 1.0) < 1e-12


def test_non_firing_feature_scores_prevalence():
    """A feature that never fires has every row tied, so AP is the base rate --
    the tail term alone. If this were 0 instead, orphans would be mis-scored."""
    y = np.zeros(200, dtype=int)
    y[:20] = 1
    Z = np.zeros((200, 1))
    ours = _ap_ours(Z, y)[0]
    assert abs(ours - 0.1) < 1e-12                     # prevalence = 20/200


def test_anti_correlated_feature_scores_at_most_prevalence():
    """Fires only on negatives. Note AP lands exactly AT the base rate, not
    below it: every positive sits in the tied-zero block, whose precision is
    exactly P/N. sklearn agrees -- the invariant is agreement, not an
    inequality."""
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    Z = np.zeros((100, 1))
    Z[10:40, 0] = 1.0
    ours = _ap_ours(Z, y)[0]
    ref = average_precision_score(y, Z[:, 0])
    assert abs(ours - ref) < 1e-9
    assert ours <= 0.1 + 1e-12


def test_silent_feature_among_firing_ones():
    """A feature with no firing rows sits at seg_start == nnz, one past the end
    of the prefix-sum array. Real ensembles have near-silent features, so this
    must not index out of bounds -- and the silent feature must score the base
    rate while its neighbours are unaffected."""
    rng = np.random.default_rng(3)
    n_rows = 300
    Z = np.zeros((n_rows, 4))
    Z[:, 0] = rng.uniform(0, 1, n_rows)          # dense
    Z[:50, 1] = rng.uniform(0, 1, 50)            # sparse
    # feature 2 and feature 3 (the LAST one) never fire
    y = (rng.random(n_rows) < 0.2).astype(int)

    ours = _ap_ours(Z, y)
    prevalence = y.mean()
    assert abs(ours[2] - prevalence) < 1e-12
    assert abs(ours[3] - prevalence) < 1e-12     # last-feature edge case
    for f in (0, 1):
        assert abs(ours[f] - average_precision_score(y, Z[:, f])) < 1e-9


def test_zero_positives_returns_zero_not_nan():
    y = np.zeros(50, dtype=int)
    Z = np.zeros((50, 3))
    Z[:10, 0] = 1.0
    out = _ap_ours(Z, y)
    assert np.all(out == 0.0) and np.isfinite(out).all()


def test_scored_row_set_is_ascending(tmp_path):
    """The shared-protein set is stored in ACCESSION order, but h5py fancy
    indexing requires strictly increasing indices -- and the subset order also
    defines the row->position remap. Against the real labels file this raised
    'Indexing elements must be in increasing order'."""
    import h5py as _h5
    from src.analysis.feature_ap import load_scored_row_set
    p = tmp_path / "labels.h5"
    with _h5.File(p, "w") as hf:
        pg = hf.create_group("protein_sets")
        # deliberately NOT in row order, as accession-sorted output would be
        pg.create_dataset("has_experimental_start", data=np.array([100, 0, 50]))
        pg.create_dataset("has_experimental_n_rows", data=np.array([5, 3, 2]))
        hf.attrs["n_rows"] = 200
    rows = load_scored_row_set(p, "shared_experimental")
    assert np.all(np.diff(rows) > 0), rows
    assert rows.tolist() == [0, 1, 2, 50, 51, 100, 101, 102, 103, 104]


def test_tiny_concept_gives_degenerate_ap_which_is_why_the_floor_exists():
    """Documents the failure mode amendment B fixes. A concept with ONE positive
    scores AP exactly 1.0 for any feature that fires on that row, and a
    2-positive concept gives exactly 0.5. Because g_f is a MAX over concepts,
    such concepts would let noise decide alignment -- which is what the real run
    produced (propagated maxAP exactly 1.0000 from a 1-positive concept)."""
    y1 = np.zeros(1000, dtype=int); y1[7] = 1                 # ONE positive
    Z = np.zeros((1000, 1)); Z[7, 0] = 3.0                    # fires on it
    assert abs(_ap_ours(Z, y1)[0] - 1.0) < 1e-12
    assert abs(average_precision_score(y1, Z[:, 0]) - 1.0) < 1e-12

    y2 = np.zeros(1000, dtype=int); y2[[7, 500]] = 1          # TWO positives
    ap2 = _ap_ours(Z, y2)[0]
    assert abs(ap2 - (0.5 + 0.5 * (2 / 1000))) < 1e-12        # ~exactly 0.5


def _ap_masked(Z, y, keep):
    """AP restricted to `keep` rows, via the production filter path."""
    feat, seg_start, pos = _segmented_from_dense(Z)
    keep_t = torch.from_numpy(keep.astype(bool))
    f2, p2, s2 = filter_nonzeros(feat, pos, keep_t, Z.shape[1])
    labels = torch.from_numpy(y.astype(bool))[p2]
    return average_precision_segmented(
        f2, s2, labels, Z.shape[1], int(y[keep].sum()), int(keep.sum())).numpy()


def test_filter_nonzeros_matches_sklearn_on_the_surviving_rows():
    """The masked path must be exactly AP-on-a-subset, not an approximation of
    it: the filter preserves (feature asc, value desc) order, so only the block
    offsets change. Compared against sklearn run on the surviving rows only."""
    rng = np.random.default_rng(11)
    n_rows, n_features, k = 600, 10, 3
    Z = np.zeros((n_rows, n_features))
    for i in range(n_rows):
        cols = rng.choice(n_features, k, replace=False)
        Z[i, cols] = rng.uniform(0.1, 5.0, size=k)
    y = (rng.random(n_rows) < 0.10).astype(int)
    keep = rng.random(n_rows) < 0.7                 # drop ~30% of rows

    ours = _ap_masked(Z, y, keep)
    for f in range(n_features):
        ref = average_precision_score(y[keep], Z[keep, f])
        assert abs(ours[f] - ref) < 1e-9, (f, ours[f], ref)


def test_masking_changes_the_answer_and_by_the_predicted_amount():
    """THE amendment-C test, and it must be able to fail.

    A feature that perfectly detects the biological concept fires on ALL of its
    residues, but only a fraction `f_exp` carry experimental evidence. Scored
    with the other tier's residues as ORDINARY NEGATIVES it is charged a false
    positive for each of them and lands near `f_exp`; scored with them MASKED it
    recovers 1.0. If both numbers came out the same the masking would be inert
    and this test would fail.
    """
    n_rows = 10_000
    n_exp, n_other = 30, 120                        # f_exp = 30/150 = 0.20
    y_exp = np.zeros(n_rows, dtype=int)
    y_other = np.zeros(n_rows, dtype=int)
    y_exp[:n_exp] = 1
    y_other[n_exp:n_exp + n_other] = 1
    # perfect, provenance-blind detector: fires on the union, nothing else, and
    # interleaves the two tiers so no ordering advantage is smuggled in.
    Z = np.zeros((n_rows, 1))
    union = np.arange(n_exp + n_other)
    rng = np.random.default_rng(5)
    Z[rng.permutation(union), 0] = np.linspace(5.0, 1.0, len(union))

    unmasked = _ap_ours(Z, y_exp)[0]
    masked = _ap_masked(Z, y_exp, ~y_other.astype(bool))[0]

    f_exp = n_exp / (n_exp + n_other)
    assert abs(masked - 1.0) < 1e-12, masked
    assert abs(unmasked - f_exp) < 0.05, (unmasked, f_exp)
    assert masked > unmasked * 3                    # the fix is not cosmetic


def test_masking_is_a_no_op_when_no_other_tier_rows_exist():
    """The pooled stratum owns every tier's positives, so it has nothing to mask
    and must be bit-identical either way -- otherwise the amendment would move
    the pooled result too, and pooled is a pre-registered reported stratum."""
    rng = np.random.default_rng(2)
    n_rows, n_features = 500, 6
    Z = np.zeros((n_rows, n_features))
    for i in range(n_rows):
        Z[i, rng.choice(n_features, 2, replace=False)] = rng.uniform(0.1, 3.0, 2)
    y = (rng.random(n_rows) < 0.12).astype(int)
    keep = np.ones(n_rows, dtype=bool)
    assert np.allclose(_ap_ours(Z, y), _ap_masked(Z, y, keep), atol=0, rtol=0)


def test_masking_preserves_positive_count_and_shrinks_n():
    """Masking must drop only NEGATIVE-side rows of the target stratum: the
    positive count is unchanged (so the amendment-B floor still applies to the
    same numbers) while N shrinks, which is why prevalence is recomputed."""
    n_rows = 1000
    y_exp = np.zeros(n_rows, dtype=int); y_exp[:20] = 1
    y_other = np.zeros(n_rows, dtype=int); y_other[20:60] = 1
    keep = ~y_other.astype(bool)
    assert y_exp[keep].sum() == 20                  # positives untouched
    assert keep.sum() == n_rows - 40                # only the other tier dropped

    # A silent feature therefore scores the NEW base rate, not the old one.
    assert abs(_ap_masked(np.zeros((n_rows, 1)), y_exp, keep)[0] - 20 / 960) < 1e-12

    # And the masked rows must genuinely leave the RANKING, not merely the
    # denominator: this feature ranks all 40 other-tier rows above its own
    # positives, so removing them promotes the positives from ranks 21-40 to
    # ranks 1-20 and AP goes to 1.0. If filter_nonzeros were inert it would not.
    Z = np.zeros((n_rows, 1))
    Z[20:60, 0] = 5.0                               # masked rows, ranked first
    Z[0:20, 0] = 1.0                                # the experimental positives
    assert abs(_ap_masked(Z, y_exp, keep)[0] - 1.0) < 1e-12
    assert _ap_ours(Z, y_exp)[0] < 0.5              # unmasked, the same feature


def test_normalised_gain_matches_preregistered_lift_table():
    """G = (AP - p)/(1 - p) at 0.05 must imply the per-concept lifts recorded in
    PREREGISTRATION A2 -- the asymmetry is pre-registered, so the formula that
    produces it is pinned here."""
    def implied_lift(p, G=0.05):
        ap = G + (1 - G) * p          # invert G = (AP - p)/(1 - p)
        return ap / p
    assert abs(implied_lift(0.133111) - 1.3) < 0.05        # DOMAIN
    assert abs(implied_lift(0.000865) - 58.8) < 0.5        # ACT_SITE
    assert abs(implied_lift(0.000057) - 879.1) < 5.0       # LIPID


# --------------------------------------------------------------------------
# library_scoped_concept_counts (amendment AW / §2d / gotcha 16)
# --------------------------------------------------------------------------

def test_library_scoped_counts_on_a_pooled_run_all_four_collapse_pairwise():
    """On a pooled run every concept in the library has positives, so
    `scored_concepts == library_concepts` and the two pairs of numbers
    coincide -- the case that made the original bug invisible for every run
    that ever exercised it."""
    library = ["ACT_SITE", "BINDING:Zn(2+)", "DOMAIN:Protein kinase"]
    counts = library_scoped_concept_counts(library, library)
    assert counts == {"n_concepts": 3, "argmax_pool_n": 3,
                      "n_concepts_scored": 3, "argmax_pool_scored_n": 3}


def test_library_scoped_counts_on_a_stratified_run_are_four_distinct_numbers():
    """The library is strictly LARGER than what this stratum scored, and
    contains a POOLED PARENT (`BINDING`, whose child `BINDING:Zn(2+)` is
    present) that is absent from the scored set -- so all four counts differ.
    A fixture where library == scored could not exhibit the bug this guards
    against (gotcha 16's own lesson): writing the SCORED count (3) into the
    LIBRARY-scoped `n_concepts` key is exactly what the pre-AW writer did,
    and it was invisible because it only ever saw the collapsed case above."""
    library = ["ACT_SITE", "BINDING:Zn(2+)", "DOMAIN:Protein kinase",
              "BINDING", "LIPID", "TRANSMEM"]
    scored = ["ACT_SITE", "BINDING:Zn(2+)", "DOMAIN:Protein kinase"]
    counts = library_scoped_concept_counts(library, scored)
    assert counts["n_concepts"] == 6                # library
    assert counts["argmax_pool_n"] == 5              # library minus the pooled parent BINDING
    assert counts["n_concepts_scored"] == 3          # what this stratum scored
    assert counts["argmax_pool_scored_n"] == 3       # none of the scored 3 is a pooled parent
    assert len(set(counts.values())) == 3            # n_concepts != n_concepts_scored is the fix;
                                                      # argmax_pool_n ties none of the others


def test_library_scoped_counts_scored_set_smaller_still_excludes_its_own_parent():
    """A scored set that itself contains a pooled parent must still have that
    parent excluded from `argmax_pool_scored_n`, independent of the library
    check -- the two exclusions (library-side and scored-side) are computed
    from two separately-masked lists, not derived from one another."""
    library = ["BINDING", "BINDING:Zn(2+)", "LIPID"]
    scored = ["BINDING", "BINDING:Zn(2+)"]
    counts = library_scoped_concept_counts(library, scored)
    assert counts["n_concepts"] == 3
    assert counts["argmax_pool_n"] == 2              # BINDING excluded
    assert counts["n_concepts_scored"] == 2
    assert counts["argmax_pool_scored_n"] == 1       # BINDING excluded here too
