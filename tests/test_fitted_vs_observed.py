"""Tests for `fitted_vs_observed` -- PREREGISTRATION amendment BB section 2.

THE FIXTURES ARE BUILT TO DISCRIMINATE, which for this module means four
things a careless fixture would hide:

  * **the premise check must see a PERMUTATION.** Its whole job is to refuse
    two paths that select the same NUMBER of cells but not the same cells, so
    the wrong-order fixture has the SAME TOTAL as the right one. A check on a
    total cannot detect a permutation (gotcha 27) and a fixture that differs
    in count could not tell the two checks apart.
  * **the acceptance criterion must fail on a moved coefficient AND on an
    absent one.** An absent key reported as agreement is the failure mode
    gotcha 15 names, so it has its own case with its own expected verdict.
  * **the span comparison must be exercised in EVERY band**, including the one
    where the observed span's interval straddles zero -- the branch that
    exists because a ratio against a near-zero denominator ranks noise
    (gotcha 25b). A fixture with only a large positive observed span would
    leave three of five branches unreached.
  * **the refit must be checked against a KNOWN answer**, not against itself.
    `design3_fit` is tested by recovery; here what matters is that the fitted
    probabilities are the ones the fit implies and that they are NaN -- never
    0.0 -- outside the analysis floor.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import expit

from src.analysis import fitted_vs_observed as fvo
from src.analysis.design3_fit import build_design, fit_design3


# ------------------------------------------------------------------ helpers

def _tiny_fit_inputs(n_seeds=3, n_feat=200, seed=0):
    rng = np.random.default_rng(seed)
    logM = rng.normal(0.0, 1.0, size=(n_seeds, n_feat))
    log_phi = rng.uniform(-3.5, -2.0, size=(n_seeds, n_feat))
    d_fs = rng.uniform(0.5, 4.0, size=(n_seeds, n_feat))
    eta = 0.2 + 0.4 * logM + 0.3 * (log_phi + 2.5) + 0.1 * d_fs
    y = rng.binomial(39, expit(eta)).astype(np.float64)
    keep = np.ones((n_seeds, n_feat), dtype=bool)
    return y, logM, log_phi, d_fs, keep


# -------------------------------------------------------------- the refit

def test_refit_reproduces_a_direct_fit_design3_call_exactly():
    """The transcription is only trustworthy if it IS the incumbent loop. This
    compares it against `fit_design3` driven directly, coefficient for
    coefficient, on the same inputs."""
    y, logM, log_phi, d_fs, keep = _tiny_fit_inputs()
    fitted, rows = fvo.refit_primary(y, logM, log_phi, d_fs, keep, 39)

    s = 1
    lm = logM[s]
    sd = float(lm.std())
    z = (lm - lm.mean()) / sd
    des = build_design(z, log_phi[s], d_fs[s])
    f = fit_design3(y[s].astype(np.float64), 39, des["X"], des["S"])
    assert rows[s]["b1_z"] == float(f["beta"][des["idx_logM"]])
    assert rows[s]["intercept"] == float(f["beta"][0])
    assert rows[s]["deviance"] == f["deviance"]
    assert rows[s]["lambda"] == f["lambda"]
    np.testing.assert_array_equal(
        fitted[s], expit(np.clip(f["eta"], -30.0, 30.0)))


def test_cells_outside_the_floor_are_NaN_never_zero():
    """A 0.0 outside the analysis set would read as 'fitted, and zero' and
    would then be averaged into a bin. NaN cannot be averaged silently."""
    y, logM, log_phi, d_fs, keep = _tiny_fit_inputs()
    keep[0, :50] = False
    fitted, _ = fvo.refit_primary(y, logM, log_phi, d_fs, keep, 39)
    assert np.isnan(fitted[0, :50]).all()
    assert np.isfinite(fitted[0, 50:]).all()
    assert ((fitted[0, 50:] > 0) & (fitted[0, 50:] < 1)).all()


def test_the_spline_contribution_is_reported_and_is_not_degenerate():
    """The spline's own contribution to the linear predictor is the quantity
    that could NOT be read off the stored artifact, so the refit must actually
    produce it rather than emit a placeholder."""
    y, logM, log_phi, d_fs, keep = _tiny_fit_inputs()
    _, rows = fvo.refit_primary(y, logM, log_phi, d_fs, keep, 39)
    assert all(r["spline_contribution_sd"] > 0 for r in rows)


# ------------------------------------------------- ACCEPTANCE A, both ways

def _ref_doc(rows, tweak=None, drop=None):
    ref = []
    for r in rows:
        d = {k: r[k] for k in fvo.FIT_IDENTITY_FIELDS}
        ref.append(d)
    if tweak is not None:
        i, k, v = tweak
        ref[i][k] = v
    if drop is not None:
        i, k = drop
        ref[i].pop(k)
    return {"identity": {"measurement": "test"}, "primary": {"per_seed": ref}}


def test_acceptance_A_passes_on_an_exact_match(tmp_path):
    y, logM, log_phi, d_fs, keep = _tiny_fit_inputs()
    _, rows = fvo.refit_primary(y, logM, log_phi, d_fs, keep, 39)
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(_ref_doc(rows)))
    out = fvo.reproduces_join(rows, p)
    assert out["verdict"] == "REPRODUCES_THE_STORED_FIT_EXACTLY"
    assert out["n_disagreements"] == 0


def test_acceptance_A_fails_on_a_coefficient_moved_in_the_LAST_BIT(tmp_path):
    """Exact equality, not a tolerance. A tolerance would accept a genuinely
    different fit whose coefficients happened to land close."""
    y, logM, log_phi, d_fs, keep = _tiny_fit_inputs()
    _, rows = fvo.refit_primary(y, logM, log_phi, d_fs, keep, 39)
    moved = np.nextafter(rows[2]["b1_z"], np.inf)
    assert moved != rows[2]["b1_z"]
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(_ref_doc(rows, tweak=(2, "b1_z", moved))))
    out = fvo.reproduces_join(rows, p)
    assert out["verdict"] == "DIFFERS"
    assert any("b1_z" in s for s in out["problems"])


def test_acceptance_A_reports_an_ABSENT_key_as_a_problem_not_agreement(tmp_path):
    """Gotcha 15: an absent key is reported absent, NEVER treated as matching."""
    y, logM, log_phi, d_fs, keep = _tiny_fit_inputs()
    _, rows = fvo.refit_primary(y, logM, log_phi, d_fs, keep, 39)
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(_ref_doc(rows, drop=(0, "edf"))))
    out = fvo.reproduces_join(rows, p)
    assert out["verdict"] == "DIFFERS"
    assert any("ABSENT" in s for s in out["problems"])


def test_acceptance_A_fails_when_the_reference_has_a_different_seed_count(tmp_path):
    y, logM, log_phi, d_fs, keep = _tiny_fit_inputs()
    _, rows = fvo.refit_primary(y, logM, log_phi, d_fs, keep, 39)
    doc = _ref_doc(rows)
    doc["primary"]["per_seed"] = doc["primary"]["per_seed"][:-1]
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(doc))
    assert fvo.reproduces_join(rows, p)["verdict"] == "DIFFERS"


# ---------------------------------------------- the premise, positionally

def test_the_premise_check_detects_a_PERMUTATION_with_the_SAME_TOTAL():
    """THE LOAD-BEARING FIXTURE. The two `keep` masks have the SAME number of
    True cells and differ only in WHICH cells, which is exactly what a count
    or a total cannot see (gotcha 27). If this fixture had different totals it
    would pass a total check too and would prove nothing."""
    keep_a = np.zeros((2, 6), dtype=bool)
    keep_a[0, :3] = True
    keep_a[1, :3] = True
    keep_b = np.zeros((2, 6), dtype=bool)
    keep_b[0, 3:] = True          # same count per row, different cells
    keep_b[1, :3] = True
    assert keep_a.sum() == keep_b.sum()
    logM = np.arange(12, dtype=np.float64).reshape(2, 6)
    y = np.ones((2, 6))
    out = fvo.cross_path_agreement(keep_a, keep_b, logM, logM, y, y)
    assert out["verdict"] == "DIFFERENT_FEATURES"
    assert out["n_cells_disagreeing_keep"] == 6


def test_the_premise_check_passes_when_every_cell_agrees():
    keep = np.ones((2, 6), dtype=bool)
    logM = np.arange(12, dtype=np.float64).reshape(2, 6)
    y = np.ones((2, 6))
    out = fvo.cross_path_agreement(keep, keep, logM, logM, y, y)
    assert out["verdict"] == "SAME_FEATURES"
    assert out["max_abs_logM_difference"] == 0.0


def test_the_premise_check_sees_a_logM_permutation_that_preserves_the_SUM():
    """Two `log M` arrays with an identical sum and identical multiset, in a
    different order. Only a positional comparison discriminates."""
    keep = np.ones((1, 4), dtype=bool)
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    b = np.array([[4.0, 3.0, 2.0, 1.0]])
    assert a.sum() == b.sum()
    y = np.ones((1, 4))
    out = fvo.cross_path_agreement(keep, keep, a, b, y, y)
    assert out["verdict"] == "DIFFERENT_FEATURES"
    assert out["n_cells_disagreeing_logM"] == 4


def test_the_premise_check_sees_a_moved_outcome_count():
    keep = np.ones((1, 4), dtype=bool)
    logM = np.array([[1.0, 2.0, 3.0, 4.0]])
    y1 = np.array([[5.0, 6.0, 7.0, 8.0]])
    y2 = np.array([[5.0, 6.0, 7.0, 9.0]])
    out = fvo.cross_path_agreement(keep, keep, logM, logM, y1, y2)
    assert out["verdict"] == "DIFFERENT_FEATURES"
    assert out["n_cells_disagreeing_outcome"] == 1


# ------------------------------------------------------------- the spans

def _per_seed_from_means(means_per_seed):
    """A `per_seed` structure `paired_span` can consume, from bin means."""
    out = []
    for means in means_per_seed:
        bins = [{"bin": i, "status": "COMPUTED", "mean_r_f": m}
                for i, m in enumerate(means)]
        out.append({"bins": bins})
    return out


def test_paired_span_is_the_within_seed_difference_combined_across_seeds():
    ps = _per_seed_from_means([[0.10, 0.20, 0.30],
                               [0.20, 0.25, 0.40],
                               [0.15, 0.15, 0.35]])
    out = fvo.paired_span(ps)
    assert out["status"] == "COMPUTED"
    assert out["n_seeds"] == 3
    # spans are 0.20, 0.20, 0.20 -> mean 20 pp, and ZERO dispersion, which a
    # fixture with varying spans could not distinguish from a mean of spans.
    assert out["mean_span_pp"] == pytest.approx(20.0)
    assert out["se_span_pp"] == pytest.approx(0.0)


def test_paired_span_is_UNDEFINED_when_no_seed_has_two_usable_bins():
    ps = [{"bins": [{"bin": 0, "status": "COMPUTED", "mean_r_f": 0.3}]}]
    assert fvo.paired_span(ps)["status"] == \
        "UNDEFINED_NO_SEED_HAS_TWO_USABLE_BINS"


# ------------------------------------------- the verdict, in EVERY branch

def _span(mean_pp, se_pp, n=20):
    lo, hi = mean_pp - fvo.Z95 * se_pp, mean_pp + fvo.Z95 * se_pp
    return {"status": "COMPUTED", "n_seeds": n, "mean_span_pp": mean_pp,
            "se_span_pp": se_pp, "ci95_span_pp": [lo, hi],
            "per_seed_span_pp": []}


@pytest.mark.parametrize("fitted_pp,expected", [
    (2.0, "FITTED_SPAN_MATERIALLY_BELOW_OBSERVED"),      # ratio 0.25
    (6.0, "FITTED_SPAN_BELOW_OBSERVED_BUT_NOT_BY_HALF"),  # ratio 0.75
    (8.0, "FITTED_SPAN_CLOSE_TO_OBSERVED"),               # ratio 1.00
    (12.0, "FITTED_SPAN_EXCEEDS_OBSERVED"),               # ratio 1.50
])
def test_every_band_of_the_span_comparison_is_reachable(fitted_pp, expected):
    """A predicate is only total if every branch can actually fire. The
    observed span is held at +8.00 pp with a tight interval so the
    straddles-zero branch cannot pre-empt any of these."""
    out = fvo.span_comparison_verdict(_span(fitted_pp, 0.1), _span(8.0, 0.5))
    assert out["verdict"] == expected
    assert out["ratio_fitted_over_observed"] == pytest.approx(fitted_pp / 8.0)


def test_a_straddling_observed_interval_is_UNINFORMATIVE_not_a_shortfall():
    """Gotcha 25b: a ratio against a denominator whose interval includes zero
    ranks noise. The branch must NOT report the model as falling short."""
    out = fvo.span_comparison_verdict(_span(0.5, 0.1), _span(1.0, 5.0))
    assert out["verdict"] == "UNINFORMATIVE_OBSERVED_SPAN_STRADDLES_ZERO"
    assert out["ratio_fitted_over_observed"] is None
    # the DIFFERENCE is still defined and still reported
    assert out["difference_fitted_minus_observed_pp"] == pytest.approx(-0.5)


def test_a_negative_observed_span_does_not_produce_a_positive_ratio():
    """Stratum 1's observed span was -6.09 pp. A naive ratio there would be
    positive whenever the fitted span is also negative, which would read as
    agreement. The straddling branch must catch it, or the sign must."""
    out = fvo.span_comparison_verdict(_span(-3.0, 0.1), _span(-6.0, 7.0))
    assert out["verdict"] == "UNINFORMATIVE_OBSERVED_SPAN_STRADDLES_ZERO"
    assert out["ratio_fitted_over_observed"] is None


def test_the_verdict_is_UNDEFINED_when_a_span_was_never_computed():
    out = fvo.span_comparison_verdict(
        {"status": "UNDEFINED_NO_SEED_HAS_TWO_USABLE_BINS"}, _span(8.0, 0.5))
    assert out["verdict"] == "UNDEFINED_A_SPAN_IS_NOT_COMPUTED"


# ------------------------------------------------------------- BB-P1

def test_bb_p1_is_total_over_the_ratio_including_the_undefined_case():
    assert fvo.score_bb_p1(
        fvo.span_comparison_verdict(_span(2.0, 0.1), _span(8.0, 0.5))
    )["outcome"] == "UNDER_HALF"
    assert fvo.score_bb_p1(
        fvo.span_comparison_verdict(_span(6.0, 0.1), _span(8.0, 0.5))
    )["outcome"] == "HALF_OR_ABOVE"
    assert fvo.score_bb_p1(
        fvo.span_comparison_verdict(_span(1.0, 0.1), _span(1.0, 5.0))
    )["outcome"].startswith("UNDEFINED")


def test_bb_p1_records_that_it_is_not_tallied():
    """BB1 adjudicated it NOT SEALED. If that standing silently changed, the
    ledger would gain an entry nobody filed."""
    out = fvo.score_bb_p1(
        fvo.span_comparison_verdict(_span(2.0, 0.1), _span(8.0, 0.5)))
    assert "NOT SEALED" in out["standing"]
    assert "NOT TALLIED" in out["standing"]


# ------------------------------------------------------- the formatter

def test_the_formatter_renders_an_absent_span_rather_than_raising():
    """BA17 item 3: a formatter that could not render a legitimately UNDEFINED
    outcome turned a completed measurement into a FAILED job."""
    assert fvo._fmt(None) == "n/a"
    assert fvo._fmt("UNDEFINED") == "n/a"
    assert fvo._fmt(1.5) == "+1.50"
    assert fvo._fmt(True) == "n/a"       # a bool is not a span
