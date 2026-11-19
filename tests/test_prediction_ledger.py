"""Tests for the derived prediction ledger (amendment AX §2c)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.prediction_ledger import derive, format_line, load

ANCHOR = {
    "after_amendment": "AV",
    "source": "test",
    "totals": {"within-system": [23, 33], "distributional shape": [17, 18]},
}


def _doc(preds):
    return {"anchor": ANCHOR, "predictions": preds}


def test_a_miss_moves_the_denominator_only():
    got = derive(_doc([{"id": "X-P1", "amendment": "AW",
                        "category": "within-system", "outcome": "MISS"}]))
    assert got["totals"]["within-system"] == [23, 34]


def test_a_hit_moves_both():
    got = derive(_doc([{"id": "X-P1", "amendment": "AW",
                        "category": "within-system", "outcome": "HIT"}]))
    assert got["totals"]["within-system"] == [24, 34]


def test_the_real_aw_arithmetic_gives_23_35_and_17_19_not_23_36_and_17_20():
    """The founding instance: AW scored three predictions and its posted line
    added five to the denominators."""
    got = derive(_doc([
        {"id": "AW-P2", "amendment": "AW",
         "category": "distributional shape", "outcome": "MISS"},
        {"id": "AW-P3", "amendment": "AW",
         "category": "within-system", "outcome": "MISS"},
        {"id": "AW-P4", "amendment": "AW",
         "category": "within-system", "outcome": "MISS"},
    ]))
    assert got["totals"]["within-system"] == [23, 35]
    assert got["totals"]["distributional shape"] == [17, 19]
    # the posted values, asserted to DIFFER so the fixture discriminates
    assert got["totals"]["within-system"] != [23, 36]
    assert got["totals"]["distributional shape"] != [17, 20]


def test_a_not_scored_prediction_moves_nothing_but_is_reported():
    got = derive(_doc([{"id": "AW-P1", "amendment": "AW",
                        "category": "distributional shape",
                        "outcome": "NOT_SCORED"}]))
    assert got["totals"]["distributional shape"] == [17, 18]
    assert [e["id"] for e in got["excluded_from_tallies"]] == ["AW-P1"]


def test_an_unmeasurable_prediction_moves_nothing_but_is_reported():
    """§CT5.1-R: a declared UNMEASURABLE branch is neither a hit nor a miss."""
    got = derive(_doc([{"id": "CT5-P1", "amendment": "CT",
                        "category": "within-system", "scored_in": "CT",
                        "outcome": "UNMEASURABLE"}]))
    assert got["totals"]["within-system"] == [23, 33]
    row = got["excluded_from_tallies"][0]
    assert row["outcome"] == "UNMEASURABLE" and row["category"] == "within-system"
    assert got["not_yet_scored"] == []


def test_a_not_sealed_prediction_moves_nothing_but_is_reported():
    got = derive(_doc([{"id": "AX-P1", "amendment": "AX",
                        "category": "distributional shape",
                        "outcome": "NOT_SEALED"}]))
    assert got["totals"]["distributional shape"] == [17, 18]
    assert got["excluded_from_tallies"][0]["outcome"] == "NOT_SEALED"


def test_an_unscored_prediction_is_listed_rather_than_skipped_silently():
    got = derive(_doc([{"id": "AX-P2", "amendment": "AX",
                        "category": "within-system", "outcome": None}]))
    assert got["not_yet_scored"] == ["AX-P2"]
    assert got["totals"]["within-system"] == [23, 33]


def test_an_unrecognised_outcome_is_refused_not_skipped():
    """A miss must not be able to leave the ledger by typo."""
    with pytest.raises(SystemExit, match="neither a scoring outcome"):
        derive(_doc([{"id": "X", "amendment": "AW",
                      "category": "within-system", "outcome": "PARTIAL"}]))


def test_a_category_the_anchor_does_not_carry_is_refused():
    with pytest.raises(SystemExit, match="anchor does not carry"):
        derive(_doc([{"id": "X", "amendment": "AW",
                      "category": "vibes", "outcome": "HIT"}]))


def test_through_filters_on_where_a_prediction_was_SCORED_not_filed():
    """AW-P1 was filed in AW, explicitly not computed there, and scored in AX.
    Filtering on the filing amendment would put it inside AW's recomputed line
    and make that line disagree with the posted one for a reason unrelated to
    AW's arithmetic -- hiding the defect the recomputation exists to expose."""
    doc = _doc([{"id": "AW-P1", "amendment": "AW", "scored_in": "AX",
                 "category": "distributional shape", "outcome": "HIT"}])
    assert derive(doc, "AW")["totals"]["distributional shape"] == [17, 18]
    assert derive(doc, "AX")["totals"]["distributional shape"] == [18, 19]


def test_scored_in_defaults_to_the_filing_amendment():
    doc = _doc([{"id": "X", "amendment": "AW",
                 "category": "within-system", "outcome": "MISS"}])
    assert derive(doc, "AW")["totals"]["within-system"] == [23, 34]


def test_through_recomputes_an_earlier_amendments_line():
    doc = _doc([
        {"id": "AW-P3", "amendment": "AW",
         "category": "within-system", "outcome": "MISS"},
        {"id": "AX-P2", "amendment": "AX",
         "category": "within-system", "outcome": "HIT"},
    ])
    assert derive(doc, "AW")["totals"]["within-system"] == [23, 34]
    assert derive(doc, "AX")["totals"]["within-system"] == [24, 35]


def test_format_line_orders_the_reliable_bins_first():
    line = format_line({"distributional shape": [17, 19],
                        "within-system": [23, 35]})
    assert line.startswith("within-system 23/35")
    assert "distributional shape 17/19" in line


# ------------------------------------------------- the shipped ledger file --

def test_the_shipped_ledger_loads_and_derives():
    """The CURRENT line. This assertion moves every time a prediction is
    scored, which is the point -- it is a canary against an accidental edit to
    the file, not a historical fact. The two `--through` tests below pin the
    facts that must NOT move."""
    doc = load(Path("docs/prediction_ledger.json"))
    got = derive(doc)
    assert got["anchor_after"] == "AV"
    # after CS (2026-09-17): CS-P1 HIT and CS-P2 MISS, both within-system, so
    # 24/38 -> 25/40. The pre-CS line is pinned separately below.
    # after CT3-R (2026-10-21): CT3-P1 HIT, within-system, so 25/40 -> 26/41.
    # after CT4-R (2026-10-22): CT4-P1 MISS and CT4-P2 MISS, so 26/41 -> 26/43.
    # after CT5.1-R: CT5-P1 UNMEASURABLE (a declared branch, non-scoring), so the
    # line stays 26/43 and nothing is left unscored.
    # after CT-pre2 (filing): seven predictions filed with outcome null.
    # after CT14-R: CT14-P1 HIT, within-system, so 26/43 -> 27/44.
    # after CT15-R: CT15-P1 HIT, within-system, so 27/44 -> 28/45.
    # after CT19-R: CT19-P1 HIT, within-system, so 28/45 -> 29/46.
    # after CT20-R: CT20-P1 MISS and CT20-P2 HIT, so 29/46 -> 30/48.
    # after CT10-R: CT10-P1 HIT and CT10-P2 HIT, so 30/48 -> 32/50; nothing unscored.
    # after CT22 (filing, 2026-10-23): CT22-P1 and CT22-P2 filed with outcome null.
    # after CT22.4-R: CT22-P1 HIT and CT22-P2 HIT, so 32/50 -> 34/52; nothing unscored.
    # after CT23-pre (filing, 2026-10-23): nine predictions of CT24-CT27 filed with
    # outcome null; the tallies do not move.
    # after CT27-R: CT27-P1 MISS, within-system, so 34/52 -> 34/53.
    # after CT26-R: CT26-P1/P2/P3 HIT, so 34/53 -> 37/56. CT24-P1 is measured but held
    # pending its null-degenerate check (CT24-R1).
    # after CT24-R2 and CT25-R: CT24-P1, CT24-P2, CT25-P1, CT25-R1 MISS, so 37/56 -> 37/60.
    # after CT24-R3: CT24-P3 HIT, so 37/60 -> 38/61; nothing unscored.
    # after CT28-pre (filing, 2026-10-24): CT28-P1/P2/P3/P5 and CT29-P1-P4 filed with
    # outcome null; CT28-P4 filed NOT_SEALED (excluded); the tallies do not move.
    # after CT28-R: CT28-P1/P2/P3/P5 MISS, so 38/61 -> 38/65; CT28-P4 measured, untallied.
    # after CT29-R: CT29-P1 HIT, so 38/65 -> 39/66; CT29-P2/P3/P4 UNMEASURABLE (arms (iii)
    # and (iv), CT5.1 cause 3), untallied; nothing unscored.
    assert got["totals"]["within-system"] == [39, 66]
    assert got["totals"]["distributional shape"] == [19, 21]
    assert got["not_yet_scored"] == []
    excluded = {e["id"]: e["outcome"] for e in got["excluded_from_tallies"]}
    assert excluded["CT28-P4"] == "NOT_SEALED"
    assert excluded["CT5-P1"] == "UNMEASURABLE"
    assert [excluded[i] for i in ("CT29-P2", "CT29-P3", "CT29-P4")] == ["UNMEASURABLE"] * 3


def test_the_shipped_ledger_reproduces_the_pre_CT_line_through_CS():
    """CS's line (25/40) must not move when CT's predictions are scored."""
    doc = load(Path("docs/prediction_ledger.json"))
    got = derive(doc, "CS")
    assert got["totals"]["within-system"] == [25, 40]


def test_the_shipped_ledger_reproduces_the_pre_CS_line_through_CQ():
    """The line as it stood from AZ through CQ (24/38 within-system): CS's two
    scored predictions must not leak backwards through `--through`."""
    doc = load(Path("docs/prediction_ledger.json"))
    got = derive(doc, "CQ")
    assert got["totals"]["within-system"] == [24, 38]
    assert got["totals"]["distributional shape"] == [19, 21]


def test_the_shipped_ledger_reproduces_AXs_posted_line_unchanged_by_later_entries():
    """AX's line is a published number. Adding AY's predictions must not move
    it -- `--through` filters on where a prediction was SCORED, so a later
    amendment's entries cannot leak backwards."""
    doc = load(Path("docs/prediction_ledger.json"))
    got = derive(doc, "AX")
    assert got["totals"]["within-system"] == [24, 36]
    assert got["totals"]["distributional shape"] == [18, 20]


def test_the_shipped_ledger_reproduces_the_corrected_aw_line():
    doc = load(Path("docs/prediction_ledger.json"))
    got = derive(doc, "AW")
    assert got["totals"]["within-system"] == [23, 35]
    assert got["totals"]["distributional shape"] == [17, 19]


def test_every_shipped_entry_has_the_fields_the_deriver_reads():
    doc = json.loads(Path("docs/prediction_ledger.json").read_text())
    for p in doc["predictions"]:
        assert set(p) >= {"id", "amendment", "category", "outcome"}
        assert p["category"] in doc["anchor"]["totals"]


def test_the_untraced_span_is_declared_rather_than_reconstructed():
    """An untraced entry is useful; a plausible-looking reconstruction is worse
    than a gap. The file must SAY which span it does not itemise."""
    doc = json.loads(Path("docs/prediction_ledger.json").read_text())
    assert "_untraced" in doc and "UNTRACED" in doc["_untraced"]


# --------------------------------- STRUCTURAL_CHECK (amendment CA-b, CA-c) --
# The author's ratification at CA-a excludes BI3(a) from every tally on the ground
# that its scored predicate is deductive. The mechanism must be shown to
# EXCLUDE (the tally moves) and to NOT exclude everything (a positive control),
# because a mechanism that excluded everything would produce the same number
# here and could only pass.


def test_a_structural_check_moves_neither_half_of_its_tally():
    got = derive(_doc([{"id": "X-P1", "amendment": "BT",
                        "category": "within-system",
                        "outcome": "STRUCTURAL_CHECK",
                        "check_verdict": "HELD"}]))
    assert got["totals"]["within-system"] == [23, 33]


def test_a_structural_check_is_distinguishable_from_a_hit_AND_from_a_miss():
    """The whole point of the new value: excluded is not retracted, and it is
    not a MISS either. Both alternatives are asserted to give DIFFERENT totals,
    so the fixture discriminates rather than merely passing."""
    entry = {"id": "X-P1", "amendment": "BT", "category": "within-system"}
    excluded = derive(_doc([{**entry, "outcome": "STRUCTURAL_CHECK",
                             "check_verdict": "HELD"}]))
    as_hit = derive(_doc([{**entry, "outcome": "HIT"}]))
    as_miss = derive(_doc([{**entry, "outcome": "MISS"}]))
    assert excluded["totals"]["within-system"] == [23, 33]
    assert as_hit["totals"]["within-system"] == [24, 34]
    assert as_miss["totals"]["within-system"] == [23, 34]
    assert excluded["totals"] != as_hit["totals"]
    assert excluded["totals"] != as_miss["totals"]


def test_a_structural_check_keeps_its_category_and_its_verdict_on_the_row():
    """An absent field is not the same as a considered one, and a reader
    landing on the row must see that the check HELD without following a
    cross-reference."""
    got = derive(_doc([{"id": "BI3(a)", "amendment": "BI", "scored_in": "BT",
                        "category": "procedure-behaviour",
                        "outcome": "STRUCTURAL_CHECK",
                        "check_verdict": "HELD"}]))
    row = got["excluded_from_tallies"][0]
    assert row["id"] == "BI3(a)"
    assert row["outcome"] == "STRUCTURAL_CHECK"
    assert row["category"] == "procedure-behaviour"
    assert row["check_verdict"] == "HELD"
    assert "DEDUCTIVE" in row["reason"]


def test_a_structural_check_is_reported_never_dropped():
    got = derive(_doc([{"id": "X-P1", "amendment": "BT",
                        "category": "within-system",
                        "outcome": "STRUCTURAL_CHECK"}]))
    assert [e["id"] for e in got["excluded_from_tallies"]] == ["X-P1"]
    assert got["counted"] == []


def test_POSITIVE_CONTROL_excluding_one_entry_leaves_its_neighbours_tallied():
    """CA-c's positive control. A mechanism that excluded EVERY entry in the
    category would produce the same 1/2 -> 0/1 movement, so the discriminating
    assertion is that the OTHER two entries still tally."""
    preds = [
        {"id": "KEEP-HIT", "amendment": "BT", "category": "procedure-behaviour",
         "outcome": "HIT"},
        {"id": "KEEP-MISS", "amendment": "BT", "category": "procedure-behaviour",
         "outcome": "MISS"},
        {"id": "DROP", "amendment": "BT", "category": "procedure-behaviour",
         "outcome": "STRUCTURAL_CHECK", "check_verdict": "HELD"},
    ]
    anchor = {"after_amendment": "AV", "source": "test",
              "totals": {"procedure-behaviour": [4, 6]}}
    got = derive({"anchor": anchor, "predictions": preds})
    # the excluded one contributed nothing ...
    assert got["totals"]["procedure-behaviour"] == [5, 8]
    # ... and BOTH survivors did, which is what rules out "excludes everything"
    assert sorted(c["id"] for c in got["counted"]) == ["KEEP-HIT", "KEEP-MISS"]
    # and had DROP been left a HIT the answer would have differed
    as_hit = derive({"anchor": anchor,
                     "predictions": [{**p, "outcome": "HIT"} if p["id"] == "DROP"
                                     else p for p in preds]})
    assert as_hit["totals"]["procedure-behaviour"] == [6, 9]


def test_the_shipped_BI3a_entry_is_excluded_and_records_that_the_check_HELD():
    doc = load(Path("docs/prediction_ledger.json"))
    entry = next(p for p in doc["predictions"] if p["id"] == "BI3(a)")
    assert entry["outcome"] == "STRUCTURAL_CHECK"
    assert entry["check_verdict"] == "HELD"          # held, not retracted
    assert entry["previous_outcome"] == "HIT"        # archived, not deleted
    assert entry["category"] == "procedure-behaviour"
    got = derive(doc)
    assert "BI3(a)" not in [c["id"] for c in got["counted"]]
    row = next(e for e in got["excluded_from_tallies"] if e["id"] == "BI3(a)")
    assert row["check_verdict"] == "HELD"


def test_the_shipped_procedure_behaviour_bin_is_9_of_13_after_the_exclusion():
    """WRITTEN DOWN, not computed from the ledger it checks (gotcha 32b). The
    baseline measured before the ratification was 10/14; excluding one HIT
    removes one from each half. Both numbers are asserted so the test fails if
    the exclusion silently stops working."""
    # `through="CA"` (amendment CQ): the pin is the line AS SHIPPED AT CA. CQ
    # later scored CQ-P1 as a procedure-behaviour HIT, which legitimately
    # moves the unrestricted bin to 10/14 -- the same digits as the pre-CA
    # baseline -- so the unrestricted comparison would no longer distinguish
    # "exclusion working, one more HIT" from "exclusion broken". The post-CQ
    # pin below asserts the new line WITH the exclusion still in force.
    got = derive(load(Path("docs/prediction_ledger.json")), through="CA")
    assert got["totals"]["procedure-behaviour"] == [9, 13]
    assert got["totals"]["procedure-behaviour"] != [10, 14]


def test_the_ledger_after_CQ_is_10_of_14_with_BI3a_still_excluded():
    """WRITTEN DOWN (gotcha 32b): CQ-P1 is one procedure-behaviour HIT on the
    CA line of 9/13. The bin reads 10/14 ONLY because of that HIT -- BI3(a)
    must still be excluded, CQ-P2 excluded as a STRUCTURAL_CHECK that DIFFERED,
    and CQ-P3 excluded as NOT_SEALED."""
    got = derive(load(Path("docs/prediction_ledger.json")))
    assert got["totals"]["procedure-behaviour"] == [10, 14]
    counted = {r["id"]: r["outcome"] for r in got["counted"]}
    assert counted["CQ-P1"] == "HIT"
    excluded = {r["id"]: r for r in got["excluded_from_tallies"]}
    assert excluded["BI3(a)"]["outcome"] == "STRUCTURAL_CHECK"
    assert excluded["CQ-P2"]["check_verdict"] == "DIFFERED"
    assert excluded["CQ-P3"]["outcome"] == "NOT_SEALED"


def test_the_ratification_moved_no_other_bin():
    """Measured before the change and written down here (scoped to the CA
    line at amendment CQ, which scored nothing in these bins either)."""
    got = derive(load(Path("docs/prediction_ledger.json")), through="CA")
    assert got["totals"]["within-system"] == [24, 38]
    assert got["totals"]["distributional shape"] == [19, 21]
    assert got["totals"]["directional mechanism"] == [2, 10]
    assert got["totals"]["cross-system transfer"] == [0, 2]
    assert got["totals"]["constructive counterexample"] == [1, 1]


def test_BI3b_is_still_tallied_because_its_footprint_is_empirical():
    """CA-d's scope guard, as an executable assertion: the ratification applies
    to BI3(a) ALONE. BI3(b)'s footprint (`gained > 0` at some step) is a fact
    about data, not a deduction, so it must still carry arithmetic."""
    got = derive(load(Path("docs/prediction_ledger.json")))
    assert "BI3(b)" in [c["id"] for c in got["counted"]]
