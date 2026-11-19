"""Derive the prediction ledger instead of maintaining it (amendment AX §2c).

WHY THIS EXISTS. The ledger line at the end of each amendment was INCREMENTED
from the previous amendment's line, and increments drift. Two errors on the same
counter, one session apart:

  * AW7 posted `within-system 23/36, distributional shape 17/20`. AW scored
    exactly three predictions (one shape MISS, two within-system MISSes) against
    an AV baseline of 23/33 and 17/18, so the correct line is **23/35 and
    17/19**. The posted line added five to the denominators where three
    predictions were scored.
  * A separately-proposed correction (15/17 shape, ~23/34 within-system) was
    also wrong, and wrong in a worse way: it came from incrementing a tally
    across conversation turns rather than from the record.

Predictions are filed in git with a category and a total predicate, so the
ledger is DERIVABLE. `docs/prediction_ledger.json` carries a cited anchor plus
one entry per prediction; this module totals them. A derived count cannot drift.

WHAT COUNTS. A MISS moves the denominator only; a HIT moves both. `NOT_SCORED`
(sealed, deliberately not computed) and `NOT_SEALED` (the quantity was already
on disk when the prediction was filed) move NEITHER, and are reported separately
rather than silently dropped -- dropping a prediction from the report is exactly
the channel gotcha 26's companion rule closes, because it would let misses leave
the ledger by attrition.

WHO ASSIGNS THE CATEGORY, AND WHEN -- THE STANDING RULE ADDED AT AMENDMENT BX
(§BX-C2), because §AX2c required a category and never said whose job it was.

    **A prediction's ledger CATEGORY is assigned in the amendment that FILES
    it, in the same subsection as the predicate, before the outcome is known.**
    One of: within-system, distributional shape, procedure-behaviour,
    directional mechanism, cross-system transfer, constructive counterexample.

THE GROUND. This module refuses an entry whose anchor carries no recognised
category -- so between BE and BV, where NO filing amendment assigned one to any
of the five predictions scored in that span, the ledger was unmaintainable by
construction and nobody could have added them without first supplying a field
the record never filled in. Assigning at filing puts the judgement with whoever
knows what the prediction claims, and makes the ledger derivable at any point.

THE AMBIGUITY, RECORDED RATHER THAN RESOLVED. A prediction's CONTENT and its
WARRANT can fall in different bins -- the clearance prediction is a comparison
of two computable quantities by content, warranted by a mechanism argument.
**Categorise by CONTENT; record the WARRANT in the entry's `note`.** The
ambiguity is kept visible because the project's empirical rule rests on the
contrast between the mechanism bin and the comparison bins: filing
mechanism-warranted predictions by content WITHOUT recording the warrant would
keep the mechanism bin artificially small while the comparison bins absorb
mechanism successes, and the rule would then measure the filing convention as
much as the epistemics. One instance does not warrant a new bin.

AND THE OUTPUT OF THIS MODULE IS THE ARTIFACT OF RECORD, NOT ANY LINE POSTED IN
PROSE. Every posted tally is a snapshot dated to the commit it was derived at;
three have gone stale in this repository (corrections entries 103, 148, 155),
and the last of them went stale in a table sitting directly under a warning not
to increment it -- because NOT INCREMENTING IS NOT THE SAME AS RE-DERIVING.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Outcomes that move the tallies, and by how much (hits, denominator).
SCORING = {
    "HIT": (1, 1),
    "MISS": (0, 1),
    "SPLIT": (0, 0),          # scored at both ends; recorded, tallied by hand
}
# Outcomes deliberately outside the tallies. Each is REPORTED, never dropped.
NON_SCORING = {
    "NOT_SCORED": "sealed, deliberately not computed in its own amendment",
    "UNMEASURABLE": "a branch of the prediction's OWN declared outcome space: the "
                    "procedure ran and refused on a cause declared in advance, so "
                    "the quantity was not obtained. Not VOID (the outcome space "
                    "did occur -- this is one of its branches), not NOT_SCORED "
                    "(the computation was attempted). Neither a hit nor a miss; "
                    "the entry's note states the cause and the count "
                    "(amendment CT, §CT5.1-R).",
    "NOT_SEALED": "the quantity was already computable from an artifact on disk "
                  "when the prediction was filed",
    "VOID": "the outcome space the prediction was defined on did not occur "
            "(void by failed premise)",
    "RETURNED": "the prediction was returned UNCOMPUTED because it is "
                "ill-POSED -- not ill-sealed. Distinct from NOT_SEALED, whose "
                "data premise holds and whose quantity was simply already "
                "computable: a RETURNED prediction cannot be scored as written "
                "at all, so filing it as NOT_SEALED would misattribute the "
                "reason and quietly retire a defect (amendment BV-A6). It is "
                "neither a hit nor a miss and enters no tally in either "
                "direction.",
    "STRUCTURAL_CHECK": "the predicate is a DEDUCTIVE consequence of the "
                        "design -- it could not have come out otherwise absent "
                        "an implementation defect -- so by gotcha 44 it is a "
                        "CHECK, never a prediction, and contributes no "
                        "arithmetic to any tally. THE ENTRY'S `check_verdict` "
                        "RECORDS WHAT THE CHECK RETURNED AND IS NOT A "
                        "RETRACTION: a STRUCTURAL_CHECK that HELD held, and its "
                        "refusal branch would still have fired on a defect. "
                        "Distinct from all four values above, and a new value "
                        "rather than a bent one: NOT_SCORED and RETURNED both "
                        "say the quantity was never computed, VOID says the "
                        "outcome space did not occur, and NOT_SEALED says the "
                        "answer was already on disk when the prediction was "
                        "filed -- here the quantity WAS computed, the outcome "
                        "space DID occur, and nothing was on disk. What "
                        "disqualifies it is the predicate's logical status, "
                        "which none of the others describes (amendment CA-b).",
}


def load(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    for key in ("anchor", "predictions"):
        if key not in doc:
            raise SystemExit(f"FATAL: {path} has no {key!r}")
    return doc


def derive(doc: dict, through: str | None = None) -> dict:
    """Totals by category = the cited anchor + every scored prediction after it.

    `through` restricts to predictions SCORED in amendments up to and including
    that letter, so each amendment's posted line can be RECOMPUTED and compared
    -- which is how AW's error is visible without trusting either posted number.

    **Scored-in, not filed-in.** A prediction can be sealed in one amendment and
    scored in a later one (AW-P1 was filed in AW, explicitly not computed there,
    and scored in AX). Filtering on where it was FILED would put AW-P1 inside
    AW's line and make the recomputation disagree with the posted one for a
    reason that has nothing to do with AW's arithmetic error -- which would hide
    the very defect this function exists to expose. `scored_in` defaults to
    `amendment` when absent, which is the common case.
    """
    totals = {k: list(v) for k, v in doc["anchor"]["totals"].items()}
    counted, excluded, unscored = [], [], []
    for p in doc["predictions"]:
        if through is not None and p.get("scored_in", p["amendment"]) > through:
            continue
        outcome = p.get("outcome")
        if outcome is None:
            unscored.append(p["id"])
            continue
        if outcome in NON_SCORING:
            row = {"id": p["id"], "outcome": outcome,
                   "reason": NON_SCORING[outcome]}
            # An excluded entry keeps its CATEGORY and, where the check
            # actually ran, its VERDICT. An absent field is not the same as a
            # considered one, and "excluded from the tally" must never read as
            # "retracted" (amendment CA-b).
            if p.get("category"):
                row["category"] = p["category"]
            if p.get("check_verdict"):
                row["check_verdict"] = p["check_verdict"]
            excluded.append(row)
            continue
        if outcome not in SCORING:
            raise SystemExit(
                f"FATAL: {p['id']} has outcome {outcome!r}, which is neither a "
                f"scoring outcome {sorted(SCORING)} nor a declared non-scoring "
                f"one {sorted(NON_SCORING)}. An unrecognised outcome must not "
                "be silently skipped -- that is how a miss leaves the ledger.")
        cat = p["category"]
        if cat not in totals:
            raise SystemExit(
                f"FATAL: {p['id']} is in category {cat!r}, which the anchor does "
                f"not carry ({sorted(totals)}). A new category needs an anchor "
                "value, not an implicit zero.")
        hit, denom = SCORING[outcome]
        totals[cat][0] += hit
        totals[cat][1] += denom
        counted.append({"id": p["id"], "category": cat, "outcome": outcome})
    return {"anchor_after": doc["anchor"]["after_amendment"],
            "through": through, "totals": totals, "counted": counted,
            "excluded_from_tallies": excluded, "not_yet_scored": unscored}


def format_line(totals: dict) -> str:
    order = ["within-system", "distributional shape", "procedure-behaviour",
             "directional mechanism", "cross-system transfer",
             "constructive counterexample"]
    parts = [f"{k} {totals[k][0]}/{totals[k][1]}" for k in order if k in totals]
    for k in totals:
        if k not in order:
            parts.append(f"{k} {totals[k][0]}/{totals[k][1]}")
    return ", ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", type=Path,
                   default=Path("docs/prediction_ledger.json"))
    p.add_argument("--through", default=None,
                   help="recompute the line as of this amendment letter")
    args = p.parse_args()
    doc = load(args.ledger)
    got = derive(doc, args.through)
    print(f"anchor: after amendment {got['anchor_after']} "
          f"({doc['anchor']['source']})")
    print(f"counted: {len(got['counted'])} prediction(s)")
    for c in got["counted"]:
        print(f"    {c['id']:<8} {c['category']:<24} {c['outcome']}")
    if got["excluded_from_tallies"]:
        print("excluded from the tallies (reported, never dropped):")
        for e in got["excluded_from_tallies"]:
            verdict = (f" [check returned {e['check_verdict']}; "
                       "NOT a retraction]" if e.get("check_verdict") else "")
            print(f"    {e['id']:<8} {e['outcome']:<17} "
                  f"{e.get('category', '-'):<20}{verdict}\n"
                  f"        {e['reason']}")
    if got["not_yet_scored"]:
        print(f"not yet scored: {', '.join(got['not_yet_scored'])}")
    print(f"\nLEDGER: {format_line(got['totals'])}")


if __name__ == "__main__":
    main()
