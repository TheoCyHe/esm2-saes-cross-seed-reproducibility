"""Cross-component consistency assertion -- runs BEFORE any main job and
refuses to launch on mismatch.

THE FAILURE CLASS THIS EXISTS FOR. Four defects in this project share one
shape: a decision landed in one place and did not propagate to its consumers.
The untruncated-counts floor fix without its companion check; the disjunctive
retention floor applied in the builder while `feature_ap.py` re-applied the
legacy 100-residue floor; the restructured sparse AP path built, validated,
and never wired in; `--labels` pointing three generations stale. Every one was
invisible to per-component testing -- which is exactly why 173 passing unit
tests caught none of them. They are failures of AGREEMENT BETWEEN components.

So this is not another unit test. It is a launch gate that reads the actual
artifacts a job is about to consume and refuses if they disagree.

WHAT IT CHECKS
  1. concept counts agree: label store == null artifact == the AP path's
     post-filter count (the T6 defect: 716 vs 611)
  2. `truncation_policy` agrees across every input (the --labels defect: untruncated
     vs all)
  3. floor parameters agree: retention mode, both disjunctive thresholds,
     min_protein_count
  4. the AP implementation carries a version marker matching the validated one
  5. `git_dirty` is False, and the `--porcelain` output is logged when it is
     not, so the REASON is visible rather than just the boolean
  6. `scored_row_set` -- the OVERLOADING AMENDMENT AE RETIRED. It was
     `row_subset` with a value literally named `population`, while
     property; `scored_row_set` selects cache rows and its literal value
     `population` was a label-store property. Checked against the run's
     declared expectation AND across artifacts, plus `n_rows` as the
     arithmetic consequence, so a mismatch is refused even when `truncation_policy`
     agrees on every input -- which it always will, since both artifacts copy
     it from the same store.
  7a. the CONCEPT AXIS, as an expected RELATIONSHIP rather than equality
     (2026-08-11). A pooled artifact's axis is the store's; a STRATIFIED
     artifact's is the store's concepts having a row in that stratum's tiers.
     Equality made the experimental pair fail PERMANENTLY for a correct reason,
     and a check that always fails detects nothing -- the coverage illusion.
     Encoded as the relationship, not exempted, so a wrong subset still fails.
  7b. the NULL's axis against the AP artifact's: exactly the argmax-pool
     subset, POSITIONALLY (gotcha 27). Nothing checked this before -- the
     null's axis was never read, which is not "checked and equal".
  7c. `ap_impl` DERIVED from whether masking is active, rather than compared
     against one constant. Masking is mandatory on the experimental stratum
     (BG1), which forces the segmented path, so the constant refused that
     artifact forever.
  8. `library_sha` -- as a DERIVED check against the store's `git_sha`. It had
     been an `equal_to_labels` check on a key no label store writes, so it
     could not fire; both compliant artifacts carried the previous store's SHA
     while the gate passed.

Per gotcha 13, every check here is verified to FAIL against the condition it
claims to detect -- see `tests/test_preflight.py`, which constructs a
deliberate violation of each one.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import h5py

from src.analysis.artifact_identity import (
    AP_ARITH_DEVICE_KEY,
    AP_IMPL_SPARSE,
    CANONICAL_IDENTITY_CHECKS,
    GATHER_DEVICE_KEY,
    IDENTITY_KEYS,
    argmax_pool_mask,
    artifact_strata,
    compare_device_class,
    derived_sources,
    expected_ap_impl,
    read_identity,
    row_subset_concept_axis,
    stratum_concept_axis,
    unwritable_equality_keys,
)


class PreflightError(RuntimeError):
    """Raised when components disagree. Never caught -- it stops a launch."""


def git_porcelain(repo_root: Path | None = None) -> str:
    """`git status --porcelain`, verbatim. Logged rather than reduced to a
    boolean: `git_dirty` counting UNTRACKED files is CORRECT (an untracked
    .py can shadow an import and change what executes), but a bare True is
    unactionable -- the reason has to be visible."""
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root,
            stderr=subprocess.DEVNULL).decode()
    except Exception:                                         # noqa: BLE001
        return ""


def concept_names(path: Path, group: str | None = None) -> list[str]:
    """Concept axis of a label store, an AP artifact, or a null artifact.

    Three layouts, and the third was added 2026-08-11: the null writes its
    concept axis as a ROOT-LEVEL dataset with no enclosing group, so the
    old branch fell through to `sorted(subgroups)[0]` and raised IndexError on
    a file that has none. It was never reached because `check` only called this
    for AP artifacts -- which is precisely why the null's axis went unchecked.
    """
    with h5py.File(path, "r") as hf:
        if "concepts" in hf and isinstance(hf["concepts"], h5py.Group):
            return sorted(hf["concepts"])                      # label store
        if group is None and "concepts" in hf:                 # null artifact
            return [c.decode() if isinstance(c, bytes) else c
                    for c in hf["concepts"][:]]
        subs = sorted(k for k in hf if isinstance(hf[k], h5py.Group))
        if group is None and not subs:
            return []
        g = hf[group] if group else hf[subs[0]]
        return [c.decode() if isinstance(c, bytes) else c
                for c in g["concepts"][:]]


def check(
    labels: Path,
    null_artifact: Path | None = None,
    ap_artifacts: tuple[Path, ...] = (),
    expect_ap_impl: str = AP_IMPL_SPARSE,
    require_clean_git: bool = True,
    repo_root: Path | None = None,
    config_excluded_types: tuple[str, ...] | None = None,
    expect_scored_row_set: str | None = "label_store",
    accept_ap_arith_split: bool = False,
) -> dict:
    """Run every cross-component check. Raises `PreflightError` on the first
    failing family, listing ALL failures in it. Returns a summary dict."""
    problems: list[str] = []
    summary: dict = {}

    if not labels.exists():
        raise PreflightError(f"label store does not exist: {labels}")

    lab_id = read_identity(labels)
    # The store's own extras, needed to DERIVE expectations it does not state
    # under the consumer's key name (`git_sha` -> `library_sha`) and to check
    # the row count a `--scored-row-set population` run must reproduce.
    lab_id_all = read_identity(labels, extra=("git_sha", "n_rows"))
    lab_concepts = concept_names(labels)
    summary["labels"] = str(labels)
    summary["label_identity"] = dict(lab_id)
    summary["n_concepts_label_store"] = len(lab_concepts)

    # ---- 5. git cleanliness (checked first: it invalidates everything else)
    porcelain = git_porcelain(repo_root)
    summary["git_porcelain"] = porcelain
    if require_clean_git and porcelain.strip():
        problems.append(
            "git_dirty is True -- the working tree has uncommitted or "
            "UNTRACKED files, so any artifact written from it records dirty "
            "provenance. `git status --porcelain` says:\n"
            + "".join(f"      {ln}\n" for ln in porcelain.splitlines())
            + "    (untracked counts, and correctly so: an untracked .py can "
              "shadow an import and change what executes)")

    # The argmax pool the LABEL STORE implies. Every artifact that takes a
    # max -- or that defines the pool a consumer will max over -- must agree
    # with this, not merely with `n_concepts`.
    expected_pool_n = sum(argmax_pool_mask(lab_concepts))

    # ---- SPEC agreement, not just cross-component agreement (M2) ---------
    # Every other check here compares artifacts TO EACH OTHER. That cannot
    # catch a specification violation that propagated consistently: HELIX,
    # STRAND, TURN and COMPBIAS are excluded by PREREGISTRATION section 1 for
    # provenance, but nothing enforced it, so all four sat in the library, the
    # AP matrix AND the null -- in perfect agreement, and all wrong. Three of
    # them ranked among the ten broadest concepts in the argmax pool.
    #
    # So this check compares the store against the CONFIG's declared exclusion
    # rather than against a peer artifact.
    if config_excluded_types is not None:
        declared = lab_id.get("provenance_excluded_types")
        if isinstance(declared, bytes):
            declared = declared.decode()
        present = sorted({c.split(":", 1)[0] for c in lab_concepts}
                         & set(config_excluded_types))
        summary["provenance_excluded_declared"] = declared
        summary["provenance_excluded_present_in_store"] = present
        if present:
            problems.append(
                f"{labels.name}: concept types {present} are present in the "
                "label store but the config excludes them for PROVENANCE "
                "(PREREGISTRATION section 1). Cross-component agreement does "
                "not imply agreement with the specification -- every artifact "
                "can agree on a contaminated axis. Rebuild the store with the "
                "exclusion applied (amendment M2).")
        if declared is None:
            summary["provenance_excluded_declared"] = (
                "ABSENT -- store predates the M2 exclusion attr")

    # ---- 1/2/3. identity agreement across every input
    others: list[tuple[str, Path, list[str]]] = []
    if null_artifact is not None:
        if not null_artifact.exists():
            problems.append(f"null artifact does not exist: {null_artifact}")
        else:
            # The null's axis is now READ. It used to be `[]` -- not "checked
            # and equal", simply never looked at -- so the one relationship
            # that ties a null to the statistic it is a null FOR was unchecked
            # by construction.
            others.append(("null", null_artifact, concept_names(null_artifact)))
    for p in ap_artifacts:
        if not p.exists():
            problems.append(f"AP artifact does not exist: {p}")
        else:
            others.append(("ap", p, concept_names(p)))

    # DERIVED expectations -- values the label store IMPLIES rather than
    # states. Keyed by the canonical table's `key`.
    # A `derived` expectation of None means NOT DERIVABLE for this run. It is
    # reported as not-applicable rather than passed -- the I3 rule ("absent is
    # not matching") applied to the expectation side.
    derived_expectations: dict = {
        "argmax_pool_n": expected_pool_n,
        # The store publishes the building code's SHA as `git_sha`; artifacts
        # copy it into `library_sha`. Comparing the two is the only way this
        # key can ever be checked -- see the note in artifact_identity.
        "library_sha": lab_id_all.get("git_sha"),
        # Only under `--scored-row-set population` does the scored row count have
        # to equal the store's own.
        "n_rows": (lab_id_all.get("n_rows")
                   if expect_scored_row_set == "label_store" else None),
    }
    # ---- SCHEMA ASSERTION: the keys the gate READS must be keys the
    # producers WRITE (amendment Q, defect 9). `library_sha` was an
    # `equal_to_labels` check against a key no label store publishes, so the
    # comparison `if k in lab_id and ...` could never fire. Reader-side
    # coverage was complete and the key was effectively unchecked -- gotcha 13
    # inside the guard built to enforce gotcha 13.
    #
    # Two rules, both hard failures:
    #   (a) no `equal_to_labels` key may be one no store writes;
    #   (b) a `derived` check's SOURCE key must be present on this store.
    # (b) is deliberately distinguished from a not-applicable check: absent
    # source is a SCHEMA defect, and conflating the two is how the dead branch
    # survived.
    unwritable = unwritable_equality_keys()
    if unwritable:
        problems.append(
            f"canonical table declares {unwritable} as `equal_to_labels`, but "
            "no label store writes those keys -- the comparison can never "
            "fire. A check that cannot fail is not a check; it reads as "
            "coverage in every audit (P8b).")
    for key, src in derived_sources().items():
        if src not in lab_id_all:
            problems.append(
                f"the label store does not publish `{src}`, which the "
                f"`{key}` check derives its expectation from -- so that check "
                "cannot fire. This is a SCHEMA mismatch between what the gate "
                "reads and what the builder writes, not a not-applicable "
                "check (P8b).")

    expected_values: dict = {"scored_row_set": expect_scored_row_set}
    summary["expect_scored_row_set"] = expect_scored_row_set
    not_applicable: list[str] = []
    consistency: dict[str, list[tuple[str, object]]] = {}

    extra_keys = tuple(sorted({c["key"] for c in CANONICAL_IDENTITY_CHECKS}
                              | {"ap_impl", "scored_row_set", "strata",
                                 "negatives", "row_set_file",
                                 GATHER_DEVICE_KEY, AP_ARITH_DEVICE_KEY}))
    device_ids: list[tuple[str, dict]] = []

    for kind, p, cs in others:
        oid = read_identity(p, extra=extra_keys)
        device_ids.append((p.name, oid))

        # ---- DRIVEN BY THE CANONICAL TABLE, not by hand-maintained branches
        # (amendment O, gotcha 2b). The gate's founding failure was a key that
        # CLAUDE.md's identity set named and the code did not check. Iterating
        # one table means adding a key adds a check, with no second place to
        # update and therefore no way for the two to drift.
        for chk in CANONICAL_IDENTITY_CHECKS:
            k, ckind = chk["key"], chk["kind"]
            if chk["applies_to"] is not None and kind not in chk["applies_to"]:
                continue
            why = chk.get("why", "")
            suffix = f" ({why})" if why else ""
            present = k in oid

            if ckind == "must_be_absent":
                if present and oid.get(k):
                    problems.append(
                        f"{p.name}: carries `{k}`={oid[k]!r} -- refusing to "
                        f"consume it.{suffix}")
                continue

            if not present:
                if chk["required"]:
                    # Wording is load-bearing: the tests assert these exact
                    # substrings as the documented contract, so a refactor
                    # cannot quietly change what the gate promises to say.
                    problems.append(
                        f"{p.name}: identity keys ABSENT: ['{k}'] -- "
                        "ABSENT IS NOT MATCHING. An artifact that cannot "
                        "state what it is cannot be asserted against one "
                        f"that can (I3).{suffix}")
                continue

            if ckind == "equal_to_labels":
                if k in lab_id and lab_id[k] != oid[k]:
                    problems.append(
                        f"{p.name}: {k}={oid[k]!r} but label store says "
                        f"{lab_id[k]!r}{suffix}")
            elif ckind == "derived":
                want = derived_expectations[k]
                if want is None:
                    not_applicable.append(
                        f"{p.name}: `{k}` not derivable for this run "
                        f"(expect_scored_row_set={expect_scored_row_set!r}) -- "
                        "reported NOT APPLICABLE, not passed.")
                elif str(oid[k]) != str(want):
                    extra = ""
                    if k == "argmax_pool_n":
                        extra = (f" ({len(lab_concepts)} concepts minus "
                                 f"{len(lab_concepts) - expected_pool_n} "
                                 "pooled parents)")
                    problems.append(
                        f"{p.name}: {k}={oid[k]} but the label store "
                        f"implies {want}{extra}.{suffix}")
            elif ckind == "expected":
                want_decl = expected_values[k]
                if want_decl is None:
                    not_applicable.append(
                        f"{p.name}: `{k}` has no declared expectation -- "
                        "reported NOT APPLICABLE, not passed.")
                elif str(oid[k]) != str(want_decl):
                    problems.append(
                        f"{p.name}: {k}={oid[k]!r} but this run declares "
                        f"{want_decl!r}.{suffix}")
            elif ckind == "consistent":
                consistency.setdefault(k, []).append((p.name, oid[k]))
            elif ckind == "must_be_true":
                if not bool(oid[k]):
                    problems.append(
                        f"{p.name}: {k}=False.{suffix}")

        # Concept-count agreement against the store's CONTENT, not its attr --
        # a separate check because `n_concepts` above compares two ATTRS and
        # would pass if both were wrong in the same way.
        n_other = oid.get("n_concepts")
        if n_other is not None and int(n_other) != len(lab_concepts):
            problems.append(
                f"{p.name}: n_concepts={int(n_other)} but the label store has "
                f"{len(lab_concepts)} concepts in its CONTENT -- a max over "
                "different concept sets is not a null for the statistic it "
                "is compared against")
        # ---- CONCEPT AXIS: the EXPECTED RELATIONSHIP, not blunt equality.
        # A pooled artifact's axis IS the store's. A STRATIFIED artifact's is
        # the store's concepts having at least one row in that stratum's
        # tiers -- a subset, permanently and by design. Asserting equality made
        # the gate refuse the experimental pair forever for a correct reason,
        # which is the coverage illusion: a real defect in that pair would have
        # been one more line in an already-red report.
        #
        # Encoded as the relationship rather than exempted, because an
        # exemption would also wave through a GENUINE axis mismatch. This form
        # still fails on a wrong subset and on a right-sized wrong subset.
        if cs and kind == "ap":
            strata = artifact_strata(oid)
            axis_kind = ("pooled" if strata in ([], ["pooled"])
                         else "stratified")
            summary.setdefault("concept_axis", {})[p.name] = {
                "strata": strata or "UNDECLARED", "axis_kind": axis_kind,
                "n_axis": len(cs)}
            rs = oid.get("scored_row_set")
            if axis_kind == "pooled" and rs in ("ct10_half_a", "ct10_half_b"):
                # amendment CT §CT10: a pooled AP on a row SUBSET scores the
                # store's concepts with >= 1 row inside it -- the expected
                # relationship, derived from the halves file it names.
                rsf = oid.get("row_set_file")
                if not rsf:
                    want_axis, why_axis = None, (
                        f"scored_row_set={rs!r} but no `row_set_file` attr names "
                        "the rows")
                else:
                    want_axis = row_subset_concept_axis(labels, Path(str(rsf)), rs)
                    why_axis = f"the store's concepts with >=1 row in {rs}"
            elif axis_kind == "pooled":
                want_axis, why_axis = lab_concepts, "the store's full axis"
            elif len(strata) != 1:
                want_axis, why_axis = None, (
                    f"strata={strata!r} names {len(strata)} strata; the "
                    "expected axis is the UNION over them and no artifact in "
                    "this project has ever written one -- reported NOT "
                    "DERIVABLE rather than guessed")
            else:
                want_axis = stratum_concept_axis(labels, strata[0])
                why_axis = (f"the store's concepts with >=1 row in the "
                            f"{strata[0]!r} tiers")
            if want_axis is None:
                not_applicable.append(
                    f"{p.name}: concept AXIS expectation NOT DERIVABLE "
                    f"({why_axis}) -- reported not-applicable, not passed.")
            elif sorted(cs) != sorted(want_axis):
                only_want = sorted(set(want_axis) - set(cs))[:5]
                only_art = sorted(set(cs) - set(want_axis))[:5]
                problems.append(
                    f"{p.name}: concept AXIS is not {why_axis} "
                    f"({len(cs)} vs {len(want_axis)}). Expected but missing: "
                    f"{only_want}; present but unexpected: {only_art}")

        # 4. AP implementation version marker -- DERIVED from whether masking
        # is active, not compared against one constant. See `expected_ap_impl`.
        if kind == "ap":
            impl = oid.get("ap_impl")
            want_impl, why_impl = expected_ap_impl(
                oid.get("negatives"), expect_ap_impl)
            summary.setdefault("ap_impl_expectation", {})[p.name] = {
                "negatives": oid.get("negatives", "<ABSENT>"),
                "expected": want_impl, "why": why_impl}
            if impl is None:
                problems.append(
                    f"{p.name}: no `ap_impl` marker -- cannot establish which "
                    "AP implementation produced it")
            elif impl != want_impl:
                problems.append(
                    f"{p.name}: ap_impl={impl!r} but the expected "
                    f"implementation is {want_impl!r} -- {why_impl}")

    # ---- THE NULL'S AXIS AGAINST THE AP ARTIFACT'S. NEW 2026-08-11.
    #
    # This relationship is the one that makes a null a null FOR a statistic,
    # and NOTHING checked it: `check` only ever read an AP artifact's axis, so
    # the null's was `[]` -- never looked at, which is not the same as checked
    # and equal (I3).
    #
    # The expected relationship, confirmed from the artifacts BEFORE it was
    # encoded rather than assumed: the null's concept set is exactly the
    # ARGMAX-POOL SUBSET of the AP artifact's axis. `g_f` is a MAX over the
    # pool, so a null that ranges over a different set is a null for a
    # different statistic (amendment I4) -- and the pooled parents are dropped
    # before the null's loop while the AP artifact retains its dense axis.
    #
    # POSITIONAL, not set-wise, and not a count. Two orderings of the same
    # objects with the same total is an invisible collision that no name and no
    # length distinguishes; for a mapping defect only a positional check
    # discriminates (gotcha 27).
    nulls = [(p, cs) for kind, p, cs in others if kind == "null" and cs]
    aps = [(p, cs) for kind, p, cs in others if kind == "ap" and cs]
    axis_pairs = []
    # POOL MEMBERSHIP IS DECIDED AGAINST THE LIBRARY, NEVER AGAINST THE
    # ARTIFACT'S OWN AXIS. `argmax_pool_mask` calls a concept a pooled parent
    # when a sub-typed CHILD exists in the list it is given -- so on a
    # stratified subset a parent whose children happen not to be scored would
    # read as in-pool, and the expectation would be wrong in exactly the
    # direction that lets a real mismatch through. The store is the authority
    # on what a parent is; the artifact's axis only selects which of them are
    # present. (Caught by a fixture whose 3-concept axis did this.)
    lib_pool = {c for c, keep in zip(lab_concepts,
                                     argmax_pool_mask(lab_concepts)) if keep}
    for np_, ncs in nulls:
        for ap_, acs in aps:
            want = [c for c in acs if c in lib_pool]
            ok = list(ncs) == want
            axis_pairs.append({
                "null": np_.name, "ap": ap_.name, "n_null": len(ncs),
                "n_ap": len(acs), "n_ap_in_pool": len(want),
                "verdict": "AGREE" if ok else "DIFFER"})
            if not ok:
                only_want = [c for c in want if c not in set(ncs)][:5]
                only_null = [c for c in ncs if c not in set(want)][:5]
                if sorted(ncs) == sorted(want):
                    first = next(i for i, (a, b) in enumerate(zip(ncs, want))
                                 if a != b)
                    detail = (f"SAME CONCEPTS, DIFFERENT ORDER -- first "
                              f"disagreement at position {first}: null has "
                              f"{ncs[first]!r}, the AP pool has {want[first]!r}")
                else:
                    detail = (f"in the AP artifact's pool but not the null: "
                              f"{only_want}; in the null but not the pool: "
                              f"{only_null}")
                problems.append(
                    f"{np_.name}: concept axis is not the argmax-pool subset "
                    f"of {ap_.name} ({len(ncs)} vs {len(want)} of "
                    f"{len(acs)}). {detail}. A max over a different concept "
                    "set is a null for a DIFFERENT STATISTIC (I4).")
    if axis_pairs:
        summary["null_vs_ap_concept_axis"] = axis_pairs

    # ---- CROSS-ARTIFACT consistency, resolved after every artifact is read.
    # This is the half the label store cannot arbitrate: it never states a
    # `scored_row_set` at all, so agreement with the store is not available and
    # agreement with each other is what the comparison actually requires.
    for k, seen in consistency.items():
        vals = {str(v) for _, v in seen}
        if len(vals) > 1:
            why = next((c.get("why", "") for c in CANONICAL_IDENTITY_CHECKS
                        if c["key"] == k and c["kind"] == "consistent"), "")
            problems.append(
                f"artifacts disagree on `{k}`: "
                + ", ".join(f"{n}={v!r}" for n, v in seen)
                + f". ({why})")
    summary["consistent_keys"] = {k: sorted({str(v) for _, v in s})
                                  for k, s in consistency.items()}

    # ---- DEVICE CLASS (gotcha 36), REPORTED PAIRWISE AND NEVER SILENT.
    #
    # Not a launch blocker, and that is a decision rather than an oversight:
    # every artifact written before the key existed lacks it, so gating on it
    # would refuse every real pair on disk today. What the gate MUST NOT do is
    # let that absence read as agreement -- `compare_device_class` returns
    # `ABSENT_ON_BOTH`, which is UNRECORDED and not equal. A recorded
    # DISAGREEMENT is a different matter and does block: gotcha 36 says an
    # observed statistic and its null computed on different device classes are
    # not comparable, and by then both sides can state what they were.
    # ONE SCOPED EXEMPTION (amendment CT, the author's decision of 2026-10-24,
    # §CT24-F-C3 / §CT25-F-C1): with `accept_ap_arith_split`, a DIFFER on the
    # AP ARITHMETIC's device class is accepted for a pair whose GATHER device
    # class AGREES -- recorded, never silent. The gather stays strict (gotcha
    # 36); an absent gather key, or a gather that differs, still blocks.
    device_report, accepted_splits = [], []
    for i in range(len(device_ids)):
        for j in range(i + 1, len(device_ids)):
            (na, ia), (nb, ib) = device_ids[i], device_ids[j]
            gather = compare_device_class(ia, ib, na, nb, key=GATHER_DEVICE_KEY)
            for key in (GATHER_DEVICE_KEY, AP_ARITH_DEVICE_KEY):
                r = gather if key == GATHER_DEVICE_KEY else compare_device_class(
                    ia, ib, na, nb, key=key)
                device_report.append(r)
                if (r["verdict"] == "DIFFER" and key == AP_ARITH_DEVICE_KEY
                        and accept_ap_arith_split and gather["verdict"] == "AGREE"):
                    accepted_splits.append({"pair": [na, nb], "ap_arith": [r[na], r[nb]],
                                            "gather_agrees": gather[na]})
                    continue
                if r["verdict"] == "DIFFER":
                    problems.append(
                        f"{na} and {nb} disagree on `{key}`: "
                        f"{r[na]!r} vs {r[nb]!r}. Gotcha 36 -- the activation "
                        "gather is device-dependent, so a statistic and its "
                        "null from different device classes are not "
                        "comparable.")
    summary["device_class"] = device_report
    summary["ap_arith_splits_accepted"] = accepted_splits

    # argmax pool, reported so a count change is never silent
    summary["argmax_pool_n"] = expected_pool_n
    summary["pooled_parents_excluded"] = len(lab_concepts) - expected_pool_n
    summary["not_applicable"] = not_applicable

    if problems:
        raise PreflightError(
            "PRE-FLIGHT FAILED -- refusing to launch.\n\n"
            + "\n".join(f"  * {p}" for p in problems)
            + "\n\nThese are cross-component disagreements. Per-component "
              "tests cannot see them; that is what this gate is for.")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--null", type=Path, default=None)
    p.add_argument("--ap", type=Path, nargs="*", default=[])
    p.add_argument("--expect-ap-impl", default=AP_IMPL_SPARSE)
    p.add_argument("--config", type=Path, default=None,
                   help="configs/mvp.yaml. When given, the label store is also "
                        "asserted against the config's PROVENANCE exclusion "
                        "list -- a SPEC check, not a peer-artifact check (M2).")
    p.add_argument("--expect-scored-row-set", default="label_store",
                   help="the CACHE-ROW selection this run requires (all | "
                        "shared_experimental | label_store). Distinct from "
                        "--truncation-policy, a LABEL-STORE property; the "
                        "value `label_store` means 'the store's own "
                        "rows'. Pass the empty string to declare no "
                        "expectation, which is reported as not-applicable.")
    p.add_argument("--allow-dirty", action="store_true",
                   help="Skip the git_dirty gate. For local debugging ONLY -- "
                        "never for a run whose output is kept.")
    args = p.parse_args()

    excluded = None
    if args.config is not None:
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        excluded = tuple(cfg.get("annotations", {})
                            .get("provenance_excluded_types", []))
    s = check(args.labels, args.null, tuple(args.ap),
              expect_ap_impl=args.expect_ap_impl,
              require_clean_git=not args.allow_dirty,
              config_excluded_types=excluded,
              expect_scored_row_set=args.expect_scored_row_set or None)
    print("PRE-FLIGHT PASSED")
    print(f"  label store      : {s['labels']}")
    print(f"  identity         : {s['label_identity']}")
    print(f"  concepts         : {s['n_concepts_label_store']}")
    print(f"  argmax pool      : {s['argmax_pool_n']} "
          f"({s['pooled_parents_excluded']} pooled parents excluded)")
    print(f"  scored_row_set       : {s['expect_scored_row_set']!r} "
          f"(agreed: {s['consistent_keys'].get('scored_row_set')})")
    for na in s["not_applicable"]:
        print(f"  NOT APPLICABLE   : {na}")


if __name__ == "__main__":
    main()
