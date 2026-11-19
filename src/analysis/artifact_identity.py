"""Artifact IDENTITY -- what an artifact IS, not merely that it exists.

WHY THIS MODULE EXISTS. `main_ap.sbatch` guarded its label store with
`[[ ! -f "$LABELS" ]]`. The guard passed. The file it pointed at was the
822-concept, `truncation_policy=with_truncated` store from three rebuilds ago,
would have produced a plausible, wrong AP matrix on the wrong population.

**A missing file fails loudly; a stale-but-present one passes silently.** An
existence check is not an identity check, and a guard satisfied by the wrong
artifact is worse than no guard at all, because it converts a loud failure
into a silent one. Every artifact therefore carries an IDENTITY SET, and every
consumer asserts on the identity, never on file presence.

Four defects in this project share one shape -- a decision landed in one place
and did not propagate to its consumers:
  - the untruncated-counts floor fix, without its companion check;
  - the disjunctive retention floor in `build_residue_labels.py`, while
    `feature_ap.py` went on re-applying the legacy 100-residue floor;
  - `sparse_ap_index.py` built and validated, but never wired into
    `feature_ap.py`;
  - `--labels` pointing three generations stale.
None was catchable by per-component testing, which is why 173 passing tests
caught none of them. They are failures of AGREEMENT BETWEEN components.
`preflight.py` turns that agreement into an assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

# Bump whenever the AP arithmetic changes in a way that could alter output.
# Recorded in every AP artifact and asserted by preflight, so an artifact
# produced by an unvalidated implementation cannot be silently consumed.
AP_IMPL_SEGMENTED = "average_precision_segmented/v1"
AP_IMPL_SPARSE = "sparse_ap_index.average_precision_from_touches/v1"

# ---------------------------------------------------------------------------
# THE DEVICE CLASS (gotcha 36).
#
# The activation gather is DEVICE-DEPENDENT and this is systematic, not noise:
# near-tie pre-activations swap under a CPU-versus-GPU float32 matmul, TopK
# then selects a different feature set, and `fire_counts` for the same
# checkpoint on the same rows come out different. Measured by recomputing the
# AP artifact on CPU and diffing: 9,018 of 204,800 (seed, feature) pairs differ
# (4.40%), and AP differs in 399,305 of 46,489,600 elements. An observed
# statistic and its null must therefore be computed on the same device class.
#
# UNTIL NOW NO ARTIFACT COULD STATE ITS OWN. `write_identity` had no
# gather-device key, so the consistency audit had to INFER every artifact's
# device from its submission script plus log evidence. That inference is the
# best that existed; it is not a measurement, and the difference is exactly the
# kind that is lost first when a label travels.
#
# TWO KEYS, BECAUSE THE TWO DEVICES ARE ROUTED SEPARATELY. `feature_ap.py`
# sends the AP arithmetic to CPU above NNZ_GPU_SAFE while the SAE forward pass
# stays on the GPU, so at this scale a GPU request buys only the gather. One
# key for each, and neither is a substitute for the other.
GATHER_DEVICE_KEY = "gather_device_class"
AP_ARITH_DEVICE_KEY = "ap_arith_device_class"


def device_class(device) -> str:
    """A DEVICE CLASS string for the device ACTUALLY used, not the one
    requested and not the one available.

    Accepts a `torch.device`, a device string, or any tensor/module exposing
    `.device` -- so a caller can hand it the OUTPUT of the gather rather than
    the variable it hoped the gather would run on. A job can request a GPU and
    run the arithmetic on CPU; that is precisely what got four tasks reclaimed.

    CPU-versus-GPU is NOT enough (gotcha 36): the divergence is between device
    CLASSES, so the name and the compute capability are recorded. Everything
    here is READ from what torch exposes and each read is guarded -- an
    attribute this build does not have is recorded as UNKNOWN, never omitted
    and never guessed, because an absent component would make two different
    devices produce the same string.
    """
    import torch

    dev = getattr(device, "device", device)
    dev = dev if isinstance(dev, torch.device) else torch.device(str(dev))

    if dev.type == "cuda":
        try:
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
        except Exception:                                   # noqa: BLE001
            idx = None
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:                                   # noqa: BLE001
            name = "UNKNOWN_NAME"
        try:
            major, minor = torch.cuda.get_device_capability(idx)
            cap = f"sm_{major}{minor}"
        except Exception:                                   # noqa: BLE001
            cap = "UNKNOWN_CAPABILITY"
        return f"cuda:{name}:{cap}"

    if dev.type == "cpu":
        import platform
        return f"cpu:{platform.machine() or 'UNKNOWN_MACHINE'}"

    # A backend nobody here has used. Named rather than coerced into one of the
    # two above -- a device class this function does not understand must not be
    # reported as one it does.
    return f"{dev.type}:UNKNOWN_CLASS"


def device_identity_attrs(record: dict, seen: dict) -> dict:
    """The device attrs to write, collapsing a per-seed `record` against the
    SET of values `seen` across the whole run.

    ONE implementation, shared by every producer: two copies of this collapse
    would make their agreement a statement about the copies (gotcha 16). A key
    that took more than one value across the run is written as
    `MIXED:<sorted values>` rather than as whichever seed finished last --
    an array that ran across two device classes is not an array with one
    device class, and gotcha 36 says the difference reaches the statistic.
    """
    out = {}
    for k, v in record.items():
        vals = sorted(seen.get(k, {v}))
        out[k] = v if len(vals) <= 1 else "MIXED:" + ",".join(vals)
    return out


def compare_device_class(a: dict, b: dict, label_a: str, label_b: str,
                         key: str = GATHER_DEVICE_KEY) -> dict:
    """Compare one device-class key across two identities, TOTALLY.

    **`ABSENT_ON_BOTH` IS NOT AGREEMENT.** Every artifact written before this
    key existed lacks it, so both-absent is the COMMON case rather than an edge
    case, and it is exactly the case in which reporting `AGREE` would be a
    claim about two devices nobody recorded. The verdict names the situation
    and `agree` is False for every verdict except `AGREE` (I3).

    The four verdicts are the complete outcome space over the two presence
    bits, and the reading of each is stated so a caller does not have to infer
    one (gotcha 30): `ABSENT_ON_BOTH` means the devices are UNRECORDED, which
    is not the same as unequal -- it licenses no conclusion in either
    direction.
    """
    va, vb = a.get(key), b.get(key)
    pa, pb = key in a, key in b
    if pa and pb:
        agree = va == vb
        verdict = "AGREE" if agree else "DIFFER"
    elif pa or pb:
        agree, verdict = False, "ABSENT_ON_ONE"
    else:
        agree, verdict = False, "ABSENT_ON_BOTH"
    return {"key": key, "verdict": verdict, "agree": agree,
            label_a: va if pa else "<ABSENT>",
            label_b: vb if pb else "<ABSENT>",
            "reading": {
                "AGREE": "both artifacts recorded the same device class",
                "DIFFER": "recorded, and DIFFERENT -- gotcha 36 says the "
                          "statistic and its null are not comparable",
                "ABSENT_ON_ONE": "one artifact cannot state its device class, "
                                 "so the pair is unverified, not matched",
                "ABSENT_ON_BOTH": "NEITHER artifact records a device class. "
                                  "This is UNRECORDED, not equal -- it "
                                  "licenses no conclusion in either direction",
            }[verdict]}

# The identity set. `library_sha` is the git SHA of the code that BUILT the
# label store (its own attr), which is distinct from the SHA of the code
# consuming it.
#
# EQUALITY KEYS -- compared artifact-to-artifact. Every one of these must
# match the label store exactly.
# ---------------------------------------------------------------------------
# THE EVIDENCE-TIER GROUPING (amendment A1 / AV4).
#
# MOVED HERE 2026-08-11 from `feature_ap.py`, where it was the only copy but
# the wrong home: `preflight.py` now has to derive a stratified artifact's
# expected concept axis from the SAME grouping the producer used, and importing
# `feature_ap` into the gate would drag torch into a launch check. A SECOND
# COPY was the alternative and is exactly the drift gotcha 16 keeps catching --
# one rule, one implementation. `feature_ap.py` re-exports the name, so every
# existing import site is unchanged.
TIER_GROUP_MAP = {
    "experimental": "experimental", "experimental_highthroughput": "experimental",
    "by_similarity": "propagated", "predicted": "propagated",
    "imported": "propagated", "curator_inference": "other", "none": "other",
}


def stratum_concept_axis(labels_path: Path, stratum: str) -> list[str]:
    """The concept axis the LABEL STORE implies a stratified artifact carries.

    THE DEFECT THIS EXISTS FOR. `preflight` asserted an AP artifact's concept
    axis EQUALS the store's. That is right for a pooled artifact and wrong for
    a stratified one: on the experimental stratum only the concepts with at
    least one experimentally-evidenced residue are scored at all, so the
    experimental artifact's axis is a 227-concept SUBSET of the store's 712 and
    the gate refused it PERMANENTLY, for a correct reason. AV14 reported that
    and did not patch it.

    A check that always fails cannot detect anything -- a genuine defect in the
    same pair would be one more line in an already-red report. That is the
    coverage illusion, and it is worse than an unguarded artifact because it
    looks like protection.

    THE FIX IS NOT AN EXEMPTION. Adding the axis to a provenance exemption list
    would also wave through a GENUINE axis mismatch, reproducing the same
    problem one level down. Instead the EXPECTED RELATIONSHIP is encoded: a
    stratified artifact's axis must be exactly the store's concepts having at
    least one row in that stratum's tiers -- derived from the store's own
    per-tier datasets, by the same `TIER_GROUP_MAP` the producer applies. That
    is STRICTLY STRONGER than equality-with-exemption: a wrong subset still
    fails, and so does a right-sized subset containing the wrong concepts.

    Cheap by construction: dataset SHAPES only, never contents. A tier is
    non-empty iff its dataset has a non-zero first dimension, and the union of
    the stratum's tiers is non-empty iff any one of them is -- so no row index
    is ever read. Measured at 0.46 s for the 712-concept store.
    """
    if stratum == "pooled":
        raise ValueError(
            "`pooled` is not a stratum subset -- it reads the `all` tier of "
            "every concept, so its axis IS the store's axis. Callers must "
            "compare against the full axis directly rather than deriving one; "
            "returning the full axis here would make the two paths look "
            "interchangeable when only one of them can fail.")
    out = []
    with h5py.File(labels_path, "r") as hf:
        cg = hf["concepts"]
        for name in sorted(cg):
            g = cg[name]
            if any(TIER_GROUP_MAP.get(t) == stratum and g[t].shape[0] > 0
                   for t in g if t != "all"):
                out.append(name)
    return out


def row_subset_concept_axis(labels_path: Path, row_set_file: Path,
                            which: str) -> list[str]:
    """The concept axis the LABEL STORE implies a POOLED artifact scored on a
    row SUBSET carries (amendment CT §CT10): the store's concepts with at least
    one `all`-tier row inside the subset, which is exactly the set
    `feature_ap` scores ("concepts with positives in the subset").

    The same reasoning as `stratum_concept_axis`: an expected RELATIONSHIP, not
    an exemption, so a wrong subset of the right size still fails. Unlike the
    stratified case it must read row CONTENTS, once, into a membership mask.
    Refuses a halves file built against another label store."""
    with h5py.File(row_set_file, "r") as hf:
        rows = hf[f"{which}_rows"][:]
        built = hf.attrs.get("labels", "")
        built = built.decode() if isinstance(built, bytes) else str(built)
    if Path(built).name != Path(labels_path).name:
        raise ValueError(f"{row_set_file} was built against {built}, not {labels_path}")
    mask = np.zeros(int(rows.max()) + 1, dtype=bool)
    mask[rows] = True
    out = []
    with h5py.File(labels_path, "r") as hf:
        cg = hf["concepts"]
        for name in sorted(cg):
            r = cg[name]["all"][:]
            r = r[r < mask.size]
            if r.size and mask[r].any():
                out.append(name)
    return out


def artifact_strata(identity: dict) -> list[str]:
    """The strata an artifact declares, parsed. The attr round-trips through
    `json.dumps` for lists, so `list()` of it is a list of CHARACTERS -- the
    exact failure gotcha 20 names. An artifact that does not declare `strata`
    returns [], reported as UNDECLARED by the caller and never defaulted to
    `pooled`; defaulting is how the stratified case would silently take the
    pooled branch."""
    raw = identity.get("strata")
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        return list(v) if isinstance(v, list) else [v]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def expected_ap_impl(negatives, default: str) -> tuple[str, str]:
    """(expected `ap_impl`, why) DERIVED from whether masking is active.

    The gate compared `ap_impl` against ONE validated constant. Under
    `--negatives unannotated` a concept's non-experimental rows are removed
    from the evaluated universe, which changes the ranks the sparse path
    computes against the full row set -- so `feature_ap.py` routes a masked run
    to the SEGMENTED implementation by construction
    (`use_sparse = not any_masked`). Masking is MANDATORY on the experimental
    stratum (BG1 / AV10b), so the experimental artifact can never carry the
    sparse marker and the gate refused it forever.

    DERIVED, NOT EXEMPTED, and the difference is that this still fails in both
    directions (gotcha 25): a SEGMENTED marker on an unmasked artifact fails,
    and a SPARSE marker on a masked one fails.

    THE ONE BRANCH WHOSE READING IS NOT OBVIOUS, enumerated per gotcha 30
    rather than left implicit: a run declaring `negatives=unannotated` in which
    NO concept turns out to have a mask would legitimately take the sparse path
    and be refused here. That refusal is INFORMATIVE -- it means the masking
    the run declared did nothing -- so it is left as a failure rather than
    softened into an accepted alternative.
    """
    neg = negatives.decode() if isinstance(negatives, bytes) else negatives
    if neg == "unannotated":
        return AP_IMPL_SEGMENTED, (
            "masking is active (`negatives=unannotated`), and a masked "
            "concept's ranks change with the evaluated universe, so the "
            "producer routes to the segmented path by construction")
    return default, "no masking declared"


IDENTITY_KEYS = (
    "truncation_policy",
    "n_concepts",
    "library_sha",
    "retention_floor_mode",
    "retention_min_residues",
    "retention_min_instances",
    "min_protein_count",
)

# ---------------------------------------------------------------------------
# THE PARAMETER / VALUE REGISTRY (amendment AE, condition 2).
#
# Two parameters used to collide twice over: `row_subset`'s value was literally
# `population`, which was the OTHER parameter's NAME, and both accepted the
# shared literal `all` meaning different things. Crossing them made every
# prevalence `n_positives / wrong_n_rows`.
#
# Renaming fixed that instance. THIS TABLE plus its test retires the CLASS:
# `tests/test_parameter_name_value_invariant.py` fails if any value equals any
# parameter name, or if two parameters share a value literal. Every `choices=`
# for these flags derives from here, so a new parameter cannot reintroduce the
# collision in a second list nobody thought to check.
PARAMETER_VALUES: dict[str, tuple[str, ...]] = {
    # label-store property: are proteins truncated at 1022 residues included?
    "truncation_policy": ("with_truncated", "untruncated"),
    # which CACHE ROWS get scored. `label_store` = the row set the label store
    # was itself built against (amendment G2).
    # `ct10_half_a` / `ct10_half_b` (amendment CT §CT10): the two cluster
    # halves of the label_store population, read from a halves FILE
    # (`--row-set-file`, `ct10_halves.py`), never from the label store.
    "scored_row_set": ("shared_experimental", "full_eval", "label_store",
                       "ct10_half_a", "ct10_half_b"),
}

# ---------------------------------------------------------------------------
# THE CANONICAL KEY SET (amendment O / gotcha 2b).
#
# WHY THIS EXISTS. The pre-flight gate's original failure was `argmax_pool_n`:
# CLAUDE.md's gotcha-15 identity set NAMED it, and the code did not CHECK it,
# so the gate passed a 716-pool null against a 703-pool AP matrix -- exactly
# the pair amendment I4 declares incompatible. The documentation was right and
# the code was wrong, and nothing structural prevented them from disagreeing.
#
# So the gate no longer maintains its checks in parallel with a list written
# somewhere else. It ITERATES THIS TABLE. Adding a key here adds a check;
# there is no second place to update and therefore no way for the two to
# drift apart. Divergence becomes structurally impossible rather than a thing
# to remember.
#
# `kind` selects the comparison:
#   "equal_to_labels"  -- must equal the label store's value
#   "derived"          -- must equal a value DERIVED from the label store
#                         (e.g. the argmax pool implied by its concept list)
#   "expected"         -- must equal a value the CALLER declares for this run
#                         (the label store cannot state it: it is a property of
#                         the run, not of the store)
#   "consistent"       -- every consuming artifact must state the SAME value.
#                         Catches a disagreement the label store cannot arbitrate.
#   "must_be_true"     -- boolean flag that must be set
#   "must_be_absent"   -- attr whose PRESENCE is disqualifying
#
# A NOTE ON "equal_to_labels" AND DEAD BRANCHES (added with `scored_row_set`).
# `equal_to_labels` silently does nothing when the label store does not publish
# the key -- `if k in lab_id and ...`. That is deliberate for keys a store may
# predate, and it is a TRAP for keys a store never writes at all: the check
# exists, reads as coverage, and cannot fire. `library_sha` was exactly that
# (the store publishes it as `git_sha`), so it is now "derived" instead. When
# adding a key here, establish which of the two it is; if no store writes it,
# "equal_to_labels" is the wrong kind.
#
# `applies_to` restricts a key to artifact kinds ("null", "ap"); None = all.
# `required` means ABSENT IS NOT MATCHING (I3): a missing key is a failure,
# never silent agreement.

# `library_sha` is NOT compared artifact-to-store by equality: the store
# publishes the SHA of the code that built it under `git_sha`, never under
# `library_sha`. It is therefore DERIVED (see the note above).
_EQUALITY_KEYS = tuple(k for k in IDENTITY_KEYS if k != "library_sha")

CANONICAL_IDENTITY_CHECKS: tuple[dict, ...] = tuple(
    [{"key": k, "kind": "equal_to_labels", "applies_to": None, "required": True}
     for k in _EQUALITY_KEYS]
    + [
        {"key": "library_sha", "kind": "derived", "applies_to": None,
         "required": True, "derived_from": "git_sha",
         "why": "the SHA of the code that BUILT the label store. The store "
                "publishes it as `git_sha`, so an `equal_to_labels` check on "
                "`library_sha` can never fire -- it was dead for the gate's "
                "whole life, and both compliant artifacts carried the SHA of "
                "the PREVIOUS store while the gate passed."},
        {"key": "scored_row_set", "kind": "expected", "applies_to": None,
         "required": True,
         "why": "THE OVERLOADING AMENDMENT AE RETIRED. `truncation_policy` is of the "
                "LABEL STORE (with_truncated|untruncated); `scored_row_set` picks CACHE "
                "ROWS (full_eval|shared_experimental|label_store). Before AE "
                "the latter's value was spelled `population` -- the former's "
                "NAME -- and both shared the literal `all`. So agreement on one "
                "says nothing about the other: `--scored-row-set full_eval` against "
                "the untruncated store scores 17,739,363 rows against a "
                "15,972,325-row store, and every prevalence -- hence every "
                "normalised gain -- comes out wrong."},
        {"key": "scored_row_set", "kind": "consistent", "applies_to": None,
         "required": True,
         "why": "and the artifacts must agree with EACH OTHER, not only with "
                "the declared expectation -- an AP matrix and a null scored "
                "over different rows are not comparable however each was "
                "declared."},
        {"key": "n_rows", "kind": "consistent", "applies_to": None,
         "required": True,
         "why": "the CONSEQUENCE of a scored_row_set mismatch, checked "
                "independently of its cause. A run that mis-declared its "
                "scored_row_set attr still cannot fake the row count it scored."},
        {"key": "n_rows", "kind": "derived", "applies_to": None,
         "required": True, "derived_from": "n_rows",
         "why": "and under `--scored-row-set label_store` the row count must equal "
                "the label store's own -- the direct, arithmetic form of the "
                "overloading guard. NOT APPLICABLE for other scored-row-sets, "
                "where it is reported as not-applicable rather than passed."},
        {"key": "argmax_pool_n", "kind": "derived", "applies_to": None,
         "required": True,
         "why": "amendment I4 -- a max over a different pool is a null for a "
                "DIFFERENT STATISTIC. `n_concepts` cannot catch this: it is "
                "the LIBRARY size and is identical for both pools."},
        # REQUIRED since amendment CT, §CT8 item 2: absent on a null was a
        # silent pass, which gotcha 15 forbids. Every null of record carries
        # it `true` (alignment-pass dump, 2026-10-04).
        {"key": "exclude_pooled_parents", "kind": "must_be_true",
         "applies_to": ("null",), "required": True,
         "why": "a null that let pooled parents win its max is a null for a "
                "statistic g_f does not compute (I4)."},
        {"key": "superseded_by", "kind": "must_be_absent", "applies_to": None,
         "required": False,
         "why": "an artifact that declares itself superseded must never be "
                "consumed, however well its other keys line up."},
        {"key": "superseded_for_axis", "kind": "must_be_absent",
         "applies_to": None, "required": False,
         "why": "amendment N11 -- reductions valid only for the pool they "
                "were drawn on (null_g_* is a max OVER concepts, so it "
                "changes when the pool changes)."},
    ]
)


# ---------------------------------------------------------------------------
# THE WRITER SIDE (amendment Q, defect 9).
#
# The canonical table made the READER complete: every key it lists gets
# checked. That closed "documented but unchecked" and it did catch
# `scored_row_set`. It does NOT close the defect it was written after.
#
# `library_sha` was an `equal_to_labels` key, and NO LABEL STORE WRITES IT --
# stores publish the building code's SHA as `git_sha`. So reader-side coverage
# was complete while the key remained effectively unchecked: the comparison
# read `if k in lab_id and ...`, the key was never in `lab_id`, and the branch
# could not fail. **That is gotcha 13 arriving inside the guard built to
# enforce gotcha 13** -- a check that cannot fail is not a check, and it reads
# as coverage in every audit.
#
# The missing assertion is about the SCHEMA, not about any pair of artifacts:
# the key set the gate READS must be reconciled against the key set the
# producers WRITE. Two rules now do that:
#
#   1. `LABEL_STORE_KEYS` -- every `equal_to_labels` key must be one a label
#      store actually publishes. Asserted against `build_residue_labels.py`'s
#      source in the test suite, so a key that no builder writes cannot be
#      added as an equality check.
#   2. `derived_from` -- a `derived` check names the STORE key its expectation
#      is computed from, and `preflight` hard-fails if the store does not
#      publish it. Absent-source is a schema defect, NOT a not-applicable
#      check; conflating them is how the dead branch survived.
#
# Keys a label store publishes and the gate may compare against directly.
LABEL_STORE_KEYS = frozenset({
    "truncation_policy", "n_concepts", "retention_floor_mode",
    "retention_min_residues", "retention_min_instances", "min_protein_count",
    "n_rows", "git_sha", "git_dirty", "provenance_excluded_types",
    "min_positive_residues",
})


def unwritable_equality_keys() -> list[str]:
    """`equal_to_labels` keys that no label store publishes -- i.e. checks
    that cannot fire. Empty is the only acceptable answer; the test suite
    asserts that and names `library_sha` as the founding case."""
    return sorted({c["key"] for c in CANONICAL_IDENTITY_CHECKS
                   if c["kind"] == "equal_to_labels"} - LABEL_STORE_KEYS)


def derived_sources() -> dict[str, str]:
    """Canonical key -> the label-store key its expectation is derived from.
    Keys derived from the store's CONTENT rather than an attr are absent."""
    return {c["key"]: c["derived_from"] for c in CANONICAL_IDENTITY_CHECKS
            if c["kind"] == "derived" and c.get("derived_from")}


def pooled_parents(concepts) -> set[str]:
    """Concepts that are a POOLED PARENT of sub-typed children.

    A bare FT-type name (no ':') is a pooled parent iff at least one
    `<TYPE>:<subtype>` concept exists in the same library. Derived from the
    concept list alone, so it works on stores built before this module
    existed and needs no extra attr to be trustworthy.

    Measured on the 716-concept gate-5 library: 9 parents -- BINDING, DOMAIN,
    MOTIF, REGION, REPEAT, SIGNAL, TOPO_DOM, TRANSIT, ZN_FING -- each 83.5% to
    99.8% covered by its own children's rows. `ACT_SITE`, `DISULFID`,
    `TRANSMEM` and the other 16 childless bare concepts are NOT parents; they
    are genuinely un-subtyped single concepts and stay in the pool.
    """
    cs = list(concepts)
    bare = [c for c in cs if ":" not in c]
    prefixed = {c.split(":", 1)[0] for c in cs if ":" in c}
    return {c for c in bare if c in prefixed}


def argmax_pool_mask(concepts) -> list[bool]:
    """True for concepts eligible to WIN the argmax that defines `g_f`.

    PREREGISTRATION amendment I4: pooled parents are retained in the library
    and in the AP matrix as a separately-reported diagnostic arm, but are
    EXCLUDED from the argmax pool. Excluding a concept from the FDR family is
    not the same as excluding it from a MAXIMUM -- `g_f` is a max, so a
    pooled parent left in the pool can still win it, and D5's
    breadth-dominance problem would be unfixed by construction.

    The mask MUST be applied identically by every path that takes a max over
    concepts (the real `g_f` and its null), or the null ranges over a
    different concept set than the statistic it is a null for -- the same
    defect the T6 check caught, in a different variable.
    """
    par = pooled_parents(concepts)
    return [c not in par for c in concepts]


def write_identity(hf, **kw) -> None:
    """Write the identity set into an open h5py File's attrs. Unknown keys are
    allowed (e.g. `ap_impl`, `argmax_pool_n`); missing ones are simply absent
    and will be reported by `assert_identity_match` rather than defaulted."""
    for k, v in kw.items():
        if v is None:
            continue
        hf.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v


def read_identity(path: Path, extra: tuple[str, ...] = ()) -> dict:
    """Identity set of an artifact, as a plain dict. Missing keys are absent
    rather than defaulted -- a defaulted identity is how a mismatch hides."""
    out = {}
    with h5py.File(path, "r") as hf:
        for k in IDENTITY_KEYS + tuple(extra):
            if k in hf.attrs:
                v = hf.attrs[k]
                out[k] = v.decode() if isinstance(v, bytes) else v
    return out


def assert_identity_match(a: dict, b: dict, label_a: str, label_b: str,
                          keys: tuple[str, ...] = IDENTITY_KEYS) -> None:
    """Raise unless every key present in BOTH identities agrees.

    Keys absent from one side are reported as a WARNING by the caller, not
    silently skipped -- `null_pilot_g.h5` shipped with no `truncation_policy` at
    all, and 'absent' must never read as 'matching'.
    """
    bad = []
    for k in keys:
        if k in a and k in b and a[k] != b[k]:
            bad.append(f"  {k}: {label_a}={a[k]!r}  vs  {label_b}={b[k]!r}")
    if bad:
        raise ValueError(
            f"ARTIFACT IDENTITY MISMATCH between {label_a} and {label_b}:\n"
            + "\n".join(bad)
            + "\n\nThese artifacts describe different experiments. Combining "
              "them would produce a plausible, wrong result -- refusing."
        )


# ---------------------------------------------------------------------------
# EVAL-CACHE IDENTITY (amendment CQ, A4). Assert the VALUES, not the presence.
# ---------------------------------------------------------------------------
# `verify_cache.py` checked that `layer` and `model` were PRESENT in the cache's
# attrs, and nothing in the gather path (`feature_ap.sparse_activations`,
# `row_index_cache.compute_or_load_row_index`) read them at all -- so a
# layer-23 cache, or one built with the final layer norm applied, would have
# been gathered silently (DATAFLOW.md section 2.5, "relied on but not
# asserted"). These are the four attrs `cache.py` writes that a consumer's
# result depends on; `dtype`, `subsampled` and `seed` describe the file, not
# the model that produced the rows.
EVAL_CACHE_IDENTITY_KEYS = ("model", "layer", "final_layer_norm_applied",
                            "max_len")

#: Sentinel for an absent attr in a report row. Never compares equal to a value.
EVAL_CACHE_ABSENT = "<ABSENT>"


class EvalCacheIdentityMismatch(ValueError):
    """The eval cache's identity attrs do not equal what the consumer expects.

    Raised, never downgraded to a warning: a consumer that continued would
    gather activations from a layer or a model other than the one every
    downstream artifact says it used, and every number after it would be
    plausible and wrong. An ABSENT key on either side is a mismatch (gotcha
    15) -- absence is not agreement.
    """


def _plain(v):
    """h5 attrs come back as numpy scalars and bytes; expectations come from
    YAML or JSON as Python scalars. Compare on the plain values."""
    if isinstance(v, bytes):
        return v.decode()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except Exception:                                   # noqa: BLE001
            return v
    return v


def expected_eval_cache_attrs_from_config(cfg: dict) -> dict:
    """The four expected values, read from the config's `model` section.

    `final_layer_norm_applied` is the cache's name for the config's
    `apply_final_layer_norm`; the two are the same fact under two names.
    """
    m = cfg["model"]
    return {
        "model": str(m["name"]),
        "layer": int(m["layer"]),
        "final_layer_norm_applied": bool(m["apply_final_layer_norm"]),
        "max_len": int(m["max_len"]),
    }


def eval_cache_identity(cache_path) -> dict:
    """The cache's own values for the identity keys. A key the file does not
    carry is simply absent from the result -- never defaulted."""
    out = {}
    with h5py.File(cache_path, "r") as hf:
        for k in EVAL_CACHE_IDENTITY_KEYS:
            if k in hf.attrs:
                out[k] = _plain(hf.attrs[k])
    return out


def assert_eval_cache_identity(cache_path, expected: dict, label: str,
                               keys: tuple[str, ...] = EVAL_CACHE_IDENTITY_KEYS,
                               ) -> dict:
    """Assert the cache's identity attrs EQUAL `expected` on every key.

    `label` names where `expected` came from (the config, a checkpoint's
    `activations_attrs`), so the failure message names both sides. Returns a
    per-key report on success; raises `EvalCacheIdentityMismatch` naming every
    disagreeing key with both values otherwise. Absent on either side is a
    mismatch.
    """
    actual = eval_cache_identity(cache_path)
    per_key, bad = {}, []
    for k in keys:
        a = actual.get(k, EVAL_CACHE_ABSENT)
        e = _plain(expected[k]) if k in expected else EVAL_CACHE_ABSENT
        ok = (k in actual) and (k in expected) and a == e
        per_key[k] = {"cache": a, label: e, "agree": bool(ok)}
        if not ok:
            bad.append(f"  {k}: cache={a!r}  vs  {label}={e!r}")
    if bad:
        raise EvalCacheIdentityMismatch(
            f"EVAL-CACHE IDENTITY MISMATCH between {cache_path} and {label}:\n"
            + "\n".join(bad)
            + "\n\nThe cache was built from a different model, layer, length "
              "cap or normalisation than this consumer expects. Gathering from "
              "it would produce plausible, wrong activations -- refusing.")
    return {"cache": str(cache_path), "reference": label, "per_key": per_key,
            "verdict": "AGREE"}
