"""
Parse residue-level feature annotations from the Swiss-Prot flat file (.dat).

These become the `g_f` labels of the headline test: a feature is
"annotation-aligned" if it predicts some annotated residue set well (AP), and an
"orphan" otherwise. So a parsing error here does not crash anything -- it
silently relabels residues and corrupts every AP number downstream. The three
traps, all handled explicitly and unit-tested:

1. **DISULFID and CROSSLNK are PAIRS, not RANGES.** `FT DISULFID 26..84` means
   residues 26 AND 84 are bonded to each other -- it does NOT mean residues
   26-84 are all disulfide residues. Treating it as a range silently labels 57
   extra residues here, and produces a perfectly plausible-looking label set.
   Same for CROSSLNK. This is the single most dangerous line in the module.

2. **UniProt positions are 1-BASED INCLUSIVE; cache rows are 0-BASED.** Row i of
   a protein's block in the activation cache is residue i (0-based), so a
   UniProt position p maps to row (p - 1). Off by one and every AP is garbage.

3. **Uncertain positions are SKIPPED, not guessed.** `<1`, `>100`, `?`, `?24`
   all mean the boundary is unknown. Guessing would mislabel residues; they are
   dropped and counted so the count is auditable.

Truncation (proteins longer than max_len) is handled by the caller against the
cache index -- annotations past the cached length must be dropped, not wrapped.
"""

from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator, NamedTuple

# ---------------------------------------------------------------------------
# EVIDENCE PROVENANCE. A Swiss-Prot annotation is not automatically an
# observation. ECO codes distinguish what actually generated the label, and the
# distinction is load-bearing for this project: a "by similarity" ACT_SITE was
# propagated along sequence homology from a characterised relative, and ESM-2 is
# known to encode homology/coevolutionary statistics. A feature "aligning" with
# such a label may be two homology-based methods agreeing with each other, not a
# detector of catalytic chemistry. Propagated labels are also more
# motif-regular than experimental ones (that regularity is what makes them
# propagatable), so they may be EASIER to align with for reasons unrelated to
# biology -- which would inflate g_f on that subset specifically and contaminate
# the quantity the headline test conditions on.
# ---------------------------------------------------------------------------
ECO_TIERS: dict[str, str] = {
    "0000269": "experimental",             # experimental evidence, manual assertion
    "0007744": "experimental_highthroughput",   # combinatorial (e.g. MS proteomics)
    "0000244": "experimental_highthroughput",   # older combinatorial code
    "0000305": "curator_inference",        # curator reasoning, manual assertion
    "0000250": "by_similarity",            # PROPAGATED from a homolog
    "0000255": "predicted",                # match to a sequence model / program
    "0000256": "predicted",                # ditto, automatic assertion
    "0000259": "predicted",                # InterPro signature match
    "0000312": "imported",                 # imported from another database
    "0000313": "imported",
}

# Most-direct-evidence-wins, so each positive residue lands in exactly ONE tier
# (disjoint strata that sum to the concept total -- required for clean
# stratified reporting).
TIER_PRIORITY: tuple[str, ...] = (
    "experimental",
    "experimental_highthroughput",
    "curator_inference",
    "by_similarity",
    "predicted",
    "imported",
    "none",
)

# Coarse grouping used for the pre-registered stratified headline. 'other' is
# reported but is deliberately in NEITHER headline stratum -- curator inference
# is a manual assertion but not an observation, and 'none' is unknown provenance.
TIER_GROUPS: dict[str, str] = {
    "experimental": "experimental",
    "experimental_highthroughput": "experimental",
    "by_similarity": "propagated",
    "predicted": "propagated",
    "imported": "propagated",
    "curator_inference": "other",
    "none": "other",
}

_ECO_RE = re.compile(r"ECO:(\d{7})")


class Feature(NamedTuple):
    ftype: str
    start: int
    end: int
    ligand: str | None
    eco: tuple[str, ...]      # raw 7-digit ECO codes found on this feature
    note: str | None = None   # /note qualifier, e.g. "Protein kinase" -- default
                              # keeps existing 5-positional-arg Feature(...) calls
                              # (tests, and this module's own construction) valid


class Record(NamedTuple):
    """One Swiss-Prot entry. `organism` and `lineage` support the stratum
    COMPOSITION report: experimental annotation is not randomly distributed --
    it concentrates in heavily-studied, crystallisable, model-organism proteins
    -- so the experimental and propagated strata differ in protein population,
    not only in evidence quality. Reporting composition is what makes that
    difference visible instead of confounding."""
    acc: str
    seq_len: int
    feats: list[Feature]
    organism: str | None
    lineage: str | None       # top-level lineage token, e.g. "Eukaryota"


def eco_to_tier(eco_codes) -> str:
    """Most direct evidence tier among the codes present ('none' if empty or
    all unrecognised)."""
    tiers = {ECO_TIERS[c] for c in eco_codes if c in ECO_TIERS}
    for tier in TIER_PRIORITY:
        if tier in tiers:
            return tier
    return "none"

# start..end denotes TWO bonded residues, not an inclusive span. See trap 1.
PAIRED_RESIDUE_TYPES = frozenset({"DISULFID", "CROSSLNK"})

# THE SUB-TYPED FEATURE TYPES LIVE IN THE CONFIG, NOT HERE (amendment CR).
# `annotations.subtype_note_types` in configs/mvp.yaml names the feature
# types whose /note qualifier names a family/kind specific enough to be its
# own concept -- e.g. "DOMAIN 25..289 /note=\"Protein kinase\"" becomes
# concept "DOMAIN:Protein kinase", the same pattern already used for
# "BINDING:ATP" via /ligand. PREREGISTRATION amendment E: pooled type-level
# concepts are structurally hostile to a feature that specifically detects
# ONE family, since its positive set includes every OTHER family the feature
# never touches and recall collapses. InterPLM evaluates at this granularity
# (256 DOMAIN concepts, 55 REGION, 37 MOTIF, 15 ZN_FING) rather than pooled
# types.
#
# Until CR this module carried a four-type default (`SUBTYPE_NOTE_TYPES`,
# DOMAIN/REGION/MOTIF/ZN_FING) that governed SILENTLY whenever the config key
# was absent, while the config listed eight. There is no default now: every
# caller passes the set it read from the config, and `build_residue_labels`
# refuses to run without the key. Excluded by policy, whatever the config
# says: DISULFID/CROSSLNK are paired residues, not spans (a /note there
# describes a specific bond, not a reusable family -- see
# PAIRED_RESIDUE_TYPES).


class Occurrence(NamedTuple):
    """One FT record's span, BEFORE flattening to individual residues --
    preserves instance identity for (a) the disjunctive retention floor's
    instance count and (b) a domain-level-recall metric (D8), both of which
    need to know how many residues make up ONE occurrence, not just which
    residues are positive at all."""
    start: int
    end: int
    tier: str

# Feature types that are not residue-level biology (provenance/variation/etc.)
# and would only add noise as "concepts".
SKIP_TYPES = frozenset({
    "CHAIN", "PEPTIDE", "UNSURE", "CONFLICT", "VAR_SEQ", "VARIANT",
    "MUTAGEN", "NON_CONS", "NON_TER", "NON_STD", "INIT_MET",
})


def parse_position(field: str) -> tuple[int, int] | None:
    """UniProt FT position field -> (start, end) 1-based inclusive, or None if
    the position is uncertain/unparseable (trap 3)."""
    if not field:
        return None
    tok = field.split()[0]                      # drop any trailing qualifier text
    if not tok:
        return None
    if any(c in tok for c in "<>?"):            # uncertain boundary -- do not guess
        return None
    a, _, b = tok.partition("..")
    if not b:
        b = a
    try:
        start, end = int(a), int(b)
    except ValueError:
        return None
    if start < 1 or end < start:
        return None
    return start, end


def residues_for_feature(ftype: str, start: int, end: int) -> list[int]:
    """1-based residue positions covered by one FT record.

    PAIRED types yield exactly the two bonded residues; everything else is an
    inclusive span. This is trap 1 -- do not "simplify" it to a range().
    """
    if ftype in PAIRED_RESIDUE_TYPES:
        return [start] if start == end else [start, end]
    return list(range(start, end + 1))


def iter_dat_records(
    path: Path, keep: set[str] | None = None
) -> Iterator[Record]:
    """Stream a (optionally gzipped) Swiss-Prot .dat file.

    Yields Record(acc, seq_len, feats, organism, lineage). Records whose primary accession
    is not in `keep` are skipped without being materialised, so the eval split
    can be pulled out of the full 3 GB file in one pass.

    Qualifier CONTINUATION matters here: a /evidence value routinely wraps over
    several FT lines, e.g.
        FT                   /evidence="ECO:0000269|PubMed:8663313,
        FT                   ECO:0000269|PubMed:9020152"
    so all continuation text for a feature is accumulated and the ECO codes are
    extracted from the whole blob. Reading only the first line would silently
    drop evidence and misclassify the label's provenance.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    acc: str | None = None
    seq_len = 0
    feats: list[Feature] = []
    organism: str | None = None
    lineage: str | None = None
    cur: tuple[str, int, int] | None = None      # feature awaiting its qualifiers
    cur_ligand: str | None = None
    cur_note: str | None = None
    cur_qual: list[str] = []                     # ALL continuation text

    def flush_current():
        nonlocal cur, cur_ligand, cur_note, cur_qual
        if cur is not None:
            eco = tuple(dict.fromkeys(_ECO_RE.findall(" ".join(cur_qual))))
            feats.append(Feature(cur[0], cur[1], cur[2], cur_ligand, eco, cur_note))
        cur, cur_ligand, cur_note, cur_qual = None, None, None, []

    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "ID":
                # "ID   001R_FRG3G   Reviewed;   256 AA."
                parts = line.split()
                seq_len = int(parts[-2]) if len(parts) >= 2 and parts[-1] == "AA." else 0
            elif tag == "AC":
                if acc is None:                  # primary accession = first listed
                    first = line[5:].strip().split(";")[0].strip()
                    if first:
                        acc = first
            elif tag == "OS":
                if organism is None:             # first OS line only
                    organism = line[5:].strip().rstrip(".")
            elif tag == "OC":
                if lineage is None:              # top-level kingdom/domain token
                    lineage = line[5:].strip().split(";")[0].strip().rstrip(".")
            elif tag == "FT":
                body = line[2:].rstrip()
                stripped = body.strip()
                if not stripped:
                    continue
                # A new feature puts its TYPE in columns 5-20; qualifier lines and
                # their wrapped continuations are indented past that. Testing the
                # type column (rather than a leading "/") is what lets a wrapped
                # /evidence value be distinguished from the next feature.
                is_continuation = line[5:21].strip() == "" if len(line) > 5 else True
                if is_continuation:
                    if cur is not None:
                        cur_qual.append(stripped)
                        if stripped.startswith("/ligand=") and cur_ligand is None:
                            # first /ligand wins; /ligand_id must not overwrite it
                            cur_ligand = stripped.split("=", 1)[1].strip().strip('"')
                        elif stripped.startswith("/note=") and cur_note is None:
                            cur_note = stripped.split("=", 1)[1].strip().strip('"')
                    continue
                flush_current()                  # a new feature starts here
                head, _, rest = stripped.partition(" ")
                if head in SKIP_TYPES:
                    continue
                pos = parse_position(rest.strip())
                if pos is None:
                    continue
                cur = (head, pos[0], pos[1])
            elif tag == "//":
                flush_current()
                if acc is not None and (keep is None or acc in keep):
                    yield Record(acc, seq_len, feats, organism, lineage)
                acc, seq_len, feats = None, 0, []
                organism, lineage = None, None
    flush_current()
    if acc is not None and (keep is None or acc in keep):
        yield Record(acc, seq_len, feats, organism, lineage)


_TRAILING_ORDINAL_RE = re.compile(r"\s+\d+$")


def normalize_subtype_note(note: str) -> str:
    """Strip a trailing POSITIONAL ordinal from a /note family name --
    'C2H2-type 1' -> 'C2H2-type', 'ABC transporter 2' -> 'ABC transporter'.

    This is UniProt's own convention for numbering multiple instances of the
    SAME family within one protein (a protein with three C2H2 zinc fingers
    gets '..type 1', '..type 2', '..type 3'), not three distinct families.
    Found by running the first sub-typed rebuild, not by anticipating it:
    ZN_FING fragmented into 17 separate 'C2H2-type N' concepts where InterPLM
    reports one 'C2H2-type' family -- exactly defeating the point of
    sub-typing (a general C2H2 detector would score against only ~1/17th of
    its true positive set). Only a trailing bare integer is stripped, so
    'ABC transmembrane type-1 1' -> 'ABC transmembrane type-1' (the embedded
    '-1' is part of the name; only the final ' 1' ordinal is removed).

    Semicolon-qualified variants ('RING-type; degenerate', 'RING-type;
    atypical') are DELIBERATELY NOT touched -- those describe a structurally
    different variant (a degenerate zinc finger lacks expected residues), not
    a position, and are kept as distinct concepts.
    """
    return _TRAILING_ORDINAL_RE.sub("", note)


def features_to_concept_occurrences(
    feats: list[Feature],
    use_ligand_subconcepts: bool = True,
    *,
    subtype_note_types: frozenset[str],
) -> dict[str, list[Occurrence]]:
    """Features -> {concept: [Occurrence, ...]}, ONE occurrence per FT record,
    NOT yet flattened to residues -- this is what preserves instance identity.

    Emits the type-level concept (e.g. "DOMAIN"), the /ligand sub-concept when
    present (e.g. "BINDING:ATP"), and -- for `subtype_note_types` -- the /note
    sub-concept (e.g. "DOMAIN:Protein kinase"). All three are separate entries
    in the returned dict (a DOMAIN occurrence appears under both "DOMAIN" and
    "DOMAIN:Protein kinase"), same pattern as the existing ligand mechanism.
    Which ones are actually tested downstream is a separate choice.

    A `subtype_note_types` occurrence with NO usable /note (missing, or empty
    after `normalize_subtype_note` strips a bare ordinal) goes to
    "<TYPE>:unspecified", not silently only into the pooled parent. Checked
    against the real label store (residue_labels_subtyped.h5): the pooled
    parent's positive set was found to be the union of its sub-concepts PLUS
    a genuine, non-trivial residual (7-15% of the parent's residues, varying
    by type) -- i.e. neither "the parent is pure duplication of its children"
    nor "the parent has no /note-labelled children" was correct on its own.
    ":unspecified" makes that residual its own real, disjoint concept
    (InterPLM's "any" category), so it can be a legitimate classification
    target rather than disappearing into a parent that is mostly redundant
    with its own children.
    """
    out: dict[str, list[Occurrence]] = defaultdict(list)
    for f in feats:
        tier = eco_to_tier(f.eco)
        occ = Occurrence(f.start, f.end, tier)
        out[f.ftype].append(occ)
        if use_ligand_subconcepts and f.ligand:
            out[f"{f.ftype}:{f.ligand}"].append(occ)
        if f.ftype in subtype_note_types:
            family = normalize_subtype_note(f.note) if f.note else ""
            out[f"{f.ftype}:{family}" if family else f"{f.ftype}:unspecified"].append(occ)
    return dict(out)


def occurrences_to_concept_positions(
    occurrences: dict[str, list[Occurrence]],
) -> dict[str, dict[int, str]]:
    """{concept: [Occurrence,...]} -> {concept: {1-based residue: evidence_tier}}.

    Each residue carries the MOST DIRECT evidence tier among all occurrences of
    that concept covering it (TIER_PRIORITY), so tiers are DISJOINT and sum to
    the concept total -- required for a clean stratified partition. The concept
    NAME's prefix before ':' recovers the underlying FT type (e.g. "DOMAIN" from
    "DOMAIN:Protein kinase") so PAIRED_RESIDUE_TYPES (trap 1) is honoured for
    sub-concepts exactly as for type-level concepts.
    """
    rank = {t: i for i, t in enumerate(TIER_PRIORITY)}
    out: dict[str, dict[int, str]] = {}
    for concept, occs in occurrences.items():
        ftype = concept.split(":", 1)[0]
        d: dict[int, str] = {}
        for o in occs:
            for r in residues_for_feature(ftype, o.start, o.end):
                prev = d.get(r)
                if prev is None or rank[o.tier] < rank[prev]:
                    d[r] = o.tier
        out[concept] = d
    return out


def features_to_concept_positions(
    feats: list[Feature],
    use_ligand_subconcepts: bool = True,
    *,
    subtype_note_types: frozenset[str],
) -> dict[str, dict[int, str]]:
    """Features -> {concept: {1-based residue: evidence_tier}}.

    Thin wrapper over `features_to_concept_occurrences` +
    `occurrences_to_concept_positions`, kept for existing callers that only
    need residue positions (not instance boundaries).
    """
    return occurrences_to_concept_positions(
        features_to_concept_occurrences(
            feats, use_ligand_subconcepts,
            subtype_note_types=subtype_note_types))
