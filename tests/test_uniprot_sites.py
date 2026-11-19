"""
Tests for Swiss-Prot residue annotation parsing and cache alignment.

Everything here guards a SILENT failure: a mis-parsed position or an off-by-one
does not raise, it just labels the wrong residues and corrupts every downstream
AP number while looking entirely plausible. So the tests pin exact residue and
row indices rather than checking shapes.
"""

from __future__ import annotations

import json

import pytest
import sys

import h5py
import numpy as np

from src.annotations.build_residue_labels import (
    load_cache_index,
    main,
    map_residues_to_rows,
    sanitize_concept_name,
)
from src.annotations.uniprot_sites import (
    Feature,
    Occurrence,
    eco_to_tier,
    features_to_concept_occurrences,
    features_to_concept_positions,
    iter_dat_records,
    normalize_subtype_note,
    occurrences_to_concept_positions,
    parse_position,
    residues_for_feature,
)

# The four-type set the retired code default carried (amendment CR removed the
# default; the expectation is written down here, not read from the code).
SUBTYPES = frozenset({"DOMAIN", "REGION", "MOTIF", "ZN_FING"})


# ---------------------------------------------------------------- positions --

def test_parse_position_single_and_range():
    assert parse_position("102") == (102, 102)
    assert parse_position("51..58") == (51, 58)
    assert parse_position("51..58  /ligand=\"ATP\"") == (51, 58)


def test_parse_position_rejects_uncertain():
    """Uncertain boundaries must be dropped, never guessed."""
    for field in ["<1..20", "21..>100", "?", "?24", "?..100", ""]:
        assert parse_position(field) is None, field


def test_parse_position_rejects_nonsense():
    assert parse_position("abc") is None
    assert parse_position("0") is None            # 1-based: position 0 invalid
    assert parse_position("50..10") is None       # end before start


# ------------------------------------------------------------ paired types --

def test_disulfide_is_a_pair_not_a_range():
    """THE trap: DISULFID 26..84 bonds residues 26 and 84. If this ever returns
    the 59-residue span, every disulfide AP silently becomes garbage."""
    assert residues_for_feature("DISULFID", 26, 84) == [26, 84]
    assert residues_for_feature("CROSSLNK", 10, 200) == [10, 200]
    # a self-referential pair collapses to one residue
    assert residues_for_feature("DISULFID", 26, 26) == [26]


def test_non_paired_types_are_inclusive_ranges():
    assert residues_for_feature("BINDING", 51, 54) == [51, 52, 53, 54]
    assert residues_for_feature("ACT_SITE", 102, 102) == [102]
    assert residues_for_feature("TRANSMEM", 5, 7) == [5, 6, 7]


# ------------------------------------------------------------- .dat parsing --

DAT = """\
ID   TEST1_HUMAN             Reviewed;          10 AA.
AC   P11111; P99999;
OS   Homo sapiens (Human).
OC   Eukaryota; Metazoa; Chordata; Craniata.
FT   ACT_SITE        3
FT                   /note="Proton acceptor"
FT                   /evidence="ECO:0000269|PubMed:12345678"
FT   DISULFID        2..9
FT                   /evidence="ECO:0000250|UniProtKB:P12345"
FT   BINDING         5..6
FT                   /ligand="ATP"
FT                   /ligand_id="ChEBI:CHEBI:30616"
FT                   /evidence="ECO:0000255|PROSITE-ProRule:PRU10141,
FT                   ECO:0000269|PubMed:99999999"
FT   VARIANT         4
FT                   /note="in a disease"
FT   CONFLICT        <1..2
SQ   SEQUENCE   10 AA;  1000 MW;  ABC CRC64;
     MKMKMKMKMK
//
ID   TEST2_HUMAN             Reviewed;           6 AA.
AC   P22222;
OS   Escherichia coli.
OC   Bacteria; Pseudomonadota.
FT   ACT_SITE        ?
FT   TRANSMEM        1..3
FT                   /evidence="ECO:0000255"
SQ   SEQUENCE   6 AA;  600 MW;  DEF CRC64;
     AAAAAA
//
"""


def _write_dat(tmp_path):
    p = tmp_path / "test.dat"
    p.write_text(DAT)
    return p


def test_iter_dat_records_extracts_accession_length_and_features(tmp_path):
    recs = list(iter_dat_records(_write_dat(tmp_path)))
    assert [r[0] for r in recs] == ["P11111", "P22222"]      # primary accession
    assert [r[1] for r in recs] == [10, 6]                    # sequence length

    feats = {(f[0], f[1], f[2]) for f in recs[0][2]}
    assert ("ACT_SITE", 3, 3) in feats
    assert ("DISULFID", 2, 9) in feats
    assert ("BINDING", 5, 6) in feats
    # VARIANT/CONFLICT are skipped as non-residue-biology / uncertain
    assert not any(f[0] in ("VARIANT", "CONFLICT") for f in recs[0][2])
    # the /ligand qualifier is captured, /ligand_id does not overwrite it
    binding = [f for f in recs[0][2] if f[0] == "BINDING"][0]
    assert binding[3] == "ATP"


def test_iter_dat_records_honours_keep_filter(tmp_path):
    recs = list(iter_dat_records(_write_dat(tmp_path), keep={"P22222"}))
    assert [r[0] for r in recs] == ["P22222"]


def test_uncertain_position_feature_is_dropped(tmp_path):
    recs = list(iter_dat_records(_write_dat(tmp_path), keep={"P22222"}))
    types = [f[0] for f in recs[0][2]]
    assert "TRANSMEM" in types
    assert "ACT_SITE" not in types                # "?" position -> dropped


def test_concept_positions_include_ligand_subconcept(tmp_path):
    recs = list(iter_dat_records(_write_dat(tmp_path), keep={"P11111"}))
    concepts = features_to_concept_positions(recs[0][2], subtype_note_types=SUBTYPES)
    assert set(concepts["ACT_SITE"]) == {3}
    assert set(concepts["DISULFID"]) == {2, 9}    # pair, NOT 2..9
    assert set(concepts["BINDING"]) == {5, 6}
    assert set(concepts["BINDING:ATP"]) == {5, 6}


# --------------------------------------------------- amendment E: sub-typing --

DAT_DOMAINS = """\
ID   TEST3_HUMAN             Reviewed;          20 AA.
AC   P33333;
OS   Homo sapiens (Human).
OC   Eukaryota; Metazoa; Chordata; Craniata.
FT   DOMAIN          1..10
FT                   /note="Protein kinase"
FT                   /evidence="ECO:0000255|PROSITE-ProRule:PRU00159"
FT   DOMAIN          11..20
FT                   /note="SH2"
FT                   /evidence="ECO:0000269|PubMed:11111111"
FT   REGION          1..5
FT                   /note="Disordered"
FT   ZN_FING         6..8
FT                   /note="C2H2-type 1"
FT   MOD_RES         2
FT                   /note="Phosphoserine"
SQ   SEQUENCE   20 AA;  2000 MW;  ABC CRC64;
     MKMKMKMKMKMKMKMKMKMK
//
"""


def _write_dat_domains(tmp_path):
    p = tmp_path / "domains.dat"
    p.write_text(DAT_DOMAINS)
    return p


def test_note_qualifier_creates_subtype_concepts(tmp_path):
    """THE amendment-E mechanism: a DOMAIN's /note becomes 'DOMAIN:<note>', same
    pattern as the existing BINDING:<ligand>. Two DOMAIN instances with
    DIFFERENT notes must NOT be merged into one sub-concept."""
    recs = list(iter_dat_records(_write_dat_domains(tmp_path)))
    feats = recs[0][2]
    concepts = features_to_concept_positions(feats, subtype_note_types=SUBTYPES)

    # type-level concept still pools everything, as before
    assert set(concepts["DOMAIN"]) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    # each note-specific sub-concept is its own, DISJOINT positive set
    assert set(concepts["DOMAIN:Protein kinase"]) == set(range(1, 11))
    assert set(concepts["DOMAIN:SH2"]) == set(range(11, 21))
    assert "DOMAIN:Protein kinase" != "DOMAIN:SH2"
    assert set(concepts["DOMAIN:Protein kinase"]).isdisjoint(concepts["DOMAIN:SH2"])

    assert set(concepts["REGION:Disordered"]) == {1, 2, 3, 4, 5}
    # the /note is "C2H2-type 1" -- the trailing ordinal is a UniProt
    # positional convention (multiple instances within one protein), not part
    # of the family name, so it is normalized away (normalize_subtype_note)
    assert set(concepts["ZN_FING:C2H2-type"]) == {6, 7, 8}
    assert "ZN_FING:C2H2-type 1" not in concepts

    # MOD_RES is NOT in SUBTYPE_NOTE_TYPES -- its /note ("Phosphoserine") must
    # NOT create a sub-concept, even though the qualifier is parsed the same way
    assert "MOD_RES:Phosphoserine" not in concepts
    assert set(concepts["MOD_RES"]) == {2}


def test_subtype_concepts_carry_evidence_tier_independently(tmp_path):
    recs = list(iter_dat_records(_write_dat_domains(tmp_path)))
    concepts = features_to_concept_positions(recs[0][2], subtype_note_types=SUBTYPES)
    # PROSITE-rule DOMAIN (0000255) -> predicted; the SH2 domain here is
    # experimental (0000269) -- sub-concepts must not share/merge evidence
    assert concepts["DOMAIN:Protein kinase"][1] == "predicted"
    assert concepts["DOMAIN:SH2"][11] == "experimental"
    # and the pooled type-level DOMAIN keeps each residue's OWN evidence too
    assert concepts["DOMAIN"][1] == "predicted"
    assert concepts["DOMAIN"][11] == "experimental"


def test_occurrences_preserve_instance_boundaries_not_flattened_residues():
    """The whole point of Occurrence over the flattened residue dict: TWO
    DOMAIN instances of the SAME family must be counted as two occurrences
    (needed for the disjunctive instance-count floor), not silently merged."""
    feats = [
        Feature("DOMAIN", 1, 10, None, (), "Protein kinase"),
        Feature("DOMAIN", 50, 60, None, (), "Protein kinase"),   # 2nd instance
        Feature("DOMAIN", 100, 105, None, (), "SH2"),
    ]
    occ = features_to_concept_occurrences(feats, subtype_note_types=SUBTYPES)
    assert len(occ["DOMAIN:Protein kinase"]) == 2
    assert len(occ["DOMAIN:SH2"]) == 1
    assert len(occ["DOMAIN"]) == 3                  # type-level pools all instances
    assert occ["DOMAIN:Protein kinase"][0] == Occurrence(1, 10, "none")
    assert occ["DOMAIN:Protein kinase"][1] == Occurrence(50, 60, "none")

    # and occurrences_to_concept_positions correctly flattens them back
    positions = occurrences_to_concept_positions(occ)
    assert set(positions["DOMAIN:Protein kinase"]) == set(range(1, 11)) | set(range(50, 61))


def test_subtyping_respects_paired_residue_types():
    """DISULFID/CROSSLNK are not in SUBTYPE_NOTE_TYPES, so even if a /note were
    present it must not create a sub-concept, and the type-level concept must
    still obey the pair rule (trap 1), not silently become a range."""
    feats = [Feature("DISULFID", 10, 90, None, (), "Interchain")]
    occ = features_to_concept_occurrences(feats, subtype_note_types=SUBTYPES)
    assert "DISULFID:Interchain" not in occ
    positions = occurrences_to_concept_positions(occ)
    assert set(positions["DISULFID"]) == {10, 90}    # pair, not the 81-residue span


def test_normalize_subtype_note_strips_trailing_ordinal_only():
    """THE fragmentation bug, found by running the real rebuild (17 separate
    'C2H2-type N' concepts where InterPLM has ONE 'C2H2-type'). Only a
    trailing bare-integer ordinal is stripped; an embedded number that is
    part of the real name (e.g. 'type-1') survives."""
    assert normalize_subtype_note("C2H2-type 1") == "C2H2-type"
    assert normalize_subtype_note("C2H2-type 17") == "C2H2-type"
    assert normalize_subtype_note("ABC transporter 2") == "ABC transporter"
    assert normalize_subtype_note("ABC transmembrane type-1 1") == "ABC transmembrane type-1"
    assert normalize_subtype_note("ABC transmembrane type-1 2") == "ABC transmembrane type-1"
    # semicolon-qualified variants are a real structural distinction, kept
    assert normalize_subtype_note("RING-type; degenerate") == "RING-type; degenerate"
    assert normalize_subtype_note("RING-type; atypical") == "RING-type; atypical"
    # no trailing ordinal -- no-op
    assert normalize_subtype_note("Protein kinase") == "Protein kinase"
    assert normalize_subtype_note("SH2") == "SH2"


def test_repeated_domain_instances_collapse_to_one_family_concept():
    """End-to-end version of the fragmentation fix: THREE ZN_FING occurrences
    in one protein, numbered 1/2/3 per UniProt convention, must land in ONE
    concept with 3 instances -- not three concepts with 1 instance each."""
    feats = [
        Feature("ZN_FING", 1, 10, None, (), "C2H2-type 1"),
        Feature("ZN_FING", 20, 29, None, (), "C2H2-type 2"),
        Feature("ZN_FING", 40, 49, None, (), "C2H2-type 3"),
    ]
    occ = features_to_concept_occurrences(feats, subtype_note_types=SUBTYPES)
    assert "ZN_FING:C2H2-type" in occ
    assert len(occ["ZN_FING:C2H2-type"]) == 3
    assert "ZN_FING:C2H2-type 1" not in occ
    assert "ZN_FING:C2H2-type 2" not in occ
    assert "ZN_FING:C2H2-type 3" not in occ


def test_note_disabled_via_empty_subtype_set():
    feats = [Feature("DOMAIN", 1, 10, None, (), "Protein kinase")]
    occ = features_to_concept_occurrences(feats, subtype_note_types=frozenset())
    assert "DOMAIN:Protein kinase" not in occ
    assert "DOMAIN" in occ


def test_occurrence_with_no_note_goes_to_unspecified_not_dropped():
    """Checked against the real label store: the pooled parent's positive set
    is neither pure duplication of its /note children NOR a pure no-note
    residual -- it's the union of both, with a genuine 7-15% residual. A
    no-note occurrence of a subtype-note FT type must become its own
    ':unspecified' concept (a real, disjoint classification target -- the
    InterPLM 'any' category), not vanish into only the pooled parent."""
    feats = [
        Feature("DOMAIN", 1, 10, None, (), "Protein kinase"),
        Feature("DOMAIN", 20, 30, None, (), None),          # no /note at all
    ]
    occ = features_to_concept_occurrences(feats, subtype_note_types=SUBTYPES)
    assert "DOMAIN:Protein kinase" in occ and len(occ["DOMAIN:Protein kinase"]) == 1
    assert "DOMAIN:unspecified" in occ and len(occ["DOMAIN:unspecified"]) == 1
    assert len(occ["DOMAIN"]) == 2                          # pooled parent still pools both
    # the two sub-concepts must be DISJOINT occurrence sets
    assert occ["DOMAIN:Protein kinase"][0] != occ["DOMAIN:unspecified"][0]


def test_occurrence_whose_note_is_empty_after_ordinal_stripping_goes_to_unspecified():
    """Edge case: a /note that is ONLY a bare ordinal (e.g. a malformed ' 3')
    normalizes to an empty string -- must fall through to ':unspecified'
    rather than silently vanishing (the pre-fix behavior)."""
    feats = [Feature("ZN_FING", 1, 5, None, (), " 3")]
    occ = features_to_concept_occurrences(feats, subtype_note_types=SUBTYPES)
    assert "ZN_FING:unspecified" in occ
    assert "ZN_FING: 3" not in occ and "ZN_FING:3" not in occ


# ---------------------------------------------------------------- evidence --

def test_eco_to_tier_prefers_most_direct_evidence():
    """A feature carrying both a prediction and an experiment is EXPERIMENTAL --
    most direct wins, so tiers stay disjoint."""
    assert eco_to_tier(["0000269"]) == "experimental"
    assert eco_to_tier(["0000250"]) == "by_similarity"
    assert eco_to_tier(["0000255"]) == "predicted"
    assert eco_to_tier(["0000255", "0000269"]) == "experimental"
    assert eco_to_tier(["0000250", "0000255"]) == "by_similarity"
    assert eco_to_tier([]) == "none"
    assert eco_to_tier(["9999999"]) == "none"          # unrecognised code


def test_evidence_codes_are_parsed_including_multiline(tmp_path):
    """A /evidence value routinely WRAPS across FT lines. Reading only the first
    line would silently drop the experimental code here and misclassify a
    propagated label as observed -- exactly the provenance error that matters."""
    recs = list(iter_dat_records(_write_dat(tmp_path), keep={"P11111"}))
    feats = {f.ftype: f for f in recs[0][2]}
    assert feats["ACT_SITE"].eco == ("0000269",)
    assert feats["DISULFID"].eco == ("0000250",)
    # wrapped across two lines: BOTH codes must be recovered
    assert set(feats["BINDING"].eco) == {"0000255", "0000269"}
    assert eco_to_tier(feats["BINDING"].eco) == "experimental"
    # the wrapped continuation must not swallow the FOLLOWING feature
    assert "VARIANT" not in feats           # skipped type, but was still parsed as new


def test_concept_positions_carry_evidence_tier(tmp_path):
    recs = list(iter_dat_records(_write_dat(tmp_path), keep={"P11111"}))
    concepts = features_to_concept_positions(recs[0][2], subtype_note_types=SUBTYPES)
    assert concepts["ACT_SITE"][3] == "experimental"
    assert concepts["DISULFID"][2] == "by_similarity"
    assert concepts["DISULFID"][9] == "by_similarity"
    assert concepts["BINDING"][5] == "experimental"     # multi-line, best wins


def test_tiers_are_disjoint_best_evidence_wins():
    """A residue annotated twice -- once by similarity, once experimentally --
    must land in exactly one tier (the stronger), so strata partition cleanly."""
    from src.annotations.uniprot_sites import Feature
    feats = [
        Feature("ACT_SITE", 10, 10, None, ("0000250",)),   # by similarity
        Feature("ACT_SITE", 10, 10, None, ("0000269",)),   # experimental
        Feature("ACT_SITE", 11, 11, None, ("0000255",)),   # predicted only
    ]
    concepts = features_to_concept_positions(feats, subtype_note_types=SUBTYPES)
    assert concepts["ACT_SITE"][10] == "experimental"
    assert concepts["ACT_SITE"][11] == "predicted"


# ---------------------------------------------------------------- alignment --

def test_map_residues_to_rows_is_one_based_to_zero_based():
    """Residue p of a block starting at row `start` is row start + (p-1)."""
    entry = {"accession": "X", "start": 100, "n_rows": 10, "seq_len": 10,
             "truncated": False, "residue_ids": None}
    dropped = {"truncated_or_out_of_range": 0}
    assert map_residues_to_rows(entry, [1], dropped) == [100]      # first residue
    assert map_residues_to_rows(entry, [10], dropped) == [109]     # last residue
    assert map_residues_to_rows(entry, [3, 5], dropped) == [102, 104]
    assert dropped["truncated_or_out_of_range"] == 0


def test_map_residues_drops_positions_past_truncation():
    """A truncated protein has fewer rows than residues; the tail annotations
    must be DROPPED, not clamped onto the last row."""
    entry = {"accession": "X", "start": 0, "n_rows": 5, "seq_len": 5,
             "truncated": True, "residue_ids": None}
    dropped = {"truncated_or_out_of_range": 0}
    rows = map_residues_to_rows(entry, [1, 5, 6, 900], dropped)
    assert rows == [0, 4]
    assert dropped["truncated_or_out_of_range"] == 2


def test_map_residues_respects_subsampled_residue_ids():
    """Subsampled caches store which residues survived; row j is residue
    residue_ids[j], so a dropped residue must not silently map to a neighbour."""
    entry = {"accession": "X", "start": 50, "n_rows": 3, "seq_len": 10,
             "truncated": False, "residue_ids": [0, 4, 9]}
    dropped = {"not_subsampled": 0}
    assert map_residues_to_rows(entry, [1, 5, 10], dropped) == [50, 51, 52]
    assert map_residues_to_rows(entry, [2], dropped) == []      # residue 1 not kept
    assert dropped["not_subsampled"] == 1


# -------------------------------------------------------------- end-to-end --

def _write_cache(tmp_path, index, n_rows, d=4):
    p = tmp_path / "eval.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("activations",
                          data=np.zeros((n_rows, d), dtype=np.float16))
        hf.create_dataset("index_json", data=np.bytes_(json.dumps(index)))
    return p


def test_end_to_end_labels_land_on_exact_rows(tmp_path, monkeypatch):
    """Full path: .dat -> concepts -> cache rows. Pins exact row indices, which
    is what an off-by-one would break.

    The index uses the REAL cache format -- cache.py stores the whole FASTA id
    token ("sp|P11111|TEST1_HUMAN"), not the bare accession the flat file uses.
    Against real data that mismatch matched 0/53,672 proteins, so this test
    keeps the normalisation honest."""
    dat = _write_dat(tmp_path)
    index = [
        {"accession": "sp|P11111|TEST1_HUMAN", "start": 0, "n_rows": 10,
         "seq_len": 10, "truncated": False, "residue_ids": None},
        {"accession": "sp|P22222|TEST2_HUMAN", "start": 10, "n_rows": 6,
         "seq_len": 6, "truncated": False, "residue_ids": None},
    ]
    cache = _write_cache(tmp_path, index, n_rows=16)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n  subtype_note_types: [DOMAIN, REGION, MOTIF, ZN_FING]\n")
    out = tmp_path / "labels.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat),
        "--cache", str(cache), "--out", str(out),
    ])
    main()

    with h5py.File(out, "r") as hf:
        got = {k: hf["concepts"][k]["all"][:].tolist() for k in hf["concepts"]}
        assert hf.attrs["n_rows"] == 16
        # tiers must PARTITION the pooled set, so stratified analysis is clean
        for name in hf["concepts"]:
            grp = hf["concepts"][name]
            tiers = [t for t in grp if t != "all"]
            tier_rows = sorted(r for t in tiers for r in grp[t][:].tolist())
            assert tier_rows == sorted(grp["all"][:].tolist())
        # provenance is preserved per concept
        assert hf["concepts"]["DISULFID"]["by_similarity"][:].tolist() == [1, 8]
        assert hf["concepts"]["ACT_SITE"]["experimental"][:].tolist() == [2]

    # P11111 starts at row 0: ACT_SITE residue 3 -> row 2
    assert got["ACT_SITE"] == [2]
    # DISULFID residues 2 and 9 -> rows 1 and 8 (NOT rows 1..8)
    assert got["DISULFID"] == [1, 8]
    # BINDING residues 5,6 -> rows 4,5
    assert got["BINDING"] == [4, 5]
    assert got["BINDING:ATP"] == [4, 5]
    # P22222 starts at row 10: TRANSMEM residues 1..3 -> rows 10,11,12
    assert got["TRANSMEM"] == [10, 11, 12]


def test_end_to_end_drops_rare_concepts(tmp_path, monkeypatch):
    """min_positive_residues must actually filter -- AP on 1 positive is noise."""
    dat = _write_dat(tmp_path)
    index = [{"accession": "P11111", "start": 0, "n_rows": 10, "seq_len": 10,
              "truncated": False, "residue_ids": None}]
    cache = _write_cache(tmp_path, index, n_rows=10)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 2\n  subtype_note_types: [DOMAIN, REGION, MOTIF, ZN_FING]\n")
    out = tmp_path / "labels.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat),
        "--cache", str(cache), "--out", str(out),
    ])
    main()
    with h5py.File(out, "r") as hf:
        names = set(hf["concepts"])
    assert "ACT_SITE" not in names          # only 1 positive residue
    assert "DISULFID" in names              # 2 positives
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert manifest["dropped_small_concepts"]["ACT_SITE"] == 1


def test_low_match_rate_fails_loudly(tmp_path, monkeypatch):
    """The guard must FIRE: a cache whose proteins are absent from the flat file
    must raise, not quietly write an empty label set. Silently emitting zero
    concepts is the confident-wrong-answer failure this project is prone to --
    it is how the accession mismatch nearly went unnoticed."""
    import pytest
    dat = _write_dat(tmp_path)
    index = [{"accession": "sp|NOTREAL1|X_HUMAN", "start": 0, "n_rows": 10,
              "seq_len": 10, "truncated": False, "residue_ids": None},
             {"accession": "sp|NOTREAL2|Y_HUMAN", "start": 10, "n_rows": 6,
              "seq_len": 6, "truncated": False, "residue_ids": None}]
    cache = _write_cache(tmp_path, index, n_rows=16)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n  subtype_note_types: [DOMAIN, REGION, MOTIF, ZN_FING]\n")
    out = tmp_path / "labels.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat),
        "--cache", str(cache), "--out", str(out),
    ])
    with pytest.raises(SystemExit):
        main()
    assert not out.exists()          # and no label file was left behind


def test_organism_and_lineage_are_parsed(tmp_path):
    """Needed for the stratum COMPOSITION report -- experimental annotation
    concentrates in model organisms, so taxonomy is part of the confound."""
    recs = list(iter_dat_records(_write_dat(tmp_path)))
    assert recs[0].organism == "Homo sapiens (Human)"
    assert recs[0].lineage == "Eukaryota"
    assert recs[1].organism == "Escherichia coli"
    assert recs[1].lineage == "Bacteria"


def test_shared_experimental_protein_set_is_emitted(tmp_path, monkeypatch):
    """Amendment A1: both strata must be restricted to proteins carrying >=1
    EXPERIMENTAL label, so evidence tier is the only thing varying between them.

    P11111 has an experimental ACT_SITE; P22222 has only a predicted TRANSMEM,
    so only P11111 belongs to the shared set."""
    dat = _write_dat(tmp_path)
    index = [
        {"accession": "sp|P11111|TEST1_HUMAN", "start": 0, "n_rows": 10,
         "seq_len": 10, "truncated": False, "residue_ids": None},
        {"accession": "sp|P22222|TEST2_HUMAN", "start": 10, "n_rows": 6,
         "seq_len": 6, "truncated": False, "residue_ids": None},
    ]
    cache = _write_cache(tmp_path, index, n_rows=16)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n  subtype_note_types: [DOMAIN, REGION, MOTIF, ZN_FING]\n")
    out = tmp_path / "labels.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat),
        "--cache", str(cache), "--out", str(out),
    ])
    main()

    with h5py.File(out, "r") as hf:
        pg = hf["protein_sets"]
        accs = [a.decode() if isinstance(a, bytes) else a
                for a in pg["has_experimental_accessions"][:]]
        assert accs == ["P11111"]
        assert pg["has_experimental_start"][:].tolist() == [0]
        assert pg["has_experimental_n_rows"][:].tolist() == [10]

    manifest = json.loads(out.with_suffix(".json").read_text())
    assert manifest["composition_all_eval"]["n_proteins"] == 2
    assert manifest["composition_shared_experimental"]["n_proteins"] == 1
    # composition must expose the taxonomic skew between the two sets
    assert manifest["composition_all_eval"]["lineage"] == {"Eukaryota": 1, "Bacteria": 1}
    assert manifest["composition_shared_experimental"]["lineage"] == {"Eukaryota": 1}


# --------------------------- amendment E: disjunctive retention floor, e2e --

def _dat_rare_domain_family(n_instances: int, protein_len: int) -> str:
    """One protein carrying `n_instances` single-residue DOMAIN occurrences of
    the SAME /note family at positions 1..n_instances -- few residues, many
    instances, exactly the case InterPLM's disjunctive floor exists for."""
    lines = [
        f"ID   TESTD_HUMAN             Reviewed;          {protein_len} AA.",
        "AC   P44444;",
        "OS   Homo sapiens (Human).",
        "OC   Eukaryota; Metazoa; Chordata; Craniata.",
    ]
    for i in range(1, n_instances + 1):
        lines.append(f"FT   DOMAIN          {i}")
        lines.append('FT                   /note="Rare Family"')
    lines.append(f"SQ   SEQUENCE   {protein_len} AA;  1 MW;  ABC CRC64;")
    lines.append("     " + "M" * protein_len)
    lines.append("//")
    return "\n".join(lines) + "\n"


def test_interplm_floor_rescues_rare_subtype_legacy_floor_drops(tmp_path, monkeypatch):
    """THE amendment-E floor test. 12 single-residue DOMAIN occurrences of one
    family = 12 residues, 12 instances. legacy (residues>=15) DROPS it;
    interplm (instances>10) KEEPS it -- and the choice must actually change
    which concepts survive, not just be computed and ignored."""
    dat = tmp_path / "rare.dat"
    dat.write_text(_dat_rare_domain_family(n_instances=12, protein_len=20))
    index = [{"accession": "P44444", "start": 0, "n_rows": 20, "seq_len": 20,
              "truncated": False, "residue_ids": None}]
    cache = _write_cache(tmp_path, index, n_rows=20)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 15\n"
                   "  retention_min_residues: 1500\n  retention_min_instances: 10\n"
                   "  subtype_note_types: [DOMAIN, REGION, MOTIF, ZN_FING]\n")

    out_legacy = tmp_path / "legacy.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat), "--cache", str(cache),
        "--out", str(out_legacy), "--retention-floor", "legacy"])
    main()
    with h5py.File(out_legacy, "r") as hf:
        assert "DOMAIN:Rare Family" not in hf["concepts"]      # 12 < 15 residues

    out_interplm = tmp_path / "interplm.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat), "--cache", str(cache),
        "--out", str(out_interplm), "--retention-floor", "interplm"])
    main()
    with h5py.File(out_interplm, "r") as hf:
        assert "DOMAIN:Rare Family" in hf["concepts"]          # 12 > 10 instances
        assert hf["concepts"]["DOMAIN:Rare Family"]["all"][:].tolist() == list(range(12))

    manifest = json.loads(out_interplm.with_suffix(".json").read_text())
    diag = manifest["retention_floor_diagnostics"]["DOMAIN:Rare Family"]
    assert diag["n_residues"] == 12 and diag["n_instances"] == 12
    assert diag["passes_legacy"] is False and diag["passes_interplm"] is True


def test_floor_decision_uses_untruncated_size_not_the_compute_artifact(tmp_path, monkeypatch):
    """P4 (measure_truncation_impact.py) found 60 of 3,122 real concepts flip
    retention-floor status depending on whether the floor is measured on
    truncated (surviving) or untruncated counts -- a concept's TRUE size
    should not depend on where ESM-2's 1022-residue cap happened to fall.
    Construct exactly that case: a single DOMAIN instance spanning 2000
    residues (> the 1500 floor), cached down to only 100 surviving rows (both
    residues=100 and instances=1 FAIL the floor on truncated counts alone) --
    must still be KEPT, sized correctly (100 surviving rows written), and the
    manifest must expose both counts so the fix is auditable."""
    dat = tmp_path / "big.dat"
    dat.write_text(
        "ID   TESTB_HUMAN             Reviewed;        2000 AA.\n"
        "AC   P66666;\n"
        "OS   Homo sapiens (Human).\n"
        "OC   Eukaryota; Metazoa; Chordata; Craniata.\n"
        "FT   DOMAIN          1..2000\n"
        'FT                   /note="Huge Family"\n'
        "SQ   SEQUENCE   2000 AA;  1 MW;  ABC CRC64;\n"
        "     " + "M" * 2000 + "\n"
        "//\n"
    )
    index = [{"accession": "P66666", "start": 0, "n_rows": 100, "seq_len": 2000,
              "truncated": True, "residue_ids": None}]
    cache = _write_cache(tmp_path, index, n_rows=100)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n"
                   "  retention_min_residues: 1500\n  retention_min_instances: 10\n"
                   "  subtype_note_types: [DOMAIN, REGION, MOTIF, ZN_FING]\n")
    out = tmp_path / "big.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat), "--cache", str(cache),
        "--out", str(out), "--retention-floor", "interplm"])
    main()

    with h5py.File(out, "r") as hf:
        assert "DOMAIN:Huge Family" in hf["concepts"]
        assert hf["concepts"]["DOMAIN:Huge Family"]["all"][:].tolist() == list(range(100))

    manifest = json.loads(out.with_suffix(".json").read_text())
    diag = manifest["retention_floor_diagnostics"]["DOMAIN:Huge Family"]
    assert diag["n_residues"] == 100                    # truncated/surviving, unchanged
    assert diag["n_residues_untruncated"] == 2000        # the TRUE size
    assert diag["n_instances"] == 1
    assert diag["n_instances_untruncated"] == 1
    assert abs(diag["truncation_residue_loss_frac"] - 1900 / 2000) < 1e-9
    assert diag["passes_interplm"] is True               # kept on UNTRUNCATED size
    # sanity: on truncated counts alone (100 residues, 1 instance) this would
    # fail BOTH floor criteria -- the fix is what rescues it
    assert not (100 > 1500 or 1 > 10)


# -------------------------------- amendment G2: the untruncated population --

def test_population_untruncated_excludes_truncated_proteins_entirely(tmp_path, monkeypatch):
    """THE amendment-G2 fix. A truncated protein contributing a rare concept
    must be excluded WHOLESALE under truncation_policy=untruncated -- not its
    tail residues (that was the OLD, amendment-E behaviour, which is exactly
    what let a concept with 11 true instances but only 1 surviving one clear
    the floor). Two proteins: P11111 (untruncated, contributes a small
    DOMAIN) and P99999 (truncated, would ALSO contribute a DOMAIN instance,
    but must not appear under this population at all)."""
    dat = tmp_path / "pop.dat"
    dat.write_text(
        "ID   TESTP_HUMAN             Reviewed;          20 AA.\n"
        "AC   P11111;\n"
        "OS   Homo sapiens (Human).\n"
        "OC   Eukaryota; Metazoa; Chordata; Craniata.\n"
        "FT   DOMAIN          1..10\n"
        'FT                   /note="Shared Family"\n'
        "SQ   SEQUENCE   20 AA;  1 MW;  ABC CRC64;\n"
        "     " + "M" * 20 + "\n"
        "//\n"
        "ID   TESTQ_HUMAN             Reviewed;        2000 AA.\n"
        "AC   P99999;\n"
        "OS   Homo sapiens (Human).\n"
        "OC   Eukaryota; Metazoa; Chordata; Craniata.\n"
        "FT   DOMAIN          1..10\n"
        'FT                   /note="Shared Family"\n'
        "SQ   SEQUENCE   2000 AA;  1 MW;  ABC CRC64;\n"
        "     " + "M" * 2000 + "\n"
        "//\n"
    )
    index = [
        {"accession": "P11111", "start": 0, "n_rows": 20, "seq_len": 20,
         "truncated": False, "residue_ids": None},
        {"accession": "P99999", "start": 20, "n_rows": 100, "seq_len": 2000,
         "truncated": True, "residue_ids": None},           # truncated!
    ]
    cache = _write_cache(tmp_path, index, n_rows=120)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n"
                   "  retention_min_residues: 5\n  retention_min_instances: 0\n"
                   "  subtype_note_types: [DOMAIN]\n")

    out_all = tmp_path / "all.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat), "--cache", str(cache),
        "--out", str(out_all), "--truncation-policy", "with_truncated"])
    main()
    with h5py.File(out_all, "r") as hf:
        # both proteins' instances present -> 2 instances, both proteins
        assert hf["concepts"]["DOMAIN:Shared Family"]["all"].shape[0] == 20  # 10+10 rows

    out_pop = tmp_path / "untrunc.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat), "--cache", str(cache),
        "--out", str(out_pop), "--truncation-policy", "untruncated"])
    main()
    with h5py.File(out_pop, "r") as hf:
        assert hf.attrs["n_rows"] == 20                 # ONLY P11111's rows
        assert hf.attrs["truncation_policy"] == "untruncated"
        # P99999 contributes NOTHING -- only P11111's 10 residues
        assert hf["concepts"]["DOMAIN:Shared Family"]["all"][:].tolist() == list(range(10))
        pg = hf["protein_sets"]
        assert pg["population_start"][:].tolist() == [0]
        assert pg["population_n_rows"][:].tolist() == [20]

    manifest = json.loads(out_pop.with_suffix(".json").read_text())
    diag = manifest["retention_floor_diagnostics"]["DOMAIN:Shared Family"]
    # under this population there is no truncation gap at all: untruncated
    # counts and surviving counts are IDENTICAL by construction
    assert diag["n_residues"] == diag["n_residues_untruncated"] == 10
    assert diag["n_proteins"] == 1


def test_min_protein_count_gate_excludes_single_protein_concepts(tmp_path, monkeypatch):
    """THE G3 gate. 'Multi' has instances in 3 distinct proteins; 'Single' has
    an instance in only 1 protein -- exactly the estimability failure the
    gate exists to catch, since Null A's joint shift moves every instance in
    one protein together. With --min-protein-count 2, 'Single' must be
    dropped and 'Multi' must survive; with the default (1), both survive."""
    dat = tmp_path / "gate.dat"
    dat.write_text(
        "ID   TESTA_HUMAN             Reviewed;          20 AA.\n"
        "AC   P11111;\n"
        "OS   Homo sapiens (Human).\n"
        "OC   Eukaryota; Metazoa; Chordata; Craniata.\n"
        "FT   DOMAIN          1..2\n"
        'FT                   /note="Multi"\n'
        "FT   DOMAIN          5..6\n"
        'FT                   /note="Single"\n'
        "SQ   SEQUENCE   20 AA;  1 MW;  ABC CRC64;\n"
        "     " + "M" * 20 + "\n"
        "//\n"
        "ID   TESTB_HUMAN             Reviewed;          20 AA.\n"
        "AC   P22222;\n"
        "OS   Homo sapiens (Human).\n"
        "OC   Eukaryota; Metazoa; Chordata; Craniata.\n"
        "FT   DOMAIN          1..2\n"
        'FT                   /note="Multi"\n'
        "SQ   SEQUENCE   20 AA;  1 MW;  ABC CRC64;\n"
        "     " + "M" * 20 + "\n"
        "//\n"
        "ID   TESTC_HUMAN             Reviewed;          20 AA.\n"
        "AC   P33333;\n"
        "OS   Homo sapiens (Human).\n"
        "OC   Eukaryota; Metazoa; Chordata; Craniata.\n"
        "FT   DOMAIN          1..2\n"
        'FT                   /note="Multi"\n'
        "SQ   SEQUENCE   20 AA;  1 MW;  ABC CRC64;\n"
        "     " + "M" * 20 + "\n"
        "//\n"
    )
    index = [
        {"accession": f"P{d}", "start": i * 20, "n_rows": 20, "seq_len": 20,
         "truncated": False, "residue_ids": None}
        for i, d in enumerate(["11111", "22222", "33333"])
    ]
    cache = _write_cache(tmp_path, index, n_rows=60)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n"
                   "  retention_min_residues: 0\n  retention_min_instances: 0\n"
                   "  subtype_note_types: [DOMAIN]\n")

    out_nogate = tmp_path / "nogate.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat), "--cache", str(cache),
        "--out", str(out_nogate)])
    main()
    with h5py.File(out_nogate, "r") as hf:
        assert "DOMAIN:Multi" in hf["concepts"]
        assert "DOMAIN:Single" in hf["concepts"]     # default: gate is a no-op

    out_gated = tmp_path / "gated.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat), "--cache", str(cache),
        "--out", str(out_gated), "--min-protein-count", "2"])
    main()
    with h5py.File(out_gated, "r") as hf:
        assert "DOMAIN:Multi" in hf["concepts"]       # 3 proteins >= 2 -- survives
        assert "DOMAIN:Single" not in hf["concepts"]  # 1 protein < 2 -- gated out
        assert hf.attrs["min_protein_count"] == 2

    manifest = json.loads(out_gated.with_suffix(".json").read_text())
    diag = manifest["retention_floor_diagnostics"]["DOMAIN:Single"]
    assert diag["n_proteins"] == 1
    assert diag["passes_gate"] is False


def test_instance_boundaries_written_for_span_not_paired_concepts(tmp_path, monkeypatch):
    dat = _write_dat(tmp_path)
    index = [
        {"accession": "sp|P11111|TEST1_HUMAN", "start": 0, "n_rows": 10,
         "seq_len": 10, "truncated": False, "residue_ids": None},
        {"accession": "sp|P22222|TEST2_HUMAN", "start": 10, "n_rows": 6,
         "seq_len": 6, "truncated": False, "residue_ids": None},
    ]
    cache = _write_cache(tmp_path, index, n_rows=16)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n  subtype_note_types: [DOMAIN, REGION, MOTIF, ZN_FING]\n")
    out = tmp_path / "labels.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat),
        "--cache", str(cache), "--out", str(out)])
    main()
    with h5py.File(out, "r") as hf:
        inst = hf["instances"]
        # TRANSMEM 1..3 on P22222 (starts at row 10) is ONE span instance
        assert inst["TRANSMEM"]["start_row"][:].tolist() == [10]
        assert inst["TRANSMEM"]["end_row"][:].tolist() == [12]
        # DISULFID is a PAIRED type -- must have NO instance boundaries at all
        assert "DISULFID" not in inst


# ------------------------------------- amendment E: two real bugs, pinned --

def test_sanitize_concept_name_replaces_ascii_slash():
    """HDF5 group names can't contain '/'. 'Alpha/beta hydrolase' is a common,
    REAL UniProt DOMAIN /note value, not a hypothetical -- this crashed the
    first amendment-E rebuild attempt (job 19942532) via create_group()
    silently interpreting it as a nested-group path."""
    assert "/" not in sanitize_concept_name("DOMAIN:Alpha/beta hydrolase")
    assert sanitize_concept_name("ACT_SITE") == "ACT_SITE"     # no-op when clean
    # ftype extraction (concept.split(':', 1)[0]) must still recover the FT
    # type after sanitization -- the substitution never touches the prefix
    assert sanitize_concept_name("DOMAIN:Alpha/beta hydrolase").split(":", 1)[0] == "DOMAIN"


def _dat_truncated_instances(n_occurrences: int, seq_len: int, cached_len: int) -> str:
    """`n_occurrences` single-residue DOMAIN occurrences of the SAME family,
    ALL placed past `cached_len` (i.e. in the truncated tail) -- every one
    should be dropped, leaving ZERO surviving residues for this concept."""
    lines = [
        f"ID   TESTT_HUMAN             Reviewed;          {seq_len} AA.",
        "AC   P55555;",
        "OS   Homo sapiens (Human).",
        "OC   Eukaryota; Metazoa; Chordata; Craniata.",
    ]
    for i in range(cached_len + 1, cached_len + 1 + n_occurrences):
        lines.append(f"FT   DOMAIN          {i}")
        lines.append('FT                   /note="Truncated Family"')
    lines.append(f"SQ   SEQUENCE   {seq_len} AA;  1 MW;  ABC CRC64;")
    lines.append("     " + "M" * seq_len)
    lines.append("//")
    return "\n".join(lines) + "\n"


def test_instance_count_ignores_truncated_occurrences_and_does_not_crash(tmp_path, monkeypatch):
    """THE second amendment-E bug, also found by running the real rebuild
    (same job): a concept could pass the disjunctive floor on RAW instance
    count while every occurrence was truncated away, leaving an EMPTY
    per-tier dict that crashed np.concatenate at h5-write time. 15 occurrences
    (> the instance floor of 10), all past the cached length -- must be
    DROPPED, and the run must not crash."""
    seq_len, cached_len = 50, 20
    dat = tmp_path / "trunc.dat"
    dat.write_text(_dat_truncated_instances(n_occurrences=15, seq_len=seq_len,
                                            cached_len=cached_len))
    index = [{"accession": "P55555", "start": 0, "n_rows": cached_len,
              "seq_len": seq_len, "truncated": True, "residue_ids": None}]
    cache = _write_cache(tmp_path, index, n_rows=cached_len)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 100\n"
                   "  retention_min_residues: 1500\n  retention_min_instances: 10\n"
                   "  subtype_note_types: [DOMAIN, REGION, MOTIF, ZN_FING]\n")
    out = tmp_path / "trunc.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat), "--cache", str(cache),
        "--out", str(out), "--retention-floor", "interplm"])
    main()                                    # must not raise
    with h5py.File(out, "r") as hf:
        assert "DOMAIN:Truncated Family" not in hf["concepts"]
    manifest = json.loads(out.with_suffix(".json").read_text())
    diag = manifest["retention_floor_diagnostics"].get("DOMAIN:Truncated Family")
    assert diag is not None
    assert diag["n_residues"] == 0 and diag["n_instances"] == 0   # NOT 15
    assert diag["passes_interplm"] is False


def test_load_cache_index_roundtrip(tmp_path):
    index = [{"accession": "A", "start": 0, "n_rows": 3, "seq_len": 3,
              "truncated": False, "residue_ids": None}]
    cache = _write_cache(tmp_path, index, n_rows=3)
    got, n_rows = load_cache_index(cache)
    assert n_rows == 3
    assert got[0]["accession"] == "A"


# ---------------------------------------- M2: provenance exclusion, ENFORCED --
# PREREGISTRATION section 1 excludes HELIX/STRAND/TURN (PDB-derived, so
# "absent" means NOT MEASURED while AP scores unlabelled rows as negative) and
# COMPBIAS (zero experimental residues). Until 2026-07-28 that exclusion lived
# only in a config comment and nothing applied it, so all four were in the
# 716-concept library and in the argmax pool defining `g_f` -- three of them
# among the ten broadest concepts in it. These tests keep it applied.

from src.annotations.build_residue_labels import provenance_excluded


def test_provenance_exclusion_drops_the_four_prereg_types():
    concepts = ["ACT_SITE", "HELIX", "STRAND", "TURN", "COMPBIAS",
                "DOMAIN:Protein kinase", "BINDING:ATP"]
    got = provenance_excluded(concepts, ["HELIX", "STRAND", "TURN", "COMPBIAS"])
    assert got == ["COMPBIAS", "HELIX", "STRAND", "TURN"]


def test_provenance_exclusion_matches_on_TYPE_so_subtypes_go_too():
    """Matching must be on the FT type, not the full concept name -- otherwise
    a sub-typed child survives its excluded parent and the PDB-provenance
    problem walks straight back into the pool under a different name."""
    got = provenance_excluded(["HELIX", "HELIX:Alpha", "DOMAIN:SH2"], ["HELIX"])
    assert got == ["HELIX", "HELIX:Alpha"]


def test_provenance_exclusion_does_not_touch_legitimate_concepts():
    """The other half of gotcha 13: the filter must also NOT fire on concepts
    it has no business removing, or 'it excludes the right things' is untested
    in the negative. TRANSMEM and TOPO_DOM are broad and structural but are
    sequence-derived, not PDB-derived, and stay."""
    concepts = ["TRANSMEM", "TOPO_DOM:Cytoplasmic", "REGION:Disordered",
                "COILED", "DOMAIN:ABC transporter"]
    assert provenance_excluded(concepts, ["HELIX", "STRAND", "TURN", "COMPBIAS"]) == []


def test_empty_exclusion_list_is_a_no_op_not_a_crash():
    concepts = ["HELIX", "ACT_SITE"]
    assert provenance_excluded(concepts, []) == []
    assert provenance_excluded(concepts, None) == []


def test_exclusion_is_not_substring_matching():
    """MUTATION GUARD (4b). A `c.startswith(t)` implementation would pass the
    tests above while also eating `TURNOVER`-style names and, worse, any
    concept whose type merely begins with an excluded type. Only an exact
    match on the pre-':' segment is correct."""
    concepts = ["HELIXIN", "STRANDED", "TURNIP", "COMPBIASED"]
    assert provenance_excluded(
        concepts, ["HELIX", "STRAND", "TURN", "COMPBIAS"]) == []


# ------------------------------------ N2: cluster map + independence axis --

def test_load_cluster_map_normalises_BOTH_columns(tmp_path):
    """Gotcha 10. The TSV holds whole FASTA tokens in both columns while the
    rest of the pipeline keys on bare accessions. Normalising only one side
    would silently produce a map that never matches -- the same failure that
    once matched 0 of 53,672 proteins while exiting 0."""
    from src.annotations.build_residue_labels import load_cluster_map
    t = tmp_path / "clu.tsv"
    t.write_text("sp|P00001|A_HUMAN\tsp|P00002|B_HUMAN\n"
                 "sp|P00001|A_HUMAN\tsp|P00001|A_HUMAN\n")
    m = load_cluster_map(t)
    assert m == {"P00002": "P00001", "P00001": "P00001"}


def test_load_cluster_map_hard_fails_on_an_empty_parse(tmp_path):
    """GOTCHA 13: the guard must fire. An empty cluster map would make every
    concept look like a 0-cluster concept -- a silently wrong stratification
    axis rather than a loud failure."""
    from src.annotations.build_residue_labels import load_cluster_map
    t = tmp_path / "clu.tsv"
    t.write_text("not-a-tsv-line\nanother\n")
    with pytest.raises(SystemExit, match="no cluster assignments"):
        load_cluster_map(t)


def test_cluster_map_ignores_malformed_rows_without_dying(tmp_path):
    from src.annotations.build_residue_labels import load_cluster_map
    t = tmp_path / "clu.tsv"
    t.write_text("sp|P1|A\tsp|P2|B\nJUNK\n\nsp|P1|A\tsp|P1|A\n")
    assert load_cluster_map(t) == {"P2": "P1", "P1": "P1"}


# ------------------------------------------------ amendment CR, Row B --
# The sub-typed feature types come from the config and from nowhere else. Until
# CR a four-type code default governed silently whenever the key was absent.

def test_occurrences_require_the_subtype_set_to_be_passed():
    """No default: omitting the set is a TypeError at the call, not a silent
    four-type set."""
    import pytest
    feats = [Feature("DOMAIN", 1, 8, None, (), "Some Domain")]
    with pytest.raises(TypeError):
        features_to_concept_occurrences(feats)               # noqa: no kwarg
    with pytest.raises(TypeError):
        features_to_concept_positions(feats)


def _cr_inputs(tmp_path):
    """P33333 (DAT_DOMAINS): two DOMAIN /notes, a REGION /note, a ZN_FING
    /note and a MOD_RES /note on a 20-residue protein."""
    dat = _write_dat_domains(tmp_path)
    index = [{"accession": "sp|P33333|TEST3_HUMAN", "start": 0, "n_rows": 20,
              "seq_len": 20, "truncated": False, "residue_ids": None}]
    return dat, _write_cache(tmp_path, index, n_rows=20)


def test_build_refuses_a_config_without_subtype_note_types(tmp_path, monkeypatch):
    import pytest
    dat, cache = _cr_inputs(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n")     # key absent
    out = tmp_path / "labels.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat),
        "--cache", str(cache), "--out", str(out),
    ])
    with pytest.raises(SystemExit) as e:
        main()
    assert "annotations.subtype_note_types" in str(e.value)
    assert not out.exists()


def test_no_note_subconcepts_is_the_explicit_way_to_run_without_the_key(
        tmp_path, monkeypatch):
    dat, cache = _cr_inputs(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n")     # key absent
    out = tmp_path / "labels.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat),
        "--cache", str(cache), "--out", str(out), "--no-note-subconcepts",
    ])
    main()
    with h5py.File(out, "r") as hf:
        names = list(hf["concepts"])
    assert "DOMAIN" in names and not any(n.startswith("DOMAIN:") for n in names)


def test_the_configured_set_governs_which_types_are_subtyped(tmp_path, monkeypatch):
    """The set is the config's, not a code constant: with only DOMAIN listed,
    a REGION /note yields no `REGION:` concept -- the four-type default would
    have produced one."""
    dat, cache = _cr_inputs(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("annotations:\n  min_positive_residues: 1\n"
                   "  subtype_note_types: [DOMAIN]\n")
    out = tmp_path / "labels.h5"
    monkeypatch.setattr(sys, "argv", [
        "build", "--config", str(cfg), "--dat", str(dat),
        "--cache", str(cache), "--out", str(out),
    ])
    main()
    with h5py.File(out, "r") as hf:
        names = list(hf["concepts"])
    assert any(n.startswith("DOMAIN:") for n in names)
    assert not any(n.startswith("REGION:") for n in names)
