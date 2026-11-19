"""Mutation matrix for amendment CR -- the config-read and refusal changes in
Rows A, B and C. NOT collected by pytest.

Run manually:  python tests/mutation_matrix_cr.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is `mutation_matrix_av.apply_and_run` -- control-first, `__pycache__`
purged after every restore, byte-compared restore, decisions on pytest's EXIT
CODE (gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. Every change in CR is a config read or a refusal,
and the characteristic defect of a refusal is that it can only pass (gotcha
25): a check whose failure branch is unreachable looks identical, from the
outside, to a check that works. So each refusal gets one mutant that removes
it or reinstates the literal it replaced, and each mutant names the fixture
that must go red -- a fixture that stays green under the mutant is a check that
guards nothing.

THE CONTROL is not one of CR's changes. It removes the pre-existing
`MIN_MATCH_FRACTION` refusal in `build_residue_labels.py` (the gotcha-10 guard
that once let 0 of 53,672 proteins match and exit 0), which the pre-existing
`test_low_match_rate_fails_loudly` must kill -- so the runner is shown to detect
a real removed refusal before any verdict on CR's own refusals is printed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402

ROOT = os.getcwd()
XC = os.path.join(ROOT, "src", "experiment_config.py")
CLU = os.path.join(ROOT, "src", "data", "cluster_swissprot.py")
SPL = os.path.join(ROOT, "src", "data", "split_train_eval.py")
SHD = os.path.join(ROOT, "src", "embeddings", "shard_fasta.py")
CAC = os.path.join(ROOT, "src", "embeddings", "cache.py")
UNI = os.path.join(ROOT, "src", "annotations", "uniprot_sites.py")
BRL = os.path.join(ROOT, "src", "annotations", "build_residue_labels.py")
MTI = os.path.join(ROOT, "src", "annotations", "measure_truncation_impact.py")
D3J = os.path.join(ROOT, "src", "analysis", "design3_join.py")

T_XC = "tests/test_experiment_config.py"
T_CLU = "tests/test_cluster_swissprot.py"
T_SPL = "tests/test_split.py"
T_SHD = "tests/test_shard_fasta.py"
T_CAC = "tests/test_cache.py"
T_UNI = "tests/test_uniprot_sites.py"
T_MTI = "tests/test_measure_truncation_impact.py"
T_D3J = "tests/test_design3_join.py"

FOUR = b'["DOMAIN", "REGION", "MOTIF", "ZN_FING"]'

CONTROL = (
    "CONTROL -- the pre-existing MIN_MATCH_FRACTION refusal (gotcha 10) removed",
    BRL,
    b'    if frac < MIN_MATCH_FRACTION:\n'
    b'        raise SystemExit(\n'
    b'            f"FATAL: only {frac:.1%} of cache proteins were found in {args.dat} "\n',
    b'    if False and frac < MIN_MATCH_FRACTION:\n'
    b'        raise SystemExit(\n'
    b'            f"FATAL: only {frac:.1%} of cache proteins were found in {args.dat} "\n',
    [f"{T_UNI}::test_low_match_rate_fails_loudly"],
)

MUTATIONS = [
    # ========================================= 1. the helper's refusal branch
    ("A1  experiment_config.resolve: disagreement branch disabled", XC,
     b'    if cli_value != cfg_value:\n',
     b'    if False and cli_value != cfg_value:\n',
     [f"{T_XC}::test_a_disagreeing_cli_value_is_refused_naming_both_values",
      f"{T_CLU}::test_a_disagreeing_max_len_is_refused_before_any_work",
      f"{T_SPL}::test_a_disagreeing_max_len_is_refused",
      f"{T_SHD}::test_a_disagreeing_max_len_is_refused",
      f"{T_CAC}::test_resolution_refuses_a_disagreeing_layer"]),

    # ================================ 2-5. each script's literal default back
    ("A2  cluster_swissprot --max-len default restored to 1022", CLU,
     b'    p.add_argument("--max-len", type=int, default=None,\n'
     b'                   help="Residues are counted as min(len, max_len) to match "\n',
     b'    p.add_argument("--max-len", type=int, default=1022,\n'
     b'                   help="Residues are counted as min(len, max_len) to match "\n',
     [f"{T_CLU}::test_default_max_len_follows_the_config"]),

    ("A3  split_train_eval --max-len default restored to 1022", SPL,
     b'    p.add_argument("--max-len", type=int, default=None,\n'
     b'                   help="Residue cap for the report; matches the cache. Read "\n',
     b'    p.add_argument("--max-len", type=int, default=1022,\n'
     b'                   help="Residue cap for the report; matches the cache. Read "\n',
     [f"{T_SPL}::test_max_len_default_follows_the_config_not_a_literal"]),

    ("A4  shard_fasta --max-len default restored to 1022", SHD,
     b'    p.add_argument("--max-len", type=int, default=None,\n'
     b'                   help="Residue cap for balancing; read from model.max_len in "\n',
     b'    p.add_argument("--max-len", type=int, default=1022,\n'
     b'                   help="Residue cap for balancing; read from model.max_len in "\n',
     [f"{T_SHD}::test_max_len_default_follows_the_config_not_a_literal"]),

    ("A5  cache.py --model / --layer / --max-len literals restored", CAC,
     b'    p.add_argument("--model", default=None,\n'
     b'                   help="model.name from --config unless restated here.")\n'
     b'    p.add_argument("--layer", type=int, default=None,\n'
     b'                   help="model.layer from --config unless restated here.")\n'
     b'    p.add_argument("--fasta", type=Path, required=True)\n'
     b'    p.add_argument("--out", type=Path, required=True)\n'
     b'    p.add_argument("--batch-size", type=int, default=8)\n'
     b'    p.add_argument("--max-len", type=int, default=None,\n',
     b'    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D",\n'
     b'                   help="model.name from --config unless restated here.")\n'
     b'    p.add_argument("--layer", type=int, default=24,\n'
     b'                   help="model.layer from --config unless restated here.")\n'
     b'    p.add_argument("--fasta", type=Path, required=True)\n'
     b'    p.add_argument("--out", type=Path, required=True)\n'
     b'    p.add_argument("--batch-size", type=int, default=8)\n'
     b'    p.add_argument("--max-len", type=int, default=1022,\n',
     [f"{T_CAC}::test_parser_defaults_are_none_so_the_config_governs",
      f"{T_CAC}::test_resolution_fills_model_layer_and_max_len_from_the_config"]),

    # ================================== 6. the four-type code default is back
    ("B1  uniprot_sites: subtype_note_types default reinstated", UNI,
     b'    *,\n'
     b'    subtype_note_types: frozenset[str],\n'
     b') -> dict[str, list[Occurrence]]:\n',
     b'    *,\n'
     b'    subtype_note_types: frozenset[str] = frozenset({"DOMAIN", "REGION", "MOTIF", "ZN_FING"}),\n'
     b') -> dict[str, list[Occurrence]]:\n',
     [f"{T_UNI}::test_occurrences_require_the_subtype_set_to_be_passed"]),

    # ========================= 7. build_residue_labels: silent fallback back
    ("B2  build_residue_labels: absence refusal replaced by the old fallback", BRL,
     b'        subtype_note_types = frozenset(\n'
     b'            lookup(cfg, "annotations.subtype_note_types", args.config))\n',
     b'        subtype_note_types = frozenset(\n'
     b'            ann_cfg.get("subtype_note_types", ' + FOUR + b'))\n',
     [f"{T_UNI}::test_build_refuses_a_config_without_subtype_note_types"]),

    # ==================== 8. measure_truncation_impact: private fallback back
    ("B3  measure_truncation_impact: .get fallback reinstated", MTI,
     b'    subtype_types = frozenset(\n'
     b'        lookup(cfg, "annotations.subtype_note_types", args.config))\n',
     b'    subtype_types = frozenset(\n'
     b'        ann.get("subtype_note_types", ' + FOUR + b'))\n',
     [f"{T_MTI}::test_refuses_a_config_without_subtype_note_types"]),

    # ============================ 9. design3_join: equality assertion disabled
    ("C1  design3_join: floor equality assertion disabled", D3J,
     b'    if floor_cfg != ANALYSIS_MIN_FIRING_ROWS:\n',
     b'    if False and floor_cfg != ANALYSIS_MIN_FIRING_ROWS:\n',
     [f"{T_D3J}::test_a_config_floor_that_differs_is_refused_naming_both",
      f"{T_D3J}::test_main_refuses_on_the_floor_before_opening_any_artifact"]),
]


def main() -> int:
    label, path, old, new, expect = CONTROL
    survived = apply_and_run(path, old, new, expect)
    if survived is None:
        print(f"CONTROL ABORT: anchor for {label} did not match exactly once. "
              "The runner cannot be shown to detect, so no verdict is printed.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: {survived}). "
              "Every SURVIVED verdict this runner could print would be "
              "meaningless, so none is printed.")
        return 2
    print(f"CONTROL detected ({len(expect)} fixtures) -- the runner detects.\n")

    n_bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"ANCHOR MISS  {label}")
            n_bad += 1
            continue
        if survived:
            print(f"SURVIVED     {label}  -> {survived}")
            n_bad += len(survived)
        else:
            print(f"KILLED       {label}  ({len(expect)} fixtures)")
    killed = sum(1 for _ in MUTATIONS) if n_bad == 0 else "?"
    print(f"\n{killed} of {len(MUTATIONS)} mutations killed; survivors/misses: {n_bad}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
