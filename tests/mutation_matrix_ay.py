"""Mutation matrix for amendment AY's new code. NOT collected by pytest.

Covers `concept_bridge` (AY sections 2a/2c/3) and `carrier_identity` (AY §2b).

Run manually:  python tests/mutation_matrix_ay.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control mutation,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-exit-code rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run, run_one  # noqa: E402

ROOT = os.getcwd()
B = os.path.join(ROOT, "src", "analysis", "concept_bridge.py")
C = os.path.join(ROOT, "src", "analysis", "carrier_identity.py")
I = os.path.join(ROOT, "src", "analysis", "concept_inventory.py")
TB = "tests/test_concept_bridge.py"
TC = "tests/test_carrier_identity.py"
TI = "tests/test_concept_inventory.py"

# THE CONTROL: make the bit-identity acceptance criterion a constant True.
# This cannot fail to break the suite -- the permuted-reference fixture asserts
# DIFFERS on an array with the SAME TOTAL. If the runner reports this
# undetected, none of its SURVIVED verdicts mean anything and none is printed.
CONTROL = (
    "CONTROL -- the acceptance criterion replaced by a constant", C,
    b"    bit = bool(same_shape and ref.dtype == counts_numeric.dtype\n"
    b"               and ref.tobytes()\n"
    b"               == np.ascontiguousarray(counts_numeric).tobytes())",
    b"    bit = True",
    [f"{TC}::test_acceptance_FAILS_on_a_permutation_with_the_SAME_TOTAL"],
)

MUTATIONS = [
    # ------------------------------------------------- carrier_identity --
    # The acceptance criterion decided on the TOTAL instead of the bytes. This
    # is the exact defect gotcha 27 is about, and the fixture is built with a
    # same-total permutation so only the elementwise form survives it.
    ("acceptance: bit-identity becomes a TOTAL comparison", C,
     b"    bit = bool(same_shape and ref.dtype == counts_numeric.dtype\n"
     b"               and ref.tobytes()\n"
     b"               == np.ascontiguousarray(counts_numeric).tobytes())",
     b"    bit = bool(same_shape and int(ref.sum()) == int(counts_numeric.sum()))",
     [f"{TC}::test_acceptance_FAILS_on_a_permutation_with_the_SAME_TOTAL",
      f"{TC}::test_acceptance_applies_the_numeric_permutation_it_is_given"]),

    # The permutation argument ignored. Numeric and lexicographic orders agree
    # only at positions 0 and 1, so this mis-pairs 18 of 20 rows in production
    # while every identity key still agrees (gotcha 29).
    ("acceptance: the numeric permutation is not applied", C,
     b"    counts_numeric = counts_lex[numeric_order]",
     b"    counts_numeric = counts_lex",
     [f"{TC}::test_acceptance_applies_the_numeric_permutation_it_is_given"]),

    # ABSENT must never read as AGREEMENT (gotcha 15 / I3).
    ("acceptance: an absent identity key is treated as matching", C,
     b'            problems.append(f"{key}: ABSENT (absent is not matching)")',
     b'            pass',
     [f"{TC}::test_acceptance_reports_an_absent_identity_key_as_absent"]),

    # Seed ordering by FILENAME rather than by the checkpoint's own meta seed.
    ("order_decoders_to: keyed on the path instead of meta['seed']", C,
     b"    return [by_seed[s] for s in want]",
     b"    return sorted(meta_seed_by_path, key=lambda p: str(p))",
     [f"{TC}::test_order_decoders_to_uses_META_seed_not_the_filename",
      f"{TC}::test_order_decoders_to_permutes_LEXICOGRAPHIC_against_NUMERIC"]),

    # A checkpoint set that is not the anchor set silently changes what every
    # rate describes. Dropping the refusal makes that invisible.
    ("order_decoders_to: extra checkpoints silently dropped", C,
     b"    if extra:\n        raise SystemExit(",
     b"    if False:\n        raise SystemExit(",
     [f"{TC}::test_order_decoders_to_is_fatal_on_every_mismatch"]),

    # theta ignored in the partner map: every mutual pair would count as a
    # match regardless of how weak the cosine is.
    ("mnn_partner_matrix: theta is not applied", C,
     b"            ok_a = (mut_a & (a_sim >= theta)).cpu().numpy()\n"
     b"            ok_b = (mut_b & (b_sim >= theta)).cpu().numpy()",
     b"            ok_a = mut_a.cpu().numpy()\n"
     b"            ok_b = mut_b.cpu().numpy()",
     [f"{TC}::test_mnn_partner_matrix_respects_theta"]),

    # Mutuality dropped -- MNN degenerates to one-way argmax, which amendment
    # W1 establishes is a strictly larger match set.
    ("mnn_partner_matrix: mutuality is not required", C,
     b"            ok_a = (mut_a & (a_sim >= theta)).cpu().numpy()",
     b"            ok_a = (a_sim >= theta).cpu().numpy()",
     [f"{TC}::test_mnn_partner_matrix_finds_the_mutual_pair_and_only_it"]),

    # THE LOAD-BEARING DISTINCTION. B collapses into A: "the carrier has a
    # partner" reported as "the carrier's partner is the other carrier".
    ("carrier_rates: B collapses into A", C,
     b"            b = p == ct",
     b"            b = p >= 0",
     [f"{TC}::test_carrier_rates_separate_B_from_A_when_the_partner_is_the_WRONG_feature"]),

    # The carrier included in its own firing-rate-matched baseline -- the
    # comparison would then be partly a feature measured against itself.
    # SCOPED TO THE HAND CASE ONLY: the singleton fixture exercises the `ok`
    # guard, not the exclusion, and listing it here would demand a kill from a
    # fixture structurally unable to deliver one.
    ("carrier_rates: the carrier is not excluded from its own baseline", C,
     b"                num = hit_in[ok] - a[ok].astype(np.int64)\n"
     b"                den = n_in[ok] - 1",
     b"                num = hit_in[ok]\n"
     b"                den = n_in[ok]",
     [f"{TC}::test_carrier_rates_match_the_hand_computation"]),

    # The usability guard on the stratum size. With `> 0` a singleton stratum
    # divides by zero after the carrier is excluded, and reports a rate instead
    # of counting the carrier as unusable.
    ("carrier_rates: the stratum-size guard admits singletons", C,
     b"            ok = n_in > 1",
     b"            ok = n_in > 0",
     [f"{TC}::test_carrier_rates_report_an_unusable_singleton_stratum"]),

    # Concepts not detected in BOTH seeds admitted into the denominator.
    ("carrier_rates: the both-detected restriction is dropped", C,
     b"            both = detected[s] & detected[t]",
     b"            both = detected[s] | detected[t]",
     [f"{TC}::test_carrier_rates_skip_concepts_not_detected_in_both"]),

    # Ties in the argmax reported as absent. A "carrier" chosen by tie-break is
    # not a carrier, and the count is how a reader sees whether any exist.
    ("carriers_from_ap: the tie count is suppressed", C,
     b"        \"n_cells_with_tied_argmax\": int((n_tied > 1).sum()),",
     b"        \"n_cells_with_tied_argmax\": 0,",
     [f"{TC}::test_carriers_from_ap_reports_ties_rather_than_hiding_them"]),

    # AY-P2's boundary. "Fewer than 40%" is strict; `<=` would score a MISS as
    # a HIT at exactly 0.40.
    ("AY-P2: the strict `<` boundary becomes `<=`", C,
     b'                        else "UNDER_40_PERCENT" if share < 0.40',
     b'                        else "UNDER_40_PERCENT" if share <= 0.40',
     [f"{TC}::test_AY_P2_boundary_is_STRICTLY_under_forty_percent"]),

    # The standing nine strata replaced by a private binning, which would make
    # this comparison incomparable to every other stratified result here.
    ("firing_bin: a private binning replaces BIN_EDGES", C,
     b"    edges = np.asarray(BIN_EDGES, dtype=np.float64)",
     b"    edges = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)",
     [f"{TC}::test_firing_bin_uses_the_standing_strata_edges"]),

    # --------------------------------------------------- concept_bridge --
    # The schedule's RANK is the discriminating fact. Hard-coding the shared
    # verdict makes the report unable to say anything else -- gotcha 25's
    # "can only produce one outcome".
    ("null construction: the shared verdict is unconditional", B,
     b'    elif dest["has_seed_axis"]:',
     b'    elif False:',
     [f"{TB}::test_per_seed_schedule_is_DISTINGUISHED_from_a_shared_one"]),

    ("null construction: an absent schedule reads as shared", B,
     b'    if not dest.get("present"):',
     b'    if False:',
     [f"{TB}::test_absent_schedule_is_reported_absent_not_as_shared"]),

    # A zero threshold turning into +inf reads as "cleared by an enormous
    # margin" when the ratio is in fact undefined.
    ("detection_margin: a zero threshold yields +inf, not NaN", B,
     b"    out = np.divide(observed_gain, thr,\n"
     b"                    out=np.full(observed_gain.shape, np.nan),\n"
     b"                    where=thr > 0)",
     b"    with np.errstate(divide='ignore', invalid='ignore'):\n"
     b"        out = observed_gain / thr",
     [f"{TB}::test_detection_margin_divides_and_refuses_a_nonpositive_threshold"]),

    # The WEAKEST seed is the right reduction: a concept is detected in all 20
    # seeds only as securely as its worst one does it.
    ("margin_summary: the mean over seeds replaces the minimum", B,
     b"    m = np.nanmin(margin[:, group_mask], axis=0)",
     b"    m = np.nanmean(margin[:, group_mask], axis=0)",
     [f"{TB}::test_margin_summary_takes_the_WEAKEST_seed"]),

    # An empty group reported as a number rather than named.
    ("margin_summary: an empty group is not named", B,
     b'        return {"n_concepts": 0, "status": "EMPTY_GROUP"}',
     b'        return {"n_concepts": 0, "status": "COMPUTED"}',
     [f"{TB}::test_margin_summary_names_an_empty_group_rather_than_returning_zero"]),

    # Splitting on the LAST colon breaks every descriptor that contains one.
    ("concept name: the split moves to the last colon", B,
     b'    return concept.split(":", 1)[0]',
     b'    return concept.rsplit(":", 1)[0]',
     [f"{TB}::test_feature_key_and_descriptor_split_on_the_first_colon"]),

    # The saturation threshold invented from an empty set.
    ("saturation: NEVER_SATURATES returns a number anyway", B,
     b'        return {"status": "NEVER_SATURATES", "threshold": None,',
     b'        return {"status": "NEVER_SATURATES", "threshold": largest_unsat,',
     [f"{TB}::test_saturation_refuses_to_invent_a_threshold_when_it_never_saturates"]),

    # Off by one region: taking the largest unsaturated size itself as the
    # threshold would report a concept that is NOT detected everywhere as being
    # above the saturation point.
    ("saturation: the threshold becomes the largest unsaturated size", B,
     b"    thr = float(size[above].min())",
     b"    thr = largest_unsat",
     [f"{TB}::test_saturation_returns_the_smallest_size_above_the_largest_unsaturated"]),

    # THE INTERVAL OF RECORD. Wilson understates the between-concept spread,
    # which is the whole substitution AY §3 makes.
    ("binned_curve: the interval reverts to the naive binomial SE", B,
     b"        se = sd / np.sqrt(n) if n > 1 else float(\"nan\")",
     b"        se = np.sqrt(v.mean() * (1 - v.mean()) / n) if n > 1 else float(\"nan\")",
     [f"{TB}::test_binned_curve_uses_between_concept_dispersion_not_wilson"]),

    # ddof: the population SD understates the between-concept dispersion the
    # interval is supposed to carry.
    ("binned_curve: ddof=1 becomes the population SD", B,
     b"        sd = float(v.std(ddof=1)) if n > 1 else float(\"nan\")",
     b"        sd = float(v.std()) if n > 1 else float(\"nan\")",
     [f"{TB}::test_binned_curve_uses_between_concept_dispersion_not_wilson"]),

    # An empty bin crashing or, worse, silently vanishing from the table.
    ("binned_curve: an empty bin is dropped instead of named", B,
     b'            rows.append({"bin": b, "n_concepts": 0, "status": "EMPTY"})\n'
     b"            continue",
     b"            continue",
     [f"{TB}::test_binned_curve_reports_an_empty_bin_rather_than_crashing"]),

    # The two experimental tiers summed rather than unioned -- a residue in
    # both would be counted twice.
    ("experimental counts: the tiers are summed, not unioned", B,
     b"            out[c] = int(np.unique(np.concatenate(rows)).size) if rows else 0",
     b"            out[c] = int(sum(r.size for r in rows)) if rows else 0",
     [f"{TB}::test_experimental_counts_UNION_the_two_tiers"]),

    # The positional guard removed from the DRIVER. Its unit test would still
    # pass; only an end-to-end fixture can see that `run` stopped calling it.
    ("run: the seed-axis guard is not called", B,
     b'    assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)\n'
     b"    pool_concepts",
     b"    pool_concepts",
     [f"{TB}::test_run_REFUSES_a_seed_axis_mismatch"]),

    # ------------------------------------------------- concept_inventory --
    # THE AO6 DEFECT ITSELF: the stored accession-order starts used as though
    # they were in row order. 0.0031% of rows were right last time this shipped.
    ("block map: the argsort into row order is dropped", I,
     b'    order = np.argsort(start, kind="mergesort")',
     b"    order = np.arange(start.size)",
     [f"{TI}::test_block_map_argsorts_and_REPORTS_that_it_had_to",
      f"{TI}::test_protein_of_row_maps_each_row_into_its_own_block",
      f"{TI}::test_run_end_to_end"]),

    # The nesting check reporting a percentage instead of refusing. This column
    # is read as fact by an external reviewer.
    ("nesting check: STRADDLES is reported, not refused", I,
     b'    if report["verdict"] == "STRADDLES":\n        raise SystemExit(',
     b'    if False:\n        raise SystemExit(',
     [f"{TI}::test_nesting_check_sees_the_ACCESSION_ORDER_map",
      f"{TI}::test_nesting_check_catches_a_span_crossing_a_boundary"]),

    # The end-of-block bound dropped: a span running past its protein's last
    # row would then count as nested.
    ("nesting check: the block END bound is not applied", I,
     b"                      & (np.maximum(e - 1, s)\n"
     b'                         < block["start"][np.clip(pa, 0, None)]\n'
     b'                         + block["n_rows"][np.clip(pa, 0, None)]))',
     b"                      & True)",
     [f"{TI}::test_nesting_check_sees_the_ACCESSION_ORDER_map"]),

    # Distinct PROTEINS, not instances. Without the unique a concept with six
    # instances in one protein would be reported as six proteins -- and the G3
    # gate's own justification is about independent protein support.
    ("proteins_per_concept: instances counted instead of proteins", I,
     b'            out[c] = int(np.unique(protein_of_row(s, block)).size) if s.size \\\n'
     b"                else 0",
     b"            out[c] = int(protein_of_row(s, block).size) if s.size \\\n"
     b"                else 0",
     [f"{TI}::test_proteins_per_concept_counts_DISTINCT_proteins"]),

    # An absent instance record reported as zero proteins. Different facts.
    ("proteins_per_concept: absent becomes 0 rather than None", I,
     b"                out[c] = None\n                continue",
     b"                out[c] = 0\n                continue",
     [f"{TI}::test_proteins_per_concept_counts_DISTINCT_proteins"]),

    ("build_rows: an absent protein count is written as 0", I,
     b'            "n_proteins": "" if npr is None else npr,',
     b'            "n_proteins": 0 if npr is None else npr,',
     [f"{TI}::test_build_rows_leaves_an_absent_protein_count_BLANK_not_zero"]),

    # A short inventory would be reviewed as though complete.
    ("assert_row_count: the library-coverage check is disabled", I,
     b"    if len(rows) != expected:",
     b"    if False:",
     [f"{TI}::test_assert_row_count_refuses_a_short_inventory"]),

    # The csv module's CRLF default. Reading the file back through csv would
    # normalise the terminator and see nothing, so the test asserts raw bytes.
    ("write_csv: the LF line terminator reverts to the csv default", I,
     b'        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\\n")',
     b"        w = csv.DictWriter(fh, fieldnames=COLUMNS)",
     [f"{TI}::test_write_csv_uses_LF_so_a_rebuild_does_not_dirty_the_checkout"]),

    ("run: the pool arithmetic is not checked", I,
     b"    if len(concepts) - len(parents) != argmax_pool_n:",
     b"    if False:",
     [f"{TI}::test_run_refuses_when_the_pool_arithmetic_disagrees"]),

    ("run: the concept axis is not checked against n_concepts", I,
     b"    if len(concepts) != n_concepts_attr:",
     b"    if False:",
     [f"{TI}::test_run_refuses_when_the_concept_axis_contradicts_n_concepts"]),
]


def main() -> int:
    fixtures = sorted({t for e in MUTATIONS for t in e[4]} | set(CONTROL[4]))
    failing = [t for t in fixtures if not run_one(t)]
    if failing:
        print("BASELINE BROKEN -- these fail before any mutation:")
        for t in failing:
            print(f"   {t}")
        return 1
    print(f"baseline: {len(fixtures)}/{len(fixtures)} fixtures pass")

    name, path, old, new, fx = CONTROL
    survived = apply_and_run(path, old, new, fx)
    if survived is None:
        print(f"CONTROL ANCHOR NOT FOUND in {path} -- aborting. The runner "
              "cannot be trusted to detect anything.")
        return 2
    if survived:
        print(f"CONTROL NOT DETECTED ({name}): {survived}. Aborting; no "
              "SURVIVED verdict from this run would mean anything.")
        return 2
    print(f"control detected: {name}")

    killed, problems = 0, []
    for name, path, old, new, fx in MUTATIONS:
        survived = apply_and_run(path, old, new, fx)
        if survived is None:
            problems.append(f"ANCHOR NOT FOUND (or not unique): {name}")
            print(f"  ANCHOR  {name}")
            continue
        if survived:
            problems.append(f"SURVIVED: {name} -> {survived}")
            print(f"  SURVIVED {name}")
        else:
            killed += 1
            print(f"  killed   {name}")

    print(f"\n{killed}/{len(MUTATIONS)} mutations killed, control detected")
    for p in problems:
        print(f"  {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
