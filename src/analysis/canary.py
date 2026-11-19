"""THE CANARY MANIFEST. PREREGISTRATION amendments AD1/AD2 and AM5.

WHAT THIS IS FOR. AD1 named four reproduction targets and AD2 fixed their
tolerance at BIT-IDENTICAL. That worked, and it lived as a procedure: a pair of
ad-hoc sbatch scripts plus a directory of copied outputs on `$PROJECT`. **A
baseline that exists only on scratch is not a baseline** -- it has no commit
date, it cannot be diffed, and nothing fails when it drifts.

AM5's addition, and it is the reason this file exists: **the N = 39 outputs are
now artifacts of record.** They carry the primary marginal, the primary mixture
fit, and the endpoint groups the analysis will consume. Nothing protected them.
Their baseline is captured here while nothing is in flight and the producing
commit is known.

=============================================================================
THE TWO ARMS STAY SEPARATE TARGETS, WITH TOLERANCE PER ARM
=============================================================================
Merging them would collapse exactly the distinction AK1 exists to keep:

  * **N = 19** is the SENSITIVITY arm of record. It must reproduce
    bit-identically, and it has a baseline predating this session.
  * **N = 39** is the PRIMARY arm. It must reproduce bit-identically **from
    here forward** -- there is no earlier baseline to compare against, and
    saying so is the point rather than an omission.

So `arm` is part of each target's identity, and `first_baselined_at` records
which commit the expectation starts from. A target whose baseline begins today
cannot be presented as having reproduced across a refactor it never saw.

=============================================================================
WHAT IS HASHED, AND THE ONE THING THAT CANNOT BE
=============================================================================
sha256 of the whole file, plus a LEAF COUNT -- the number of scalar leaves in a
JSON document, or dataset shapes and attr count for an h5. The leaf count is
not redundant with the hash: a hash says "something changed" and a leaf count
says "a field appeared or vanished", which is the failure the declared-additions
allowlist in `rf_extended_n.py` is about.

**A whole-file sha256 CANNOT be used for the M1 documents, and pretending
otherwise would build a check that can only fail** (gotcha 25a). They embed
`identity/git_sha` -- the commit that produced them -- so their hash moves on
every re-run by construction. For those targets the manifest records
`hash_scope: statistics_only` and the comparison is delegated to
`rf_extended_n.compare_n19`, which already excludes provenance by full path.
The h5 and mixture targets have no such field and are hashed whole.

**IS THE EXEMPTED CONTENT COVERED ELSEWHERE? YES, and stating it here is the
point -- otherwise the exemption reads as a gap** (amendment AN5a). What a
whole-file hash would have covered and this does not is the M1 documents'
NON-STATISTICAL content: the identity block and the pre-flight report. Both are
covered, by a stricter check than a hash:

  * `rf_extended_n.compare_n19` compares **every key the baseline carries** for
    exact equality, excluding only `identity/git_sha` and `identity/git_dirty`
    BY FULL PATH -- so `n_seeds`, `n_anchors`, `n_compare_seeds`,
    `n_pairs_computed`, the seed labels, the checkpoint hashes and the whole
    `preflight` subtree (including the ensemble's own shared `git_sha`) are all
    still compared, and an UNDECLARED added key is a failure.
  * `rf_marginal.verify_ensemble_identity` re-derives the ensemble identity from
    the 40 `meta.json` files on every run and hard-fails on disagreement, so the
    pre-flight block is regenerated rather than merely remembered.

So the only content outside every check is the two provenance fields whose whole
purpose is to differ. The leaf count here adds the one thing neither of those
covers: **a field appearing or vanishing at a path the baseline never had.**

=============================================================================
THE `BIT_IDENTICAL` TOLERANCE IS CONDITIONAL ON DEVICE CLASS (amendment BC7)
=============================================================================
**A BIT-IDENTITY CRITERION ALSO ASSERTS A DEVICE CLASS, and until BC that
condition lived only in a defect log.** Measured at BB12 item 3: re-running the
carrier job on a CPU partition instead of `rtx_5000_ada` produced MNN counts
differing in **4 of 204,800 cells** (1,929,312 against 1,929,316, **2.1e-6**),
consistent with cosine similarities landing within float32 rounding of
`theta = 0.70`. The acceptance criterion refused, wrote nothing and exited
non-zero -- which is the criterion working, not a defect.

**Why this belongs HERE and not only in the defect log.** A future session that
regenerates a target on a different GPU type would see a `DIFFERS` verdict with
no stated reason, and the two available responses -- widen the tolerance, or
hunt for a phantom regression -- are both wrong. Widening a tolerance because an
undocumented condition was violated is how a reproduction guarantee quietly
stops guaranteeing anything.

**Nothing here widens any tolerance and no target is added.** This makes an
existing guarantee say what it actually guarantees.

**AND THE PER-TARGET CLAIM IS BOUNDED BY WHAT HAS BEEN MEASURED.** Only the
mechanism BB12.3 measured is asserted: `MNN_COSINE_THRESHOLD` where a target's
content descends from that float comparison, `NOT_ESTABLISHED` otherwise.
**`NOT_ESTABLISHED` DOES NOT MEAN DEVICE-INDEPENDENT** -- it means nobody has
measured it, and an absent measurement is reported as absent rather than read as
agreement (gotcha 15).

=============================================================================
THE `build_inputs` ARM CARRIES A WEAKER CLAIM THAN EVERY ARM ABOVE (BK)
=============================================================================
**FILE INTEGRITY, NOT REPRODUCTION, AND THE DIFFERENCE IS THE WHOLE POINT.**
Every arm above says *this artifact reproduces*. `build_inputs` says only *this
artifact has not changed on disk*. It is a weaker guarantee and it is labelled
weaker here, in `TOLERANCE["build_inputs"]`, and in each target's `why`.

**WHY THE WEAKER CLAIM IS THE RIGHT ONE HERE rather than a shortfall.** What
threatens the classification build's six inputs between now and its run is a
silent overwrite, a partial write, or an unrecorded regeneration -- and a prior
session did overwrite `bg3_propagation_ac7.json` in place, so this is observed
rather than hypothetical. A whole-file sha256 catches all three. It does NOT
catch code drift, and no target on this arm may be read as saying it does.

**AND A NUMERIC TOLERANCE IS NOT AVAILABLE WITHOUT A PRODUCING RE-RUN.** Five
of the six cannot be re-derived bit-for-bit (gotcha 35: `index_add_` accumulates
atomically on CUDA and `use_deterministic_algorithms` is enabled nowhere here;
and the pooled AP ran with `--ap-device auto`, which moves the arithmetic to CPU
above 150M non-zeros so associativity alone shifts it). **THE POOLED NULL'S
MEASURED FLOOR IS NOT TRANSFERRED TO ANY OF THEM.** That floor was measured on
`null_pilot_k.h5`, on a CUDA path whose atomics are the stated mechanism; four
of the six new inputs were produced on **CPU**, where whether `index_add_`
varies run to run is **unknown and is not assumed in either direction**. A
constant carried across artifacts is the propagation failure this project keeps
catching, and "the CUDA argument does not apply on CPU" is a mechanism claim,
which is not an input to a decision (gotcha 22). **If a bit-identity claim is
wanted for a CPU-produced artifact it is MEASURED or it is not made.**

**PRECEDENT, and it is exact:** the pre-migration git bundle is verified by
file-integrity against a recorded hash rather than by a rebuild, because bundles
are not byte-reproducible. Same distinction, already ratified.

=============================================================================
THE DEVICE FIELDS ARE THREE, NOT ONE (BK, ratified by the author)
=============================================================================
BJ-C3 returned the canary question unanswered because BC7's two-value
`device_class` could not describe an AP or null artifact honestly. The answer
is three separate fields, because they answer three different questions and
this record has already paid for a key whose name said one thing while it fired
for another (the renamed unverifiable cell at BE3.2):

  1. **`device_class` -- WHICH MEASURED MECHANISM makes this target
     device-sensitive**, so that a `DIFFERS` verdict after a device change is
     diagnosable. It gains `TOPK_GATHER_NEAR_TIE` alongside
     `MNN_COSINE_THRESHOLD`. `NOT_ESTABLISHED` keeps its meaning exactly -- no
     device-sensitivity measurement exists -- and continues NOT to mean
     device-independent.
  2. **`build_device_class` -- WHAT TO REPRODUCE ON**, per target. Previously
     one module constant naming one GPU type, which is simply wrong for the
     four inputs whose numeric content was computed on CPU.
     `BASELINE_DEVICE_CLASS` is RETAINED as a documented default with its scope
     stated, so a target that does not state its own is not silently
     re-described -- and existing targets state none, so none moves.
  3. **`build_device_provenance` -- HOW (2) WAS ESTABLISHED**, `MEASURED`
     against `INFERRED_FROM_PRODUCING_JOB_LOGS`. All six additions are
     inferred. Recording an inferred device with no such flag would make it
     indistinguishable from a measured one, which is exactly what the identity
     schema's no-back-fill rule protects. **Nothing is back-filled.**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# THE DEVICE-CLASS VOCABULARY (amendment BC7). TWO VALUES, and the second one
# is a statement of IGNORANCE rather than of independence.
# ---------------------------------------------------------------------------
# The target's content descends from the MNN cosine comparison against `theta`,
# which BB12.3 MEASURED as device-sensitive: 4 of 204,800 cells moved between
# a CPU partition and `rtx_5000_ada`. Its BIT_IDENTICAL tolerance therefore
# also asserts the device class.
MNN = "MNN_COSINE_THRESHOLD"
# No MNN threshold is known to lie in this target's production path AND no
# device-sensitivity measurement exists for it. **THIS IS NOT A CLAIM OF DEVICE
# INDEPENDENCE.** Establishing that would need the target regenerated on a
# second device class, which nobody has done. An absent measurement is reported
# as absent, never read as agreement (gotcha 15).
# AMENDMENT BK. The target's content descends from the TOPK GATHER, which
# gotcha 36 MEASURED as device-sensitive on this exact artifact family: CPU and
# `rtx_5000_ada` give different `fire_counts` for the same checkpoint on the
# same rows, because near-tie pre-activations swap under a float32 matmul and
# top-k then selects a different feature set (`nnz` is identical either way --
# 9,018 of 204,800 (seed, feature) pairs differ, 4.40%). It is a SEPARATE value
# from `MNN` rather than a widening of it: widening `MNN_COSINE_THRESHOLD` to
# mean "some measured device-sensitive mechanism" would collapse the per-target
# distinction BC7 drew, and the two mechanisms have different remedies.
TOPK_GATHER = "TOPK_GATHER_NEAR_TIE"
UNKNOWN = "NOT_ESTABLISHED"
DEVICE_CLASSES = (MNN, TOPK_GATHER, UNKNOWN)

# The device class every GPU job in this project requests, and therefore the
# class the MNN-descended baselines were produced under. It is a CONSEQUENCE OF
# A LOCKED DECISION (CLAUDE.md gotcha 7: always request the type, because
# `gpu_devel` is heterogeneous and torch cu126 has no Blackwell kernels), not a
# per-target measurement -- and it is recorded at module level rather than per
# target so it cannot be mistaken for one.
#
# **AMENDMENT BK NARROWS ITS SCOPE RATHER THAN MOVING IT.** It is now the
# DOCUMENTED DEFAULT for a target that states no `build_device_class` of its
# own, and every target that predates BK states none -- so this constant still
# describes exactly the targets it described before, and none is re-described.
# It is NOT a claim about the `build_inputs` arm, whose targets each state their
# own.
BASELINE_DEVICE_CLASS = "rtx_5000_ada"

#: What a target's `build_device_class` may say: the device class its NUMERIC
#: CONTENT was computed on -- for an AP or null artifact, the class the TopK
#: GATHER ran on, which is not always the class the producing job requested and
#: is never the class of a downstream reslice or reduction.
BUILD_DEVICE_CLASSES = ("rtx_5000_ada", "cpu")

#: How a `build_device_class` was established. **The two are not
#: interchangeable and must not be collapsed**: an inferred device recorded with
#: no flag is indistinguishable from a measured one, which is the state the
#: identity schema's no-back-fill rule exists to prevent (gotcha 15).
#: `MEASURED` means the producing code recorded the device it actually ran on
#: (`artifact_identity.gather_device_class`, NEW ARTIFACTS ONLY).
#: `INFERRED_FROM_PRODUCING_JOB_LOGS` means it was read off the job's own log
#: and `sacct` afterwards -- strong evidence, and still not the artifact
#: speaking for itself.
BUILD_DEVICE_PROVENANCE = ("MEASURED", "INFERRED_FROM_PRODUCING_JOB_LOGS")

#: Bumped whenever a per-target FIELD is added, so a manifest written by an
#: older module is distinguishable from one that simply omitted a value.
#: 1 = AD1/AD2/AM5/BB4/BC7. 2 = BK (`build_device_class`,
#: `build_device_provenance`, and `describe`'s recursive dataset walk).
MANIFEST_SCHEMA_VERSION = 2

DEVICE_CLASS_CONDITION = (
    "A BIT_IDENTICAL tolerance also asserts a DEVICE CLASS. Measured at BB12.3: "
    "the MNN counts differ in 4 of 204,800 cells (1.95e-5; the totals, "
    "1,929,312 against 1,929,316, differ by 2.1e-6 relative) between a CPU "
    "partition and rtx_5000_ada, consistent with cosine similarities landing "
    "within float32 rounding of theta = 0.70. A target marked "
    f"'{MNN}' that reports DIFFERS after being regenerated on another device "
    "class has NOT necessarily drifted -- and the answer is to reproduce it on "
    f"{BASELINE_DEVICE_CLASS}, never to widen the tolerance. "
    f"'{UNKNOWN}' means no device-sensitivity measurement exists for that "
    "target; it does NOT mean device-independent. "
    f"'{TOPK_GATHER}' (amendment BK) names the OTHER measured mechanism: the "
    "TopK gather selects a different feature set on CPU than on rtx_5000_ada "
    "because near-tie pre-activations swap under a float32 matmul -- 9,018 of "
    "204,800 (seed, feature) fire_counts pairs, 4.40%, for the same checkpoint "
    "on the same rows (gotcha 36). It is a DIFFERENT mechanism from the MNN "
    "cosine threshold and is named separately so a DIFFERS verdict says which "
    "one to look at. "
    "A target's own 'build_device_class' says WHAT TO REPRODUCE ON and is the "
    f"field to read; where a target states none, '{BASELINE_DEVICE_CLASS}' "
    "above is the documented default and is NOT a per-target measurement. "
    "'build_device_provenance' says how that value was established, and "
    "INFERRED_FROM_PRODUCING_JOB_LOGS is not MEASURED.")

# ---------------------------------------------------------------------------
# The declared targets. `arm` is part of the identity; `hash_scope` says whether
# a whole-file hash is meaningful for that target at all; `device_class` says
# whether its bit-identity expectation is conditional on the device (BC7).
# ---------------------------------------------------------------------------
TARGETS: tuple[dict, ...] = (
    # ---- AD1's original four, N = 19 ------------------------------------
    {"name": "rf_marginal_n19", "arm": "n19", "path": "rf_marginal_m1_n19.json",
     "hash_scope": "statistics_only", "device_class": MNN,
     "why": "AD1 canary for the 20 decoders, the identity attrs and the X7 firewall"},
    {"name": "rf_counts_n19", "arm": "n19", "path": "rf_counts_theta070_n19.h5",
     "hash_scope": "whole_file", "device_class": MNN,
     "why": "the per-anchor MNN counts every downstream r_f consumer reads"},
    {"name": "cv_ci_hardbin", "arm": "n19", "path": "cv_ci_hardbin.json",
     "hash_scope": "whole_file", "device_class": UNKNOWN,
     "why": "AD1 canary for mu_s, the strata and the concept axis"},
    {"name": "m3_degenerate", "arm": "n19", "path": "mu_s_degenerate_m3.json",
     "hash_scope": "whole_file", "device_class": UNKNOWN,
     "why": "AD1 canary for the 22 degenerate concepts and the per-seed logic"},
    {"name": "m4_c1_all_seeds", "arm": "n19", "path": "m4_c1_all_seeds.json",
     "hash_scope": "whole_file", "device_class": MNN,
     "why": "AD1 canary for the firewall-crossing job, mu_hat x r_f"},
    # ---- AM5's addition, N = 39: the artifacts of record -----------------
    {"name": "rf_marginal_n39", "arm": "n39", "path": "rf_marginal_m1_n39.json",
     "hash_scope": "statistics_only", "device_class": MNN,
     "why": "THE PRIMARY marginal: 20 anchors x 39 comparison seeds, 590 pairs"},
    {"name": "rf_counts_n39", "arm": "n39", "path": "rf_counts_theta070_n39.h5",
     "hash_scope": "whole_file", "device_class": MNN,
     "why": "THE PRIMARY per-anchor MNN counts at n_compare = 39"},
    {"name": "rf_mixture_n39", "arm": "n39", "path": "rf_mixture_n39.json",
     "hash_scope": "whole_file", "device_class": MNN,
     "why": "THE PRIMARY mixture fit, including the w0 profile AL6 turns on"},
    {"name": "rf_mixture_n19", "arm": "n19", "path": "rf_mixture_n19.json",
     "hash_scope": "whole_file", "device_class": MNN,
     "why": "the sensitivity-arm mixture, so the N-invariance check has both sides"},
    # ---- BB's addition: the ALIGNMENT PREDICTOR --------------------------
    # THIS IS AN ADDITION TO COVERAGE, NOT A REBASELINE OF ANYTHING EXISTING.
    # The two are different operations with different justifications and a
    # reader should not have to infer which happened (amendment BB section 4).
    #
    # THE GAP, found at BA14a while answering a different question: none of
    # the nine existing targets reads a scratch path -- so the reproduction
    # guarantee has no 60-day expiry -- but **none of them covers the
    # alignment predictor either.** `M` is the axis BOTH headline results are
    # compared through (AZ12, AZ13, design 3's primary), and until BA14b it
    # lived on a filesystem with a purge clock. A canary that is safe from the
    # purge is not the same as an artifact set that is, and only the second
    # question protects the headline. **A silent change to this file would
    # move the primary result and nothing would fail.**
    {"name": "alignment_predictor", "arm": "predictor",
     "path": "gtilde_compliant_b_ratio.h5",
     "hash_scope": "whole_file", "device_class": UNKNOWN,
     "why": "THE PREDICTOR OF RECORD -- M, the alignment axis both headlines "
            "are compared through (AZ12, AZ13, design 3's primary). Moved off "
            "scratch at BA14b; uncovered by the canary until BB section 4"},
    # ---- THE BG/BH BUILD's addition: THE POOLED NULL ---------------------
    # ADDITION TO COVERAGE, NOT A REBASELINE. Same distinction BB drew above.
    #
    # THE GAP: **no null artifact has ever been a canary target.** The nine
    # above are the MNN/reproducibility side. So when the BG/BH build ran the
    # first reproduction check ever applied to the pooled null, it found a
    # property of the pipeline nobody had observed -- the null is not
    # bit-reproducible run to run (CLAUDE.md gotcha 35).
    #
    # WHY BOTH CHANNELS, and the reason is NOT the obvious one. The dual
    # baseline was previously justified by "the reductions are maxima and a
    # maximum cannot detect changes beneath it". That is half the story and was
    # stated as the whole one. For NUMERICAL NOISE it inverts: the raw array is
    # float32 and quantises a 1e-15 perturbation away entirely, while the
    # reductions are float64 and preserve it -- measured, raw agreed on all but
    # 1-5 of 434,380,800 elements while the float64 reductions disagreed across
    # most of theirs. **Each channel is blind to what the other detects**: raw
    # catches structural change, the reductions catch last-bit change.
    # THE `why` BELOW WAS REWRITTEN AT BK, and the rewrite is the point.
    # BJ-C2 found that this target's artifact carries `superseded_for_axis` --
    # a `must_be_absent` key -- so `preflight.py` refuses it for every consumer,
    # while the pooled null the classification actually reads is uncovered. The
    # TARGET is not the defect; the INFERENCE a reader would draw from a green
    # verify on it is. **The canary answers "did this artifact move", not "is
    # this the artifact we report", and a target's reason must say which
    # question it answers.** Kept rather than re-pointed: removing a target
    # loses reproduction history, and the compliant artifact is ADDED beside it
    # on the `build_inputs` arm.
    {"name": "null_pooled_reductions", "arm": "null",
     "path": "null_pilot_k.h5",
     "hash_scope": "whole_file", "device_class": UNKNOWN,
     "why": "covered as a reproduction check on the null CODE PATH -- the "
            "artifact the run-to-run floor was measured on, carrying the "
            "float64 reductions (null_g_*, null_cmax_*, the schedules, "
            "fire_rate) that PRESERVE last-bit movement. REFUSED FOR CONSUMERS "
            "by the pre-flight gate (`superseded_for_axis`, 716 lib / 707 "
            "pool); NOT the artifact any reported result descends from. The "
            "pooled null the classification reads is null_pilot_compliant.h5, "
            "covered separately on the `build_inputs` arm (BJ-C2, BK)"},
    # The raw arrays themselves live on SCRATCH (23.44 GiB, purging), so they
    # are covered through their DURABLE DIGEST RECORD on project instead:
    # `null_baseline_a2.json` carries 4 whole-array sha256 plus 1,600
    # per-(seed, shift) digests and an independent summary channel. Hashing a
    # 23 GiB purging file every canary run would be neither durable nor cheap,
    # and BB's note that no target reads a scratch path -- so the guarantee has
    # no expiry -- is a property worth keeping.
    {"name": "null_pooled_raw_digests", "arm": "null",
     "path": "null_baseline_a2.json",
     "hash_scope": "whole_file", "device_class": UNKNOWN,
     "why": "THE POOLED NULL's raw channel, via its durable digests on "
            "project (the arrays are on purging scratch) -- the float32 "
            "channel, which QUANTISES last-bit movement away and therefore "
            "catches structural change the reductions cannot"},
    # ---- BK's addition: THE CLASSIFICATION BUILD'S SIX INPUTS -------------
    # ADDITION TO COVERAGE, NOT A REBASELINE. Same distinction BB and BG/BH
    # drew above, and the same machinery enforces it (`--assert-addition-only`).
    #
    # THE GAP, enumerated at BJ-C1 and re-confirmed here against every file's
    # OWN attrs: the classification build reads six artifacts and NOT ONE was a
    # canary target. A later statement that the build reproduced would not have
    # been distinguishable from a statement that its inputs moved.
    #
    # **EVERY TARGET BELOW CARRIES A FILE-INTEGRITY CLAIM, NOT A REPRODUCTION
    # CLAIM**, and `TOLERANCE["build_inputs"]` says so at length. Read that
    # before quoting any verdict on this arm.
    #
    # `build_device_class` is the class the target's NUMERIC CONTENT was
    # computed on -- the TopK gather -- which for #1 and #2 is NOT the device
    # their own producing job ran on, because both are downstream of an earlier
    # gather. The chain is named per target. Every value here is
    # INFERRED_FROM_PRODUCING_JOB_LOGS: none of these artifacts predates
    # `write_identity`'s `gather_device_class`, so none states its own, and
    # NOTHING IS BACK-FILLED into them.
    {"name": "ap_pooled_compliant", "arm": "build_inputs",
     "path": "feature_ap_main_compliant.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "rtx_5000_ada",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE ORPHAN SIDE -- pooled AP over the 703-concept pool, and the "
            "screen the ORPHAN bar is taken against. FILE INTEGRITY ONLY. "
            "Gather on rtx_5000_ada: job 20184677 (reslice, bit-identical "
            "column slice per its own VERIFY(1)) of feature_ap_main.h5 from "
            "job 20130645, whose log reads `ap_device=cpu (... the SAE forward "
            "pass still runs on cuda)` -- so the GATHER is GPU and the AP "
            "ARITHMETIC is CPU, and the gather is what device_class names"},
    {"name": "null_pooled_compliant", "arm": "build_inputs",
     "path": "null_pilot_compliant.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "rtx_5000_ada",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE POOLED NULL THE CLASSIFICATION ACTUALLY READS -- null_g_*, "
            "null_cmax_* (703) and fire_rate. This is the artifact "
            "null_pooled_reductions is NOT (BJ-C2). FILE INTEGRITY ONLY: the "
            "pooled null's measured run-to-run floor was obtained on "
            "null_pilot_k.h5 and is NOT transferred here. Gather on "
            "rtx_5000_ada: job 20142136 (`device=cuda`); the reductions were "
            "derived by job 20198126 on a CPU `day` node, which is the "
            "producing job and not where the numeric content came from"},
    {"name": "ap_experimental_cpu", "arm": "build_inputs",
     "path": "feature_ap_experimental_av_cpu.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE ALIGNED SIDE -- experimental-stratum AP, and `n_positives`, "
            "which is the thinness grid's own input (BI). FILE INTEGRITY ONLY. "
            "Gather on CPU: array 21712883, all 20 task logs `WARNING: no GPU "
            "visible`, merged by job 21712884. Its rtx_5000_ada twin "
            "feature_ap_experimental_av.h5 is a DIFFERENT artifact and is not "
            "covered here -- gotcha 36 is why this one is the live one"},
    {"name": "null_experimental_reductions", "arm": "build_inputs",
     "path": "null_experimental_bg.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE EXPERIMENTAL NULL's reductions -- null_cmax_*, "
            "surviving_positives_*, bh3_case, exceedance_resolution_*, "
            "n_positives_observed, on 218 pool concepts. FILE INTEGRITY ONLY, "
            "and the pooled null's CUDA-measured floor is NOT carried over: "
            "this ran on CPU, where whether index_add_ varies run to run is "
            "UNKNOWN and is not assumed either way. Gather on CPU: array "
            "21515225 (partition `day`, AllocTRES carries no gres/gpu), merged "
            "by job 21515226"},
    {"name": "null_experimental_raw", "arm": "build_inputs",
     "path": "null_gain_raw_experimental_bg.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE RAW EXPERIMENTAL NULL the Benjamini-Hochberg pass runs over "
            "-- gain_a / gain_b, (20, 20, 218, 10240), 8.0 GiB. FILE INTEGRITY "
            "ONLY. Same array and same merge as null_experimental_reductions, "
            "so same identity and same CPU gather; covered separately because "
            "the raw and the reductions are blind to different things (the "
            "float32 raw catches STRUCTURAL change, the float64 reductions "
            "catch LAST-BIT change -- the argument the `null` arm records)"},
    # ---- BS's addition: THE SEVENTH INPUT ---------------------------------
    # ADDITION TO COVERAGE, NOT A REBASELINE. No pre-existing target's fields
    # move, and `--assert-addition-only` is what makes that a measurement
    # rather than a claim.
    #
    # THE GAP, reported at BQ7c and left unacted twice (BQ, BR): the pooled RAW
    # null is an input to the classification build's ORPHAN side and no target
    # watched it. `null_pooled_raw_digests` above does NOT cover it -- that
    # target hangs off `null_pilot_k.h5`'s lineage, the SUPERSEDED axis, and
    # covers digests of `null_gain_raw_k.h5`. `null_pooled_compliant` holds the
    # reductions, whose `null_cmax_*` are MAXIMA and cannot produce per-pair
    # p-values. Neither is this array.
    #
    # RELOCATED TO PROJECT 2026-08-16 (amendment BT, part A3). This target
    # declared `"root": "scratch"` from BS until BT, because the artifact was
    # the only one on SCRATCH -- which purges at 30 days, putting a load-bearing
    # classification input on a deletion clock. BT copied it to the analysis
    # root and re-pointed this spec; the SCRATCH copy is left to purge on its
    # own schedule rather than deleted. The `root` key is GONE, so the target
    # now inherits `analysis` like every other one, and `sha256` is unchanged
    # across the move -- `compare_manifests` classifies that as `relocated`,
    # which is a different thing from a rebaseline and is asserted as such.
    #
    # `resolve`, ROOT_NAMES and --scratch-root are RETAINED even though no
    # target names a non-default root today: they are the mechanism a future
    # target on another filesystem needs, they are directly tested, and
    # removing them would delete a working guard to tidy up.
    {"name": "null_pooled_raw_compliant", "arm": "build_inputs",
     "path": "null_gain_raw_compliant.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "rtx_5000_ada",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE SEVENTH INPUT -- the POOLED raw null the orphan side reads, "
            "gain_a / gain_b (20, 20, 703, 10240) float32, 21.48 GiB. "
            "FILE INTEGRITY ONLY, AND THE CLAIM IS NARROWER THAN THE OTHER "
            "null TARGETS': this says the artifact has NOT CHANGED ON DISK, "
            "not that the code reproduces it. NO CONTROL RUN OF THIS ARTIFACT "
            "EXISTS, so NO run-to-run floor has been measured for it and none "
            "is assumed -- the `null` arm's measured floor was obtained on "
            "null_pilot_k.h5 and is NOT transferred here (gotcha 35's rule "
            "that a tolerance must be measured from a control, never argued). "
            "Gather on rtx_5000_ada: job 20142136 (`sae_null_pilot`, log line "
            "10 reads `device=cuda`) wrote null_gain_raw_k.h5; job 20184677 "
            "(`sae_reslice_axis`) resliced THIS file from it and its own "
            "VERIFY(1) reports the slice bit-exact against source -- a reslice "
            "performs no gather, so the gather device is the source job's. "
            "ITS SEED AXIS CARRIES NO LABELS (no `seeds` dataset) and the "
            "order is RECOVERED FROM THE REDUCTIONS' DATA, not read; the "
            "sidecar seed_order_pooled_raw_null.json records the mapping and "
            "the method. RELOCATED TO PROJECT 2026-08-16 (BT-A3): it was on "
            "SCRATCH, which purges at 30 days, and at mtime 2026-07-29 it "
            "would have gone ~2026-08-28. The scratch copy is NOT deleted, it "
            "is left to purge on its own schedule, so from ~2026-08-28 there "
            "is one copy and it is this one. The artifact's own `note` attr "
            "still says `60-day purge`, which is SUPERSEDED and is NOT edited "
            "(report-do-not-fix) -- getquota reads 30 days",
     },
    {"name": "label_store_f67485d", "arm": "build_inputs",
     "path": "residue_labels_subtyped_untruncated_pgate5_f67485d.h5",
     "hash_scope": "whole_file", "device_class": UNKNOWN,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE LABEL STORE every identity key above is asserted against -- "
            "712 concepts, 703 argmax pool, and the positive sets themselves. "
            "UPSTREAM OF ALL FIVE OTHERS, so a silent change here would move "
            "every one of them. FILE INTEGRITY ONLY. device_class is "
            "NOT_ESTABLISHED and that is precise, not a shrug: there is no "
            "TopK gather and no MNN threshold anywhere in its production path, "
            "so no device-sensitivity measurement exists for it -- which is "
            "still not a claim of device-independence. Built by job 20180679 "
            "(partition `day`, cpu=4, AllocTRES carries no gres/gpu)"},
    # ---- BM's addition: THE INFERENTIAL PRIMARY'S OWN ARTIFACT ------------
    # ADDITION TO COVERAGE, NOT A REBASELINE. Same distinction BB and the
    # BG/BH build drew above, and no existing target's fields move.
    #
    # THE GAP, reported at BL-E6 while answering a different question and not
    # acted on there: **`design3_join_ar.json` is not a canary target.** It is
    # the artifact carrying design 3's primary -- `b1_z`, the theta curve, the
    # per-seed fits, the variance shares and the meta-analysis -- and every
    # headline statement about alignment is read out of it. Eighteen targets
    # watched the inputs and the reproducibility side; none watched this.
    # A silent change here would move the inferential primary and nothing
    # would fail.
    {"name": "design3_join_ar", "arm": "inferential_primary",
     "path": "design3_join_ar.json",
     "hash_scope": "whole_file", "device_class": UNKNOWN,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE INFERENTIAL PRIMARY'S ARTIFACT -- b1_z at the reference "
            "theta, the 13-point theta curve, the 20 per-seed fits and their "
            "variance shares, and the meta-analysis every alignment statement "
            "is quoted from. FILE INTEGRITY ONLY, and the arm's tolerance says "
            "why that is the weaker claim. device_class is NOT_ESTABLISHED and "
            "that is precise rather than a shrug: design3_join.py READS "
            "rf_counts and computes no cosine and no TopK gather itself, so no "
            "device-sensitivity measurement exists for its own path -- but it "
            "INHERITS MNN threshold sensitivity through rf_counts, which is an "
            "inherited exposure and not an absent one. Built by job 20813625 "
            "(partition `day`, cpu=8, AllocTRES carries no gres/gpu)"},

    # ---- BY's addition: THE BUILD OUTPUTS THE PAPER'S NUMBERS COME FROM -----
    # ADDITION TO COVERAGE, NOT A REBASELINE. No existing target's fields move,
    # and this is a NEW ARM rather than a widening of `build_inputs` or
    # `inferential_primary`: both of those tolerances reason about a FIXED SET
    # ("these six", "this one artifact"), so extending either would silently
    # change what a committed tolerance was written about.
    #
    # THE GAP, established by enumerating the twenty committed targets rather
    # than by assuming it: every one of them is an INPUT, a NULL, a PREDICTOR, a
    # LABEL STORE or a MARGINAL. **Not one is a build OUTPUT.** The inputs were
    # brought under coverage at BB/BG/BH and the inferential primary at BM; the
    # artifacts the paper's concept-level, classification, grid, paired-arm and
    # decomposition numbers are actually READ FROM were watched by nothing. That
    # is the same coverage problem one level downstream of where it was fixed.
    #
    # THE CLAIM IS FILE INTEGRITY ONLY, ON EVERY TARGET IN THIS ARM, and the
    # arm's tolerance says why. Where INDEPENDENT-RECOMPUTATION evidence exists
    # it is recorded in that target's own `why` WITH ITS SCOPE -- because the
    # evidence covers a dataset, not a file, and a sha256 over the file cannot
    # be described as verifying it (gotcha 33's shape: the claim must match what
    # the instrument does).
    {"name": "concept_repro_ax", "arm": "build_outputs",
     "path": "concept_repro_ax_v2.json",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE CONCEPT-LEVEL DESCRIPTIVE HEADLINE'S ARTIFACT -- D(c) for all "
            "703 concepts, the AW1.5 percentile curve, mean_A/sd_A, and the "
            "per-concept block every concept-level statement is quoted from, "
            "including the never-detected set (D == 0). FILE INTEGRITY ONLY: no "
            "control run of this artifact has ever been made, so no floor has "
            "been MEASURED for it and a numeric criterion would have nothing to "
            "rest on (gotcha 35). device_class is TOPK_GATHER because the AP it "
            "reduces descends from a TopK gather, which is device-sensitive "
            "between classes (gotcha 36) -- an INHERITED exposure, not an "
            "absent one"},
    {"name": "classification_bh_experimental_bo", "arm": "build_outputs",
     "path": "classification_bh_experimental_bo.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "BUILD COMPONENT 1 -- the multiple-testing pass over the raw "
            "experimental null. FILE INTEGRITY ONLY. **INDEPENDENT-"
            "RECOMPUTATION EVIDENCE EXISTS AND IS SCOPED**: BT-C2 recomputed "
            "the BH `reject` arrays and found them bit-identical in 40 of 40 "
            "(seed, arm) cells. That covers the `reject` DATASET, not the "
            "file's bytes -- so it supports a reproduction claim about the BH "
            "computation and NOT about this target, whose verdict is still a "
            "sha256 over the whole file"},
    {"name": "classification_2x2_bq", "arm": "build_outputs",
     "path": "classification_2x2_bq.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "BUILD COMPONENTS 2 AND 3 -- pooled exceedance and the 2x2 "
            "classification at threshold 1, the unrestricted primary. Carries "
            "the per-(seed, feature) cell assignment, both arms' p-values and "
            "the recovered seed positions. TWO components share one artifact, "
            "which is why an enumeration by COMPONENT INDEX and one by FILE do "
            "not agree in count. FILE INTEGRITY ONLY; no control run exists"},
    {"name": "grid_composition_bt", "arm": "build_outputs",
     "path": "grid_composition_bt.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "BUILD COMPONENT 4 -- the remaining four grid points as "
            "classification composition, the thinness sensitivity grid. FILE "
            "INTEGRITY ONLY; no control run of this artifact exists"},
    {"name": "matched_pair_arm_bu", "arm": "build_outputs",
     "path": "matched_pair_arm_bu.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "BUILD COMPONENT 5 -- the firing-rate-matched paired arm at "
            "threshold 1, the pre-registered robustness arm of the governing "
            "constraint. FILE INTEGRITY ONLY; no control run exists"},
    {"name": "br_floor_concentration", "arm": "build_outputs",
     "path": "br_floor_concentration.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE RESOLUTION-FLOOR CHARACTERISATION (BR) -- per-(seed, arm, "
            "concept) at-floor feature counts on both arms, which is what any "
            "statement about what the BH route COULD have detected rests on "
            "(gotcha 42). Also an input to BY's never-detected join. FILE "
            "INTEGRITY ONLY; no control run exists"},
    {"name": "protein_vs_residue_bw", "arm": "build_outputs",
     "path": "protein_vs_residue_bw.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE PROTEIN-VERSUS-RESIDUE DECOMPOSITION (BW) -- both detection "
            "matrices in full, both thresholds, best_gain and the per-concept "
            "cell. FILE INTEGRITY ONLY. **INDEPENDENT-RECOMPUTATION EVIDENCE "
            "EXISTS AND IS SCOPED**: BY's percentile curve recomputed this "
            "artifact's 95th-percentile row from the same stored draws and "
            "found both thresholds and both D vectors bit-identical (worst "
            "|delta| exactly 0). That covers those four DATASETS, not the "
            "file's bytes"},
    {"name": "protein_vs_residue_feature_view_bw", "arm": "build_outputs",
     "path": "protein_vs_residue_feature_view_bw.json",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE DECOMPOSITION'S FEATURE-UNIT VIEW (BW) -- the cell-level "
            "figures, which are on a DIFFERENT BASIS from the concept-unit ones "
            "and are not comparable to them (BX). FILE INTEGRITY ONLY; no "
            "control run exists"},
    {"name": "decomposition_reporting_bx", "arm": "build_outputs",
     "path": "decomposition_reporting_bx.json",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "INFERRED_FROM_PRODUCING_JOB_LOGS",
     "why": "THE DECOMPOSITION'S REPORTING BASIS (BX) -- the full grid over "
            "every detection rate a 20-anchor rate can take, each figure with "
            "its own denominator and criterion, the unit/population bridge, and "
            "the common-basis comparison. This is the artifact that makes the "
            "circulating figures quotable at all. FILE INTEGRITY ONLY; no "
            "control run exists"},
    {"name": "decomposition_percentile_curve_by", "arm": "build_outputs",
     "path": "decomposition_percentile_curve_by.json",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "MEASURED",
     "why": "THE AW1.5 PERCENTILE CURVE for the decomposition's within-protein "
            "arm (BY) -- the five estimable percentiles, both arms at each, and "
            "the range verdict on both bases. FILE INTEGRITY ONLY. "
            "build_device_provenance is MEASURED rather than inferred: this "
            "session ran the producing job itself and read the device off it, "
            "rather than reconstructing it from logs"},
    {"name": "never_detected_join_by", "arm": "build_outputs",
     "path": "never_detected_join_by.h5",
     "hash_scope": "whole_file", "device_class": TOPK_GATHER,
     "build_device_class": "cpu",
     "build_device_provenance": "MEASURED",
     "why": "THE NEVER-DETECTED DESCRIPTIVE JOIN (BY) -- annotation density, "
            "experimental support, experimental-stratum membership and "
            "at-floor supply for the 43 concepts no anchor seed detected. "
            "DESCRIPTIVE and NOT reportable as a finding, which its own attrs "
            "carry; covered here because a paper sentence about the "
            "never-detected set will be read out of it either way. FILE "
            "INTEGRITY ONLY. build_device_provenance is MEASURED: this session "
            "ran the producing job"},
    # ---- CC's addition: THE DETERMINISM PROBE'S REDUCED ARTIFACTS ---------
    # ADDED AS COVERAGE, NOT A REBASELINE OF ANYTHING EXISTING.
    #
    # WHY THESE EIGHT, AND ON A CORRECTED GROUND (amendment CC-C3). The brief
    # proposing them said they become "the SOLE surviving evidence for three
    # committed figures" once `$SCRATCH/determinism_probe/` purges (~2026-09-04).
    # THAT IS OVERSTATED AND IS NOT THE GROUND USED HERE: the three figures are
    # committed IN GIT in four places -- this module's own `null` tolerance
    # string below, CLAUDE.md gotcha 35, docs/GOTCHAS.md #35, and
    # docs/PROJECT.md's running narrative preamble (NOT its §5: that pointer
    # came from §CB-d2's table and is wrong -- corrections entry 183)
    # -- and git is not on scratch. Further, `getquota`
    # reports `Backup: Yes` for the project fileset and `Backup: No` for
    # scratch, so these eight already have a second copy.
    #
    # WHAT IS TRUE, AND IT IS THE NARROWER AND BETTER CLAIM: after the scratch
    # purges these become the sole surviving DERIVATION of those figures -- the
    # only place the 446,308/614,400 decomposition behind "up to 72%", the
    # per-array 1+4 split behind "~5 of 434,380,800", `git_sha 002673b` and both
    # source paths can be read. Git holds the numbers; these hold where they
    # came from. Under gotcha 35's own rule that a tolerance must be MEASURED
    # from a control and never argued, a criterion whose control cannot be
    # produced is indistinguishable from one loosened after it failed -- so the
    # derivation is load-bearing, not decorative.
    #
    # AND THE RE-RUN THAT WOULD REGENERATE THEM EXPIRES FIRST:
    # slurm/null_determinism_probe.sbatch:77 passes --eval-cache, and
    # cache/eval.h5 purges ~2026-08-23, TWELVE DAYS BEFORE the scratch files.
    #
    # `device_class` IS TOPK_GATHER BECAUSE THE CONTENT DESCENDS FROM THE
    # GATHER, AND THAT IS NOT THE MECHANISM THESE DOCUMENTS MEASURE. They
    # quantify RUN-TO-RUN variation on the SAME device class, whose measured
    # cause is `index_add_` CUDA atomics (gotcha 35). Gotcha 36's BETWEEN-class
    # TopK near-tie sensitivity -- what this field names -- has NOT been
    # measured on these artifacts and is not claimed. Two different gotchas,
    # stated apart so a reader does not read one as the other.
    #
    # `build_device_class` IS DELIBERATELY UNSTATED, so these INHERIT the
    # documented BASELINE_DEVICE_CLASS default. The producing job requests
    # `--gpus=rtx_5000_ada:1` (line 4 of its sbatch), which is exactly the basis
    # the default records -- a REQUEST, project-wide. It is not read off the
    # artifact and no job log was opened, so neither MEASURED nor
    # INFERRED_FROM_PRODUCING_JOB_LOGS would be true, and stating a class with a
    # provenance flag that overstates it is the defect the pairing rule exists
    # to prevent (gotcha 15).
    *(
        {"name": f"determ_{cmp}_{kind}", "arm": "determinism",
         "path": f"determ_{cmp}_{kind}.json",
         "hash_scope": "whole_file", "device_class": TOPK_GATHER,
         "why": (
             f"THE DETERMINISM PROBE'S {'REDUCTIONS' if kind == 'red' else 'RAW-ARRAY'} "
             f"COMPARISON for {cmp} -- one of the eight difference summaries "
             "null_determinism_probe.sbatch:122-135 writes to $PROJECT, 61,006 "
             "bytes for the eight, durable since 2026-08-05 and subject to no "
             "purge. FILE INTEGRITY ONLY: this says the artifact has NOT "
             "CHANGED ON DISK, not that the code reproduces it -- and the code "
             "CANNOT, because the probe consumes cache/eval.h5, which purges "
             "~2026-08-23. THE CONTROL COMPARISON IS determ_AB_*: the same code "
             "run twice on the same device class, from which gotcha 35's three "
             "committed tolerance figures reproduce exactly. NO run-to-run "
             "floor is measured for the JSON itself and none is assumed; it is "
             "a text artifact nothing writes to anymore, so a DIFFERS verdict "
             "means something overwrote it and a MISSING_NOW verdict -- the "
             "live risk, with project at 92% of quota -- means it was deleted. "
             "Added as COVERAGE, not rebaselined: `first_baselined_at` is the "
             "commit that added it, and 'has reproduced since' cannot be "
             "back-dated to a period when nothing was watching")}
        for cmp in ("AB", "AC", "A_vs_ref", "C_vs_ref")
        for kind in ("red", "raw")
    ),
)

TOLERANCE = {
    "n19": "BIT_IDENTICAL -- the sensitivity arm of record (AD2)",
    "n39": "BIT_IDENTICAL FROM THE FIRST BASELINE FORWARD -- the primary arm "
           "(AM5). No earlier baseline exists; it has not reproduced across a "
           "refactor it never saw, and must not be described as having done so.",
    "predictor": "BIT_IDENTICAL FROM THE FIRST BASELINE FORWARD -- the "
                 "alignment predictor (BB section 4). Added as COVERAGE, not "
                 "rebaselined: `first_baselined_at` is the commit that added "
                 "it, so 'has reproduced since' cannot be back-dated to a "
                 "period when nothing was watching.",
    # THE TOLERANCE IS MEASURED, NOT CHOSEN, and the distinction is the whole
    # point of the entry.
    "null": "BIT_IDENTICAL FOR THE STORED BYTES; a REGENERATION is held to a "
            "MEASURED FLOOR instead, and the two are different claims. "
            "The file on disk does not change by itself, so a sha256 that "
            "moves means something wrote it -- that is what this target "
            "watches, and bit-identity is the right criterion for it. But the "
            "pooled null CANNOT be regenerated bit-for-bit, by any version of "
            "the code including the one that wrote the baseline. "
            "THE FLOOR, OBTAINED BY RUNNING UNCHANGED CODE TWICE and not by "
            "any judgement about what is acceptable: 7.105e-15 maximum "
            "absolute difference in the float64 reductions (differing in up "
            "to 72% of elements), and ~5 differing elements of 434,380,800 "
            "per array in the float32 raw. Cause: `index_add_` on CUDA "
            "accumulates atomically and `use_deterministic_algorithms` is "
            "enabled nowhere in this repo. "
            "SO: a DIFFERS verdict here means the stored artifact was "
            "overwritten. It does NOT license comparing a re-run against it "
            "by bit-identity -- use the floor above, and note that a "
            "criterion loosened after it failed is indistinguishable from one "
            "corrected after being shown ill-posed unless the control is "
            "actually run (CLAUDE.md gotcha 35). "
            "Added as COVERAGE, not rebaselined: `first_baselined_at` is the "
            "commit that added it.",
    # AMENDMENT CC. THE ARM WHOSE PRODUCER CANNOT BE RE-RUN AT ALL, WHICH IS
    # WHY IT IS A SEPARATE ARM AND NOT FOLDED INTO `build_inputs`.
    "determinism":
        "FILE INTEGRITY ONLY -- A WEAKER CLAIM THAN EVERY REPRODUCTION ARM IN "
        "THIS MANIFEST, AND NOT A REPRODUCTION GUARANTEE. "
        "What a REPRODUCES verdict here means: the artifact's bytes on disk are "
        "the bytes recorded at `first_baselined_at`. What it does NOT mean: "
        "that the code still produces them. "
        "AND HERE THAT SECOND CLAUSE IS NOT A CAUTION BUT A FACT: the producing "
        "job, slurm/null_determinism_probe.sbatch, passes --eval-cache at line "
        "77, and cache/eval.h5 purges ~2026-08-23. After that date these eight "
        "documents cannot be regenerated by any version of the code, on any "
        "device, until the activation cache is rebuilt. NO NUMERIC TOLERANCE IS "
        "STATED AND NONE COULD BE MEASURED: a control-measured floor needs a "
        "producing re-run, which is exactly what is unavailable. "
        "THE POOLED NULL'S MEASURED FLOOR IS NOT TRANSFERRED HERE. It was "
        "measured on null_pilot_k.h5's float64 reductions and float32 raw "
        "arrays; these are JSON difference summaries, a different artifact in a "
        "different format, and same code family is not same artifact (CLAUDE.md "
        "gotcha 35). "
        "WHAT THESE EIGHT UNIQUELY HOLD, and the reason the arm exists: the "
        "three committed tolerance figures are in git in four places, so this "
        "arm does NOT protect the figures -- it protects their DERIVATION. The "
        "446,308/614,400 decomposition behind 'up to 72%', the per-array 1+4 "
        "behind '~5 of 434,380,800', git_sha 002673b and both source paths live "
        "nowhere else once $SCRATCH/determinism_probe/ purges ~2026-09-04. "
        "SO: a DIFFERS verdict means the stored artifact was overwritten; a "
        "MISSING_NOW verdict means it was deleted, which is the live risk while "
        "project sits at 92% of quota. Neither licenses any comparison of a "
        "re-run against this baseline, because there can be no re-run. "
        "Added as COVERAGE, not rebaselined: `first_baselined_at` is the commit "
        "that added it, and 'has reproduced since' cannot be back-dated to a "
        "period when nothing was watching.",
    # THE ONLY ARM WHOSE CLAIM IS NOT REPRODUCTION. Stated first and stated
    # plainly, because a green verdict on this arm reads like the others.
    "build_inputs":
        "FILE INTEGRITY ONLY -- THIS IS A WEAKER CLAIM THAN EVERY OTHER ARM "
        "IN THIS MANIFEST, AND IT IS NOT A REPRODUCTION GUARANTEE. "
        "What a REPRODUCES verdict here means: the artifact's bytes on disk are "
        "the bytes recorded at `first_baselined_at`. What it does NOT mean: "
        "that the code still produces it. It catches a silent overwrite, a "
        "partial write, and an unrecorded regeneration -- the three things that "
        "actually threaten these six between now and the classification build, "
        "and a prior session did overwrite an artifact in place rather than "
        "writing to a new path. It catches CODE DRIFT NOT AT ALL. "
        "WHY NOT A NUMERIC TOLERANCE: five of the six cannot be re-derived "
        "bit-for-bit, and a control-measured floor for them needs a producing "
        "re-run this session was not authorised to launch (the experimental "
        "null alone cost 20 CPU tasks at ~12.5 h each). "
        "THE POOLED NULL'S MEASURED FLOOR IS NOT TRANSFERRED TO ANY TARGET ON "
        "THIS ARM. It was measured on null_pilot_k.h5 via a CUDA path whose "
        "index_add_ atomics are the stated mechanism; four of these six were "
        "produced on CPU, where whether index_add_ varies run to run is "
        "UNKNOWN and is NOT assumed in either direction. Same code family is "
        "not same artifact (CLAUDE.md gotcha 35), and 'the CUDA argument does "
        "not apply on CPU' would be a mechanism claim standing in for a "
        "measurement (gotcha 22). "
        "SO: a DIFFERS verdict here means the stored artifact was overwritten "
        "and the build's input is not what it was. It licenses NO comparison "
        "of a re-run against this baseline by bit-identity. "
        "PRECEDENT: the pre-migration git bundle is verified by file-integrity "
        "against a recorded hash rather than by a rebuild, for the same reason. "
        "Added as COVERAGE, not rebaselined: `first_baselined_at` is the "
        "commit that added it, and 'has reproduced since' cannot be back-dated "
        "to a period when nothing was watching.",
    # THE SECOND ARM WHOSE CLAIM IS NOT REPRODUCTION. Stated as plainly as
    # `build_inputs` above, and for the same reason: a green verdict here reads
    # exactly like a green verdict on `n39`, and it means much less.
    "inferential_primary":
        "FILE INTEGRITY ONLY -- A WEAKER CLAIM THAN EVERY REPRODUCTION ARM IN "
        "THIS MANIFEST, AND NOT A REPRODUCTION GUARANTEE. "
        "What a REPRODUCES verdict here means: the artifact's bytes on disk are "
        "the bytes recorded at `first_baselined_at`. What it does NOT mean: "
        "that the code still produces it, or that re-running the join would "
        "return these numbers. It catches a silent overwrite, a partial write "
        "and an unrecorded regeneration. It catches CODE DRIFT NOT AT ALL. "
        "WHY NOT A NUMERIC TOLERANCE: no control run of this artifact has ever "
        "been made, so no floor has been MEASURED for it, and a floor is the "
        "only thing that would license a numeric criterion (gotcha 35). "
        "Re-running the join to measure one is a real job (20813625: 7 m 22 s, "
        "96 GiB, 8 cpu) and BM was not authorised to launch it. "
        "THE POOLED NULL'S MEASURED FLOOR IS NOT TRANSFERRED HERE, on exactly "
        "the grounds `build_inputs` states: it was measured on a different "
        "artifact via a CUDA index_add_ path, this one was produced on CPU, and "
        "'the CUDA argument does not apply on CPU' would be a mechanism claim "
        "standing in for a measurement (gotcha 22). "
        "SO: a DIFFERS verdict here means the stored artifact was overwritten "
        "and the inferential primary is not what it was. It licenses NO "
        "comparison of a re-run against this baseline by bit-identity. "
        "Added as COVERAGE, not rebaselined: `first_baselined_at` is the commit "
        "that added it, and 'has reproduced since' cannot be back-dated to a "
        "period when nothing was watching.",
    "build_outputs":
        "FILE INTEGRITY ONLY -- A WEAKER CLAIM THAN EVERY REPRODUCTION ARM IN "
        "THIS MANIFEST, AND NOT A REPRODUCTION GUARANTEE. What a REPRODUCES "
        "verdict here means: each artifact's bytes on disk are the bytes "
        "recorded at its `first_baselined_at`. What it does NOT mean: that the "
        "code still produces them, or that re-running any of these builds would "
        "return these numbers. It catches a silent overwrite, a partial write "
        "and an unrecorded regeneration. It catches CODE DRIFT NOT AT ALL. "
        "WHY THIS ARM EXISTS: the other five arms watch INPUTS, NULLS, the "
        "PREDICTOR, the LABEL STORE and the INFERENTIAL PRIMARY. Established by "
        "enumerating the twenty committed targets rather than assumed, NOT ONE "
        "was a build OUTPUT -- so the artifacts the paper's concept-level, "
        "classification, grid, paired-arm and decomposition numbers are read "
        "from were watched by nothing (amendment BY Part A). "
        "WHY A NEW ARM RATHER THAN WIDENING ONE: `build_inputs`' tolerance "
        "reasons about a fixed set of six and `inferential_primary`'s about a "
        "single artifact, so extending either would silently change what a "
        "committed tolerance was written about -- the same reason BM created an "
        "arm instead of widening BB's. "
        "WHY NOT A NUMERIC TOLERANCE: no control run exists for nine of the "
        "eleven, so no floor has been MEASURED for them, and a measured floor "
        "is the only thing that would license a numeric criterion (gotcha 35). "
        "THE POOLED NULL'S MEASURED FLOOR IS NOT TRANSFERRED TO ANY TARGET ON "
        "THIS ARM, on exactly the grounds `build_inputs` states: it was "
        "measured on a different artifact via a CUDA `index_add_` path, these "
        "were produced on CPU, and 'the CUDA argument does not apply on CPU' "
        "would be a mechanism claim standing in for a measurement (gotcha 22). "
        "WHERE INDEPENDENT-RECOMPUTATION EVIDENCE DOES EXIST IT IS RECORDED IN "
        "THE TARGET'S OWN `why` FIELD WITH ITS SCOPE, and it is scoped to a "
        "DATASET rather than a file in both cases -- BT-C2's 40/40 bit-identical "
        "BH `reject`, and BY's bit-identical recomputation of the "
        "decomposition's 95th-percentile thresholds and D vectors. A sha256 over "
        "the whole file does not verify either, and must not be described as "
        "doing so. "
        "SO: a DIFFERS verdict here means the stored artifact was overwritten "
        "and a number the paper reads is not what it was. It licenses NO "
        "comparison of a re-run against this baseline by bit-identity. "
        "Added as COVERAGE, not rebaselined: `first_baselined_at` is the commit "
        "that added it, and 'has reproduced since' cannot be back-dated to a "
        "period when nothing was watching.",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_leaves(obj) -> int:
    if isinstance(obj, dict):
        return sum(_json_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_json_leaves(v) for v in obj)
    return 1


def describe(p: Path) -> dict:
    """Hash plus a structural fingerprint. Missing files are REPORTED, not
    skipped -- a manifest that silently omits an absent target is a check that
    passes on nothing."""
    if not p.exists():
        return {"status": "ABSENT", "path": str(p)}
    out = {"status": "PRESENT", "path": str(p), "bytes": p.stat().st_size,
           "sha256": sha256_file(p)}
    if p.suffix == ".json":
        doc = json.loads(p.read_text())
        out["n_leaves"] = _json_leaves(doc)
        out["top_level_keys"] = sorted(doc) if isinstance(doc, dict) else None
    elif p.suffix == ".h5":
        import h5py
        with h5py.File(p, "r") as hf:
            # THE WALK IS RECURSIVE (amendment BK), AND IT HAD TO BECOME SO.
            # This read `{k: list(hf[k].shape) for k in hf}`, which raises
            # `AttributeError: 'Group' object has no attribute 'shape'` on any
            # GROUPED file -- so `describe` could not have described three of
            # BK's six additions at all: the two `feature_ap*` artifacts nest
            # under `/pooled` and `/experimental`, and the label store under
            # `/concepts`, `/instances`, `/protein_sets`.
            #
            # IT RE-DESCRIBES NOTHING ALREADY BASELINED. Measured on disk
            # before the change, not argued: all four h5 targets predating BK
            # (`rf_counts_theta070_n19`, `rf_counts_theta070_n39`,
            # `gtilde_compliant_b_ratio`, `null_pilot_k`) are FLAT, and the
            # recursive walk returns a dict equal to the flat one for every
            # one of them. `datasets` is descriptive in any case -- it is not
            # in BASELINE_FIELDS, so `compare_manifests` never reads it.
            datasets: dict[str, list[int]] = {}
            hf.visititems(
                lambda name, obj: datasets.__setitem__(name, list(obj.shape))
                if isinstance(obj, h5py.Dataset) else None)
            # NO `n_datasets` COMPANION KEY: it is derivable from `datasets`,
            # and adding it would make every PRE-EXISTING target's manifest row
            # gain a field on the capture that adds BK's six. That is a
            # re-description of targets this amendment authorises only an
            # addition beside.
            out["datasets"] = datasets
            out["n_attrs"] = len(hf.attrs)
            for k in ("n_compare_seeds", "n_anchors", "theta"):
                if k in hf.attrs:
                    v = hf.attrs[k]
                    out[k] = v.item() if hasattr(v, "item") else v
    return out


def head_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=repo).decode().strip()


#: The roots a target may name. `analysis` is `--root` and is the default for
#: every target that does not say otherwise. `scratch` exists because BS's
#: seventh input is the one artifact NOT on project (BS-C2), and naming it is
#: what keeps `--root`'s indirection intact for the test harness.
ROOT_NAMES = ("analysis", "scratch")


def resolve(roots: dict[str, Path], spec: dict) -> Path:
    """Where a target's file lives, given the roots in play.

    A spec naming no root gets `analysis`, so no pre-existing target changes.
    A spec naming a root that was not supplied falls back to `analysis` -- and
    the file will then be reported ABSENT rather than silently matching
    something else, which is the failure direction this module already
    prefers (`describe` reports missing files, it does not skip them).
    """
    name = spec.get("root", "analysis")
    if name not in ROOT_NAMES:
        raise ValueError(
            f"target {spec['name']!r} names root {name!r}, which is not one of "
            f"{ROOT_NAMES}; a typo here would resolve to a path nobody checks")
    return roots.get(name, roots["analysis"]) / spec["path"]


def _roots(root: Path, scratch_root: Path | None) -> dict[str, Path]:
    return {"analysis": root, "scratch": scratch_root or root}


def capture(root: Path, repo: Path, previous: dict | None,
            scratch_root: Path | None = None) -> dict:
    """Build a manifest. `first_baselined_at` is CARRIED FORWARD from a previous
    manifest where one exists, so a re-capture cannot silently reset the date a
    target's expectation starts from."""
    sha = head_sha(repo)
    roots = _roots(root, scratch_root)
    prev_first = {t["name"]: t.get("first_baselined_at")
                  for t in (previous or {}).get("targets", [])}
    targets = []
    for spec in TARGETS:
        d = describe(resolve(roots, spec))
        targets.append({**spec, **d,
                        "first_baselined_at": prev_first.get(spec["name"]) or sha,
                        "captured_at": sha})
    return {
        "identity": {"measurement": "canary_manifest",
                     "amendments": ["AD1", "AD2", "AM5", "BB4", "BC7", "BK",
                                    "BM"],
                     "schema_version": MANIFEST_SCHEMA_VERSION,
                     "captured_at_git_sha": sha, "root": str(root),
                     "tolerance": TOLERANCE,
                     # BC7: the condition every BIT_IDENTICAL tolerance above
                     # silently carried until it was written down.
                     "device_class_condition": DEVICE_CLASS_CONDITION,
                     # BK: RETAINED, and now explicitly the DEFAULT for a
                     # target stating no `build_device_class` of its own --
                     # never a per-target measurement. Read the per-target
                     # field first; `build_device_coverage` says which targets
                     # have one.
                     "baseline_device_class": BASELINE_DEVICE_CLASS,
                     "build_device_coverage": build_device_coverage()},
        "targets": targets,
    }


# The fields that ARE the baseline. `captured_at` is deliberately absent: it
# records when the manifest was last written and moves on every capture by
# construction, so comparing it would build a check that can only fail
# (gotcha 25a).
BASELINE_FIELDS = ("arm", "path", "hash_scope", "sha256", "n_leaves",
                   "first_baselined_at", "bytes", "status")

# BASELINE_FIELDS minus `path` -- the fields that describe WHAT the artifact IS
# rather than WHERE it is. A relocation is defined as `path` moving while every
# one of these holds (amendment BT, part A3). DERIVED from BASELINE_FIELDS so a
# field added there cannot be silently omitted from the relocation conjunction.
CONTENT_FIELDS = tuple(f for f in BASELINE_FIELDS if f != "path")


def tolerance_coverage(targets: tuple[dict, ...] | None = None,
                       tolerance: dict | None = None) -> dict:
    """Every arm named by a target must have a stated tolerance.

    A target on an arm with no TOLERANCE entry would be watched with no
    declared expectation -- which reads as coverage and is not. DERIVED, so
    adding a target on a new arm and forgetting its tolerance fails a test
    rather than shipping.

    **`targets` and `tolerance` ARE INJECTABLE FOR ONE REASON: the real
    TARGETS are fully covered, so a test that could only call this with them
    can never exhibit the condition it checks** -- and a mutation removing the
    `missing` branch survived exactly that way on the first matrix run
    (gotcha 16's fixture lesson). The defaults are the module's own, so no
    caller changes.
    """
    targets = TARGETS if targets is None else targets
    tolerance = TOLERANCE if tolerance is None else tolerance
    arms = sorted({t["arm"] for t in targets})
    missing = [a for a in arms if a not in tolerance]
    unused = [a for a in tolerance if a not in arms]
    return {"arms_in_targets": arms, "arms_with_tolerance": sorted(tolerance),
            "arms_without_tolerance": missing, "tolerance_without_targets": unused,
            "verdict": "COVERED" if not missing else "UNCOVERED_ARM"}


def device_class_coverage(targets: tuple[dict, ...] | None = None,
                          vocabulary: tuple[str, ...] | None = None) -> dict:
    """Every target must carry a `device_class` from the declared vocabulary.

    Amendment BC7. A target with no `device_class`, or with a value nobody
    declared, is one whose bit-identity expectation carries an UNSTATED
    condition -- which is the state the whole of BC7 exists to leave behind.
    DERIVED, so adding a target and forgetting the field fails a test rather
    than shipping.

    **`targets` and `vocabulary` ARE INJECTABLE FOR THE SAME REASON
    `tolerance_coverage`'s ARE**: the real TARGETS are fully annotated, so a
    test that could only call this with them can never exhibit either failing
    branch (gotcha 16, and BB12 item 2's lesson applied before rather than
    after). The defaults are the module's own, so no caller changes.
    """
    targets = TARGETS if targets is None else targets
    vocabulary = DEVICE_CLASSES if vocabulary is None else vocabulary
    missing = [t["name"] for t in targets if not t.get("device_class")]
    undeclared = sorted({t["device_class"] for t in targets
                         if t.get("device_class")
                         and t["device_class"] not in vocabulary})
    by_class: dict[str, list[str]] = {}
    for t in targets:
        by_class.setdefault(t.get("device_class") or "__MISSING__",
                            []).append(t["name"])
    return {
        "vocabulary": list(vocabulary),
        "baseline_device_class": BASELINE_DEVICE_CLASS,
        "condition": DEVICE_CLASS_CONDITION,
        "targets_by_device_class": {k: sorted(v) for k, v in by_class.items()},
        "targets_without_device_class": missing,
        "undeclared_device_classes": undeclared,
        # DERIVED (gotcha 20).
        "verdict": ("COVERED" if not missing and not undeclared
                    else "TARGET_WITHOUT_DEVICE_CLASS" if missing
                    else "UNDECLARED_DEVICE_CLASS"),
    }


def build_device_coverage(targets: tuple[dict, ...] | None = None,
                          classes: tuple[str, ...] | None = None,
                          provenance: tuple[str, ...] | None = None) -> dict:
    """`build_device_class` and `build_device_provenance` travel TOGETHER or
    not at all -- amendment BK.

    The field is OPTIONAL: a target that states none inherits
    `BASELINE_DEVICE_CLASS` as a DOCUMENTED DEFAULT, which is how every target
    predating BK keeps exactly the description it already had. What is NOT
    optional is the pairing. A `build_device_class` with no provenance flag is
    the defect the flag exists to prevent -- an inferred device becomes
    indistinguishable from a measured one the moment nothing records which it
    was (gotcha 15). A provenance flag with no class is a claim about nothing.

    **BOTH FAILING BRANCHES AND THE UNDECLARED-VALUE BRANCHES ARE UNREACHABLE
    FROM THE REAL `TARGETS`**, which are consistent -- so `targets` and the two
    vocabularies are INJECTABLE for the same reason `tolerance_coverage`'s and
    `device_class_coverage`'s are (gotcha 16's fixture lesson: a fixture that
    cannot exhibit a branch is not evidence the branch works). The defaults are
    the module's own, so no caller changes.

    DERIVED, never asserted (gotcha 20): `inheriting_the_default` is computed
    from the targets, so a target added without the field appears there rather
    than silently acquiring `rtx_5000_ada`.
    """
    targets = TARGETS if targets is None else targets
    classes = BUILD_DEVICE_CLASSES if classes is None else classes
    provenance = BUILD_DEVICE_PROVENANCE if provenance is None else provenance

    stating, inheriting = [], []
    class_without_provenance, provenance_without_class = [], []
    undeclared_classes, undeclared_provenance = set(), set()
    for t in targets:
        c, pv = t.get("build_device_class"), t.get("build_device_provenance")
        if c is None and pv is None:
            inheriting.append(t["name"])
            continue
        if c is not None and pv is None:
            class_without_provenance.append(t["name"])
        elif pv is not None and c is None:
            provenance_without_class.append(t["name"])
        else:
            stating.append(t["name"])
        if c is not None and c not in classes:
            undeclared_classes.add(c)
        if pv is not None and pv not in provenance:
            undeclared_provenance.add(pv)

    by_class: dict[str, list[str]] = {}
    for t in targets:
        by_class.setdefault(t.get("build_device_class")
                            or f"__DEFAULT__{BASELINE_DEVICE_CLASS}",
                            []).append(t["name"])

    if class_without_provenance or provenance_without_class:
        verdict = "UNPAIRED_BUILD_DEVICE_FIELDS"
    elif undeclared_classes or undeclared_provenance:
        verdict = "UNDECLARED_BUILD_DEVICE_VALUE"
    else:
        verdict = "COVERED"
    return {
        "build_device_classes": list(classes),
        "build_device_provenance_values": list(provenance),
        "module_default": BASELINE_DEVICE_CLASS,
        "targets_stating_their_own": sorted(stating),
        "targets_inheriting_the_default": sorted(inheriting),
        "targets_by_build_device": {k: sorted(v) for k, v in by_class.items()},
        "class_without_provenance": sorted(class_without_provenance),
        "provenance_without_class": sorted(provenance_without_class),
        "undeclared_build_device_classes": sorted(undeclared_classes),
        "undeclared_build_device_provenance": sorted(undeclared_provenance),
        "verdict": verdict,
    }


def compare_manifests(previous: dict, current: dict) -> dict:
    """IS THIS AN ADDITION, A RELOCATION, OR A REBASELINE? -- BB section 4, BT.

    `capture` re-describes every target and writes whatever is on disk today,
    so a target that had DRIFTED would be silently re-baselined by the same
    call that adds a new one. **The three are different operations with
    different justifications**, and BB section 4 authorises only the first.

    So this compares, per target that existed before: every baseline field for
    EXACT equality. A pre-existing target whose baseline moved is a REBASELINE
    and is reported as a problem, never folded into the addition.

    A target present before and ABSENT now is also a problem -- a manifest
    that quietly drops a target is a check that stops checking.

    ---------------------------------------------------------------------
    RELOCATION IS A THIRD CATEGORY (amendment BT, part A3).
    ---------------------------------------------------------------------
    `describe` stores the RESOLVED ABSOLUTE path, so moving a target's file to
    another filesystem moves the `path` baseline field while every field
    describing its CONTENT stays identical. Before BT that was indistinguishable
    from a rebaseline, and the only way to record a move was to drop
    `--assert-addition-only` -- which removes the guard entirely, i.e. exactly
    the "rebaseline wearing a path change's clothes" the instruction to stop
    exists to prevent.

    A `relocated` entry therefore requires the moved-field set to be EXACTLY
    `{"path"}`. `sha256`, `bytes`, `n_leaves`, `status`, `arm`, `hash_scope` and
    `first_baselined_at` must all be unchanged; if any of them moved as well,
    the entry is a REBASELINE and stays a problem. **The classification is a
    conjunction over the content fields, not a judgement about intent** -- a
    file whose path AND bytes both changed cannot reach the relocation branch.

    A relocation is NOT a problem, so it does not by itself fail
    `--assert-addition-only`; the CLI requires `--allow-relocation` on top, so
    the move has to be declared at the call site rather than inferred here.
    """
    prev = {t["name"]: t for t in previous.get("targets", [])}
    cur = {t["name"]: t for t in current.get("targets", [])}
    added, rebaselined, relocated, unchanged, removed = [], [], [], [], []
    for name, t in cur.items():
        if name not in prev:
            added.append({"name": name, "arm": t.get("arm"),
                          "path": t.get("path"),
                          "first_baselined_at": t.get("first_baselined_at")})
            continue
        moved = {k: {"was": prev[name].get(k), "now": t.get(k)}
                 for k in BASELINE_FIELDS
                 if prev[name].get(k) != t.get(k)}
        if set(moved) == {"path"}:
            relocated.append({
                "name": name, "arm": t.get("arm"),
                "was": prev[name].get("path"), "now": t.get("path"),
                # SHOWN, not asserted in prose: the content fields a reader has
                # to see to accept that this is a move and not a rewrite.
                "sha256_unchanged": prev[name].get("sha256"),
                "bytes_unchanged": prev[name].get("bytes"),
                "n_leaves_unchanged": prev[name].get("n_leaves"),
                "content_fields_compared": list(CONTENT_FIELDS)})
        elif moved:
            rebaselined.append({"name": name, "fields": moved})
        else:
            unchanged.append(name)
    for name in prev:
        if name not in cur:
            removed.append(name)
    problems = []
    for r in rebaselined:
        problems.append(f"{r['name']}: baseline moved on "
                        f"{sorted(r['fields'])} -- that is a REBASELINE, not "
                        f"an addition")
    for name in removed:
        problems.append(f"{name}: present in the previous manifest and absent "
                        f"from this one")
    # DERIVED, never asserted (gotcha 20), and TOTAL over the four inputs
    # (gotcha 32c: a summary formatter needs a branch for every state).
    if problems:
        verdict = "NOT_AN_ADDITION_ONLY_CHANGE"
    elif added and relocated:
        verdict = "ADDITION_AND_RELOCATION"
    elif added:
        verdict = "ADDITION_ONLY"
    elif relocated:
        verdict = "RELOCATION_ONLY"
    else:
        verdict = "NO_CHANGE"
    return {
        "n_targets_before": len(prev),
        "n_targets_now": len(cur),
        "added": added,
        "unchanged": sorted(unchanged),
        "rebaselined": rebaselined,
        "relocated": relocated,
        "removed": removed,
        "problems": problems,
        "verdict": verdict,
        "fields_compared": list(BASELINE_FIELDS),
        "content_fields": list(CONTENT_FIELDS),
        "note": "`captured_at` is EXCLUDED: it moves on every capture by "
                "construction, so comparing it would build a check that can "
                "only fail (gotcha 25a).",
    }


def verify(root: Path, manifest: dict) -> dict:
    """Compare on-disk artifacts against the manifest, per arm."""
    rows, problems = [], []
    for t in manifest["targets"]:
        cur = describe(root / t["path"])
        if t.get("status") == "ABSENT":
            rows.append({"name": t["name"], "arm": t["arm"],
                         "verdict": "NO_BASELINE",
                         "note": "absent when the manifest was captured"})
            continue
        if cur["status"] == "ABSENT":
            rows.append({"name": t["name"], "arm": t["arm"],
                         "verdict": "MISSING_NOW"})
            problems.append(f"{t['name']}: baselined but now absent")
            continue
        if t["hash_scope"] == "whole_file":
            same = cur["sha256"] == t["sha256"]
            v = "REPRODUCES" if same else "DIFFERS"
            if not same:
                problems.append(f"{t['name']}: sha256 moved")
        else:
            # A whole-file hash is meaningless here (embedded provenance), so the
            # structural fingerprint is checked and the value comparison is
            # delegated. Reported as such rather than dressed up as a hash match.
            same = cur.get("n_leaves") == t.get("n_leaves")
            v = "STRUCTURE_OK" if same else "STRUCTURE_DIFFERS"
            if not same:
                problems.append(
                    f"{t['name']}: leaf count {t.get('n_leaves')} -> "
                    f"{cur.get('n_leaves')}")
        rows.append({"name": t["name"], "arm": t["arm"], "verdict": v,
                     "hash_scope": t["hash_scope"],
                     "baseline_sha256": t.get("sha256"),
                     "current_sha256": cur.get("sha256"),
                     "baseline_n_leaves": t.get("n_leaves"),
                     "current_n_leaves": cur.get("n_leaves"),
                     "first_baselined_at": t.get("first_baselined_at")})
    by_arm: dict[str, dict[str, int]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], {}).setdefault(r["verdict"], 0)
        by_arm[r["arm"]][r["verdict"]] += 1
    return {"rows": rows, "by_arm": by_arm, "problems": problems,
            "verdict": "OK" if not problems else "FAILED"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="the analysis directory holding the targets")
    ap.add_argument("--scratch-root", type=Path, default=None,
                    help="the SCRATCH directory holding targets that declare "
                         "root=scratch (BS's seventh input). Omitting it makes "
                         "those targets resolve under --root and be reported "
                         "ABSENT -- visibly wrong, never silently matched.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--capture", action="store_true",
                    help="write the manifest instead of verifying against it")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--assert-addition-only", action="store_true",
                    help="on capture, REFUSE if any pre-existing target's "
                         "baseline moved. BB section 4 authorises an addition "
                         "to coverage and NOT a rebaseline; this is what makes "
                         "the difference a measurement rather than a claim.")
    ap.add_argument("--allow-relocation", action="store_true",
                    help="permit a pre-existing target whose ONLY moved "
                         "baseline field is `path` (amendment BT part A3). "
                         "Content fields must all be unchanged -- a path that "
                         "moved together with sha256 is a REBASELINE and is "
                         "refused whatever this flag says. Required on top of "
                         "--assert-addition-only so a move is declared at the "
                         "call site, never inferred.")
    args = ap.parse_args()

    cov = tolerance_coverage()
    if cov["verdict"] != "COVERED":
        print(f"[canary] FATAL: {cov['arms_without_tolerance']} name targets "
              f"but have no declared tolerance", flush=True)
        raise SystemExit(1)

    dev = device_class_coverage()
    if dev["verdict"] != "COVERED":
        print(f"[canary] FATAL: device-class coverage is {dev['verdict']} -- "
              f"missing {dev['targets_without_device_class']}, undeclared "
              f"{dev['undeclared_device_classes']}", flush=True)
        raise SystemExit(1)
    print(f"[canary] device-class coverage: {dev['verdict']} "
          f"(baseline {BASELINE_DEVICE_CLASS}); "
          + ", ".join(f"{k}={len(v)}"
                      for k, v in sorted(dev["targets_by_device_class"].items())))

    # BK. A gate for the same reason the two above are gates: a target whose
    # build device is recorded without saying how it was established is worse
    # than one that records nothing, because it reads as measured.
    bdev = build_device_coverage()
    if bdev["verdict"] != "COVERED":
        print(f"[canary] FATAL: build-device coverage is {bdev['verdict']} -- "
              f"class without provenance {bdev['class_without_provenance']}, "
              f"provenance without class {bdev['provenance_without_class']}, "
              f"undeclared {bdev['undeclared_build_device_classes']}"
              f"{bdev['undeclared_build_device_provenance']}", flush=True)
        raise SystemExit(1)
    print(f"[canary] build-device coverage: {bdev['verdict']}; "
          f"{len(bdev['targets_stating_their_own'])} state their own, "
          f"{len(bdev['targets_inheriting_the_default'])} inherit the "
          f"{BASELINE_DEVICE_CLASS} default; "
          + ", ".join(f"{k}={len(v)}"
                      for k, v in sorted(bdev["targets_by_build_device"].items())))

    if args.capture:
        prev = (json.loads(args.manifest.read_text())
                if args.manifest.exists() else None)
        need_scratch = [t["name"] for t in TARGETS
                        if t.get("root") == "scratch"]
        if need_scratch and args.scratch_root is None:
            print(f"[canary] WARNING: {len(need_scratch)} target(s) declare "
                  f"root=scratch ({', '.join(need_scratch)}) and no "
                  f"--scratch-root was given; they will resolve under --root "
                  f"and be captured ABSENT", flush=True)
        man = capture(args.root, args.repo, prev, args.scratch_root)
        delta = (compare_manifests(prev, man) if prev is not None else
                 {"verdict": "NO_PREVIOUS_MANIFEST", "added": man["targets"],
                  "problems": []})
        print(f"[canary] captured {len(man['targets'])} targets at "
              f"{man['identity']['captured_at_git_sha'][:10]}")
        for t in man["targets"]:
            print(f"  {t['arm']:>9}  {t['name']:<20} {t['status']:<8} "
                  f"{t.get('sha256', '')[:16]}  leaves="
                  f"{t.get('n_leaves', '-')}  first={t['first_baselined_at'][:10]}")
        print(f"[canary] addition-vs-rebaseline: {delta['verdict']}")
        for a in delta.get("added", []):
            nm = a["name"] if isinstance(a, dict) else a
            print(f"  ADDED       {nm}")
        for r in delta.get("rebaselined", []):
            print(f"  REBASELINED {r['name']}: {sorted(r['fields'])}")
        for r in delta.get("relocated", []):
            # The hash is printed WITH the move, so the two are read together
            # and a reader never has to take "only the path changed" on trust.
            print(f"  RELOCATED   {r['name']}: {r['was']} -> {r['now']}")
            print(f"              sha256 UNCHANGED {r['sha256_unchanged']}  "
                  f"bytes UNCHANGED {r['bytes_unchanged']}  "
                  f"(content fields compared: "
                  f"{','.join(r['content_fields_compared'])})")
        for p in delta.get("problems", []):
            print(f"  PROBLEM {p}")
        if args.assert_addition_only:
            if delta["problems"]:
                print("[canary] REFUSING: this capture is not addition-only",
                      flush=True)
                raise SystemExit(1)
            if delta.get("relocated") and not args.allow_relocation:
                print(f"[canary] REFUSING: {len(delta['relocated'])} target(s) "
                      f"changed path. That is a RELOCATION and it must be "
                      f"declared with --allow-relocation, not absorbed into an "
                      f"addition", flush=True)
                raise SystemExit(1)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(man, indent=2))
        print(f"[canary] wrote {args.manifest}")
        return

    man = json.loads(args.manifest.read_text())
    rep = verify(args.root, man)
    for r in rep["rows"]:
        print(f"  {r['arm']:>9}  {r['name']:<20} {r['verdict']}")
    print(f"[canary] by arm: {json.dumps(rep['by_arm'])}")
    for p in rep["problems"]:
        print(f"  PROBLEM {p}")
    print(f"[canary] verdict {rep['verdict']}")
    if rep["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
