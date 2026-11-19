# esm2-saes-cross-seed-reproducibility
What this is for: the front page of the released code — which study it belongs to, what the release contains, how to check the study's reported numbers against it, and where the project's record is.
Who reads it: a reader who has the paper in hand and has not opened any other document here. `GUIDE.md` is the map of the repository; `RELEASE_NOTES.md` lists what each part of the release contains.
Current as of: 2026-11-18 (the release built from the working repository's commit `8f6a313`).

Which interpretable features in ESM-2 survive reseeding, and is survival
predictable from biology and geometry? In the main experiment, forty TopK
sparse autoencoders, each called a dictionary here, are trained on ESM-2-650M
layer-24 activations with only the random seed varied (`n_seeds` = 40). Twenty
of them, the anchors (`n_anchors` = 20), carry the alignment measurement, and
each anchor is compared with the other 39 dictionaries (`n_compare` = 39). The
test is whether features aligned with a biological annotation reproduce across
seeds better than features aligned with none (orphans) **once firing rate is
controlled**. A further hundred dictionaries were trained for the study's
comparisons across dictionary width and `k`, in nine more sets ("What is
released" lists them). All experiment parameters are in `configs/mvp.yaml` and,
for the nine other sets, in the files of the folders beside it.

`n_seeds`, `n_anchors` and `n_compare` are the project record's names for
these three counts. They are not read from the configuration: each of the
10 configuration files sets `sae.n_seeds: 20`, and the
training code does not read that key. A training run takes one seed from its
command line, and the job arrays give the seeds: the first twenty in
`slurm/train_sae_array.sbatch` and the other twenty in
`slurm/train_ext_seeds.sbatch`, both with `configs/mvp.yaml`.

## Paper

Author: Theodore He.

Title of the workshop paper: "Cross-seed reproducibility depends on the measurement, not only the model: sparse autoencoder features in a protein language model". The extended version's title
may differ.

Venue: Interpretability as a Science: NeurIPS 2026 Workshop (InterpScience 2026). The workshop is non-archival.

**Scope.** This release accompanies the full study. The workshop paper
reports a subset of it: `RELEASE_NOTES.md` §2.5 lists, by name, the analyses
in the release that the workshop paper does not report, with the rows of the
coverage table behind them. An extended version of the paper is in
preparation.

`CITATION.cff` cites this software and gives the workshop paper, under the
title above, as the preferred citation. The extended version's citation will
come with it.

## What is released

**This repository**, https://github.com/TheoCyHe/esm2-saes-cross-seed-reproducibility: the code, and the manifests of the data.
The work was done in a private working repository, which keeps the full
history; this repository is an export of one commit of it and has a single
commit of its own. In the export, account names, cluster and local paths, and
other identifiers are replaced by placeholders such as `$PROJECT` and
`<netid>` (`RELEASE_NOTES.md` §3 has the table of replacements). The code is
under the MIT licence (`LICENSE`). The copy of this repository that Zenodo
archives has a DOI of its own, which is listed among the related identifiers
of the data record below.

**Two sets of data files**, under CC-BY 4.0. Each has a manifest in this
repository: a table with one row per file and the file's sha256.
- the **derived bundle**, the derived data from which most of the study's
  reported numbers are recovered, as tables and the analyses' result files: the
  126 files `release/derived/MANIFEST.csv` names, and the
  manifest itself (441 MB in all). Its two metadata files,
  `meta.json` and `meta_ext.json`, are also in this repository.
  `RELEASE_NOTES.md` §6 says what every column of its tables other than the
  manifest means;
- the **decoders**, the decoder matrix of every trained dictionary:
  10 files, one per set of dictionaries, 10.2 GB in
  all (`release/decoders/MANIFEST.csv`). The sets are the 40-seed ensemble;
  `k = 64` at the ensemble's width (20 dictionaries); four widths at the
  ensemble's `k` (10 each); and four other combinations of width and `k`
  (10 each): 140 dictionaries. The study's "nine arms" are
  the nine sets other than the ensemble.

Where the data files are published:
- with this repository's GitHub release `v1.0`, as its assets
  (`https://github.com/TheoCyHe/esm2-saes-cross-seed-reproducibility/releases/download/v1.0/<file name>`): the 127 files of the
  bundle, its manifest among them under the name `MANIFEST.csv`, and the
  10 decoder files. The decoders' manifest is not an asset;
  it is `release/decoders/MANIFEST.csv` here;
- on Zenodo, DOI `10.5281/zenodo.24074394`, in 14 files:
  `release_derived.tar`, the bundle as one archive;
  `derived_MANIFEST.csv` and `decoders_MANIFEST.csv`, which are copies of
  `release/derived/MANIFEST.csv` and `release/decoders/MANIFEST.csv` here
  (both are named `MANIFEST.csv` in this repository);
  `reference_environment.yml`, a copy of `env/reference_environment.yml`;
  and the 10 decoder files.

**The project record is not public.** It is what the project wrote down as it
went: the pre-registration log (the decisions fixed in advance and every later
change to them; the workshop paper calls it "the specification document", and
it is not the record's file `docs/SPECIFICATION.md`, which specifies the
regression and the matched-pair arm), the table of numbers (the study's
main results, each with the result file or log section that owns it; the
other numbers trace to the log or to a working document, as the owner column
of `RELEASE_NOTES.md` §2 says), the corrections
log, the documents that map the code, and evidence files, each named one by
one by the table of numbers, the corrections log or the prediction ledger.
The record is deposited at DOI `10.5281/zenodo.24226557`; its files are
restricted and are available on request, through the request form of that
record. (A few parts are left out of it or replaced: the log's later sections,
which are the release's own decisions, briefs and checks; the list of the
reviewers' requests; three corrections entries; and a few words. The record's
own README lists them.) **The
prediction ledger ships with the code**: `docs/prediction_ledger.json`.
It itemises what was filed from amendment AW of the log on:
60 entries, each with its outcome. 44 of
them are predictions that were scored as a hit or a miss; the other
16 are marked NOT_SEALED (10), UNMEASURABLE (4), STRUCTURAL_CHECK (2) and are not counted in its tallies. Some
entries' own notes say what was known when they were filed; not every note does. The 70 predictions scored before that amendment
are in it as 6 totals by category, not item by item.

**Citations in the code.** Comments, docstrings and messages cite the working
repository's documents by path and by section or entry number, and the
citations are left as written. Where they resolve:
- most name documents of the project record (`docs/NUMBERS_OF_RECORD.md`,
  `PREREGISTRATION.md`, the corrections log), which are not here, apart from
  the 4 files `GUIDE.md` §1 lists;
- `CLAUDE.md`, `docs/RELEASE.md`, `docs/CONCEPT_INVENTORY.md`, `docs/LITERATURE.md`, `docs/RETROSPECTIVE_VERIFICATION.md` are documents of the working repository that are
  in neither part of the release. `CLAUDE.md` is its operating manual: a
  citation of "`CLAUDE.md` gotcha n" is a numbered rule whose case is entry n
  of the record's `docs/GOTCHAS.md`. `docs/RELEASE.md` is its release
  checklist; `RELEASE_NOTES.md` here has its coverage table, and lists the
  runs made with the released script;
- the sections of the pre-registration log numbered §CT18-D… are the
  release's own decisions, briefs and checks. §CT18-D, §CT18-D2, §CT18-D3, §CT18-D3-R, §CT18-D4, §CT18-D4-R, §CT18-D5 are in the record: they
  declare the two reproduction tiers, the decoder files and the bundle's
  build. The later ones (§CT18-D6 onwards) and §CT1, which lists the
  reviewers' requests, are left out of it. Three of those are cited here:
  §CT18-D10 (how the log was split into files), §CT18-D11 decision 6 (the
  check of the decoder files against the trained dictionaries) and
  §CT18-D17 (that the project record is deposited apart); the comments and
  docstrings that cite them restate what was declared. (§CT18-D9 appears in
  `src/analysis/prereg_log.py` only as an example of a section label);
- by the build's count, 12 other files that the code names are not
  released either: 9 under `analysis/adhoc/` (ad hoc and probe scripts, and outputs), 2 under `release/zenodo/` (the deposits' working files) and 1 under `docs/`.
  The count takes a bare file name as the one file of the working repository
  that has that name, so it includes a few names that are not a citation of
  that file: a test's fixture, an output's file name, or a file of the same
  name in another place.

Reproducing the numbers needs none of these.

## Setup

**The documented way to reproduce the study's reported numbers is the pinned
reference environment** (Linux x86_64):

```bash
conda env create -f env/reference_environment.yml
conda activate esm2-sae-reference
```

It is the cluster's environment as exported on 2026-11-01. In it the
reproduction script gives the record's values bit for bit.

`env/environment.yml` is the everyday environment (`conda activate esm-interp`).
It pins nothing, so what it installs changes with time: it is a best-effort
file. It does not list `mmseqs2`, which is not built for Windows; only the
pipeline's clustering step needs it, and the file's header says how to add it.

Both files install PyTorch. The reproduction script and the tests need it
installed and use no GPU, so a CPU-only build is enough.

## Reproducing the reported numbers

**In short.** Most of the study's reported numbers are recovered from the
derived bundle alone, with no cluster, no GPU and no path to set, and level 2
below regenerates the main ones. The coverage table (`RELEASE_NOTES.md` §2) is
keyed to the study's reported numbers, of which the workshop paper reports a
subset (§2.5 there). It has one row for each reported number or group of
numbers, 195 rows, among them a row for every measured result that
the workshop paper prints, and gives each row a coverage level:
- 148 rows are recovered from the bundle, at one of four
  coverage levels:
  - REGENERATED (33): recomputed by
    `scripts/reproduce_table1.py`, level 2 below;
  - RECOMPUTED (39): recomputed by a check that ran when the
    bundle was built;
  - RECOUNTED (24): counted again from rows the bundle ships;
  - RE-READ (52): read from an owner file, which is the result
    file an analysis wrote and the record takes the number from; the bundle
    ships these as `owner__*.json`;
- 42 rows are NOT IN BUNDLE, each with its reason there.
  They include setup facts, published comparators, figures read from files or
  job logs that are not released, and the tallies of the prediction ledger,
  which `python -m src.analysis.prediction_ledger` derives from the ledger
  file in this repository;
- 5 rows are for a passage of the study's text that has no number in it.

**Re-running the pipeline from raw data (level 1) does not work as shipped:**
it needs the `$PROJECT`, `$SCRATCH` and `$HOME` placeholders set, in the
124 files `release/path_placeholders.tsv` lists. Level 1
below says how.

**There are three levels of reproduction.**

1. **Pipeline reproduction** reruns everything from the Swiss-Prot release
   through training and analysis. It needs the activation cache (ESM-2's
   layer-24 activations, one row per residue), which is regenerated rather
   than stored, and GPU time. Checkpoints, average-precision tensors,
   activation caches and null arrays (each feature's alignment score
   recomputed against annotations shifted circularly within each protein,
   which give the score's null distribution) are not in the release.
   **The path placeholders.** Paths written `$PROJECT/...`, `$SCRATCH/...` and
   `$HOME/...` in the code, the job scripts and the configuration files are
   placeholders for the author's cluster directories: the export wrote them in
   place of absolute paths. No file here gives `$PROJECT` or `$SCRATCH` a
   directory (where a job script reads `PROJECT=$PROJECT`, it assigns the
   variable to itself), and no
   Python or configuration file expands a placeholder: there it is a literal
   string, as in `Path("$PROJECT")`. (In a shell script `$HOME` is the
   shell's own variable, the home directory of whoever runs it.) A rerun
   replaces them with its own directories first, in the files
   `release/path_placeholders.tsv` lists (`GUIDE.md` §8 counts them by
   folder). In a job script's command lines, exporting variables of the same
   names has the same effect; `#SBATCH` lines, Python files and configuration
   files need the edit. The partition and GPU names are the cluster's.
   **The decoders ship.** Each decoder file holds `W_dec` (dictionaries ×
   1,280 × width, float32) and its seed labels. The build checked every matrix
   against its checkpoint bit for bit, and the export checked every matrix
   again when it rewrote the files' attributes.
   - **What they enable:** decoder-space matching. Every decoder-space match
     rate in the study is a function of these matrices alone. No script here
     computes a rate from the decoder files: the released matching code reads
     the same matrices from the checkpoints, so using the files means loading
     `W_dec` from them instead. To verify a downloaded file, compare its
     sha256 with its row of `release/decoders/MANIFEST.csv`. (The one tool
     here that reads the files, `src/analysis/release_decoders.py
     --verify-dir`, compares them with the checkpoints, which are not
     released.)
   - **The concept-subspace comparisons need more:** they choose each concept's
     features by average precision against its annotation, and those scores
     are not released.
   - **Device class:** the record's match counts, one for each feature of
     each anchor (204,800 in all), were computed on GPU. Between CPU and
     GPU, 4 of them differ at the similarity threshold θ = 0.70.
   - **What they do not enable:** the matching rule that requires the encoder
     matching and the decoder matching to agree (the record's rule 3), which
     needs the encoders;
     and alignment scores, and the activation-correlation matching and its
     rates, which need the activation cache and the encoders. Neither is
     released.
2. **Recomputation from the derived bundle.**

   ```bash
   python scripts/reproduce_table1.py --bundle <an empty folder> \
       --download https://github.com/TheoCyHe/esm2-saes-cross-seed-reproducibility/releases/download/v1.0
   ```

   downloads every file the manifest names into that folder, verifies each
   against the manifest in this repository, and refuses if a file in the
   folder is absent or altered. (A file missing at the URL stops the download
   with an error.) With the Zenodo copy, unpack `release_derived.tar` and pass
   `--bundle <the unpacked release_derived folder>` without `--download`; the
   files are then verified against the `MANIFEST.csv` in that folder, so check
   first that it is identical to `release/derived/MANIFEST.csv` here. Without
   `--download` and `--bundle`, the script reads `release/derived/` in place.
   In a fresh clone that folder holds 3 of the bundle's 127
   files (the manifest and the two metadata files), so the default works only
   after the other 124 have been put there.
   It then recomputes the following from the bundle alone, and checks each
   against the record's value at the record's rounding:
   - the headline mean match rate, and its curve's endpoints at θ = 0.30 and
     0.90;
   - Table 1: its three slopes with their fixed-effect intervals, and for
     the two beta-binomial rows the across-seed and jackknife intervals;
   - the single-matched carrier excess: how much more often a concept's
     carrier (the feature with the highest average precision for it in one
     dictionary) has a matched partner in another dictionary than the other
     features of the same firing-rate bin of its dictionary do. The carrier
     itself is left out of that baseline (its bin's matches less its own,
     over the bin's features less one). A carrier that is alone in its bin
     has no baseline: it counts in the carriers' rate and not in the
     baseline's mean. `RELEASE_NOTES.md` §6 describes every column of the
     bundle's tables;
   - the nine-arm width-and-`k` table.

   The record's values are written in the script; their source is the
   record's table of numbers.

   **What the labels refer to.** The script's name, its docstring, the
   labels it prints and some comments in the code name tables and appendices
   ("Table 1", "Appendix K"), and some comments say "camera-ready". These,
   and the row labels of the coverage table (`RELEASE_NOTES.md` §2), refer to
   the planned full write-up of the study. That write-up was first planned as
   the camera-ready text, before the workshop paper was scoped to a subset of
   the study. It is not in this repository, and the workshop paper's own
   labels need not match.

   **It reports one of two tiers.** The tier is the standard the checks are
   held to, and the environment the script runs in decides it.
   - **REFERENCE**: Linux, with the Python, NumPy, SciPy, h5py and OpenBLAS
     versions of `env/reference_environment.yml`, and one BLAS thread. Every
     value is held to the record's rounding.
   - **PORTABLE**: any other environment. Counted values and fixed effects are
     still held to the record's rounding. A fitted interval endpoint passes
     within 0.001 percentage points of the record, and is reported as
     PASS_BY_TOLERANCE.

   **What it prints.** On standard output: with `--download`, how many files
   were fetched; then the tier, one line per check with the record's value
   and the recomputed one, and a last line,
   `94 of 94 checks pass (tier REFERENCE)`. In the
   PORTABLE tier that line names the tier and, where fitted interval endpoints
   passed by tolerance, how many. On standard error, while the fits run:
   progress, one line per stage and one per fit, with the time elapsed. It
   exits 0 only if every
   check passes; a line that begins `FAIL` names the value that differs.

   **What it needs.** PyTorch installed (the fit's module imports it; no GPU
   is used); under 2 GB of memory, where it was measured (on Linux); and, in
   the recorded runs, 81 minutes on four cores of a shared node of the cluster the record's numbers were computed on, in 1 run; 42 to 49 minutes on a workstation with an AMD Ryzen 7 7840U processor, in 4 runs
   (`RELEASE_NOTES.md` §4 lists them; one of the workstation's runs is not
   in the table below). The fits run serially on one BLAS thread. The
   script's docstring also gives run times, which it says are of runs before
   this version of the script, and counts of checks that failed at the
   record's rounding in earlier runs, which it dates. Neither is of the runs
   listed here.

   **Where it has been run.** With the script and the bundle as released, all
   94 checks pass in each of these (`RELEASE_NOTES.md` §4 gives the
   package versions and the run times):

   | environment | tier reported | result |
   |---|---|---|
   | the cluster the record's numbers were computed on (4 cores of a shared node; job 33328511), its own environment, of which `env/reference_environment.yml` is the export | REFERENCE | 94 of 94 |
   | Linux (WSL 2 on a workstation), a fresh environment from `env/reference_environment.yml` | REFERENCE | 94 of 94 |
   | Linux (WSL 2 on a workstation), a fresh environment from `env/environment.yml` | PORTABLE | 94 of 94 |
   | Windows 11 (the same workstation), a fresh environment from `env/environment.yml` | PORTABLE | 94 of 94, 4 of them fitted interval endpoints within the tolerance |
3. **The provenance check.** Most result files record the commit they were
   computed at, and the working repository's history records when each section
   of the pre-registration log was committed. For each result file,
   `scripts/e2_order_check.py` compares the commit that the file records with
   that of the section the file names. It needs that history, which
   this repository does not carry, so it cannot be rerun here. Its output is
   part of the project record, and `RELEASE_NOTES.md` §1 gives its counts,
   its limits and what it shows for the results the workshop paper prints.

## Tests

```bash
python -m pytest tests/ -q --deselect tests/test_reproduce_table1.py::test_reproduce_table1_on_the_bundle
```

No GPU, ESM download or cluster access is needed. The deselected test is
level 2 run as a test: it needs the whole bundle in `release/derived/` (a
fresh clone has 3 of its files there) and takes as long as the script does. `GUIDE.md` §4 says which tests of the working
repository are not here, which skip, and where the suite has been run.

The suite was developed on Windows, with MKL, and is also run on Linux in
both environments above. On Windows, an environment built from
`env/environment.yml` holds two OpenMP runtimes (MKL's, with NumPy from conda,
and PyTorch's, from pip). The whole suite runs there, but a test file run
alone can fail at `import torch`, depending on what was imported before it.
The reproduction script is not affected: it imports PyTorch first and uses
one thread.

## Requests

For the project record's files, use the request form of its Zenodo record
(DOI `10.5281/zenodo.24226557`). For questions about the release, open an issue at
https://github.com/TheoCyHe/esm2-saes-cross-seed-reproducibility/issues.
