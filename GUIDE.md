# GUIDE.md — a reader's map of this repository
What this is for: what each folder and each script holds, which record files are kept here and what reads them, which tests are absent or skip, where each claim of the study is checked, and which files carry a path placeholder. `README.md` says what the release is and how to reproduce the study's reported numbers; `RELEASE_NOTES.md` lists what each part of the release contains.
Who reads it: a reader of the released repository.
Current as of: 2026-11-18 (the release built from the working repository's commit `8f6a313`).

## 1. What this repository is

An export of a private working repository, built from its commit `8f6a313d9cc325a87dbf408fed71cab5e5dd507b`. `README.md` ("What is released") says what that means and what the release's other parts are. The release accompanies the full study. The workshop paper reports a subset of it (`RELEASE_NOTES.md` §2.5 lists what it leaves out), and an extended version of the paper is in preparation. The repository holds 429 files; `RELEASE_NOTES.md` §1 counts them by kind.

**4 of them are record and evidence files.** The project record is not public (`README.md`, "What is released"); these are here because released code or a released test reads them:

| file | read by |
|---|---|
| `analysis/adhoc/probes/2026-10-24/ct24_ct25/slope_w2560_ct.json` | one test of `tests/test_release_derived_ext.py` |
| `analysis/adhoc/probes/2026-10-29/release/reproduce_windows.json` | one test of `tests/test_reproduce_table1.py` |
| `docs/ct22_tuning_set.json` | `src/analysis/concept_subspace.py`, `src/analysis/ct26_nonanchor.py` and four job scripts: an input of the concept-subspace analyses |
| `docs/prediction_ledger.json` | `src/analysis/prediction_ledger.py`, which derives the tallies the study reports, and twelve tests of `tests/test_prediction_ledger.py` |

## 2. The code

- **`src/`** is the pipeline: data and split (`src/data/`), embeddings and the activation cache (`src/embeddings/`), annotations (`src/annotations/`), SAE training (`src/sae/`), the analyses (`src/analysis/`), the helpers for the cluster's scheduler (`src/cluster/`) and the configuration loader (`src/experiment_config.py`).
- **`scripts/`** holds seven tools:
  - `reproduce_table1.py` regenerates the study's main numbers from the derived bundle (`README.md`, level 2).
  - `e2_order_check.py` is the ordering check (`README.md`, level 3). It needs the working repository's history.
  - `ct22_draw_tuning_set.py` drew, once, the tuning set of the concept-subspace analyses and wrote `docs/ct22_tuning_set.json`.
  - `submit_pinned.sh` clones the repository at one commit and submits a job from the clone only if the clone is that commit and is clean.
  - `remote_wait.sh` waits for cluster jobs to finish, reading both the scheduler's queue and its accounting database before it reports them done.
  - `deposit_package.py` wrote the checksum lists of the data deposit, and `zenodo_verify.py` compares a Zenodo draft's file list with them. Both name `release/zenodo/` as their default folder, which is not exported.
- **`slurm/`** holds the batch scripts as they ran, **`configs/`** the experiment parameters, **`env/`** the two environment files, `activate.sh` (which the batch scripts source to activate the environment) and `verify_sync.sh` (which checks that a cluster checkout is at an expected commit and clean), **`tests/`** the test suite.
- **`canary/`** holds `baseline.json`: for each stored artifact the record depends on, its hash and its count of values, with the commit the expectation starts from. `src/analysis/canary.py` compares the artifacts with it, so a change in what the pipeline reproduces is detected. It needs the artifacts, which are not released.
- **`migrations/`** holds four one-time scripts that renamed identity attributes, and added seed labels, on artifacts already on disk when a pre-registration amendment changed a name. They are kept as the record of what was done to those artifacts. The pipeline's code does not import them; two test files (`tests/test_ak_json_migration.py`, `tests/test_as_seed_label_migration.py`) do, and one job script ran one.
- **`release/`** holds the manifests of the two sets of data files (`derived/`, `decoders/`), `export_hash_map.csv` (§7), `path_placeholders.tsv` (§8) and `EXPORT.json`, which marks this tree as an export.
- **`docs/`** and **`analysis/`** hold only the four files of §1.
- **At the root:** `LICENSE`; `CITATION.cff`, how to cite; `.zenodo.json`, the Zenodo metadata of the code record; `pytest.ini`, which limits test collection to `tests/`; `.gitattributes` and `.gitignore`.
- **Not exported:** `scripts/export_release.py`, `scripts/export_idscan.py`, `tests/test_export_release.py`, `tests/test_export_idscan.py`, `slurm/export_decoders.sbatch`, `tests/test_prereg_log.py`, `release/zenodo/`, `release/private/`. `RELEASE_NOTES.md` §1 says what each is.

## 3. The record is not in this repository

`README.md` ("What is released") says what the project record is and where it is deposited. Three tools here work on the record and do nothing useful without it: `src/analysis/prereg_log.py` (reads and indexes the pre-registration log), `scripts/e2_order_check.py` (which also needs the working repository's history) and the stated-hash gate inside `src/analysis/release_derived_ext.py` (the check, made when the bundle was built, of the sha256 of each stored artifact that the build read against the hash the record states for it; `RELEASE_NOTES.md` §2.6).

**What the comments cite.** Comments, docstrings and messages in the code that cite the instruction file (`CLAUDE.md`), a numbered "gotcha" or a section label such as §CT18-F refer to the project's working instructions and record, which are not in this repository (apart from the 4 files of §1); `README.md` ("Citations in the code") says where each kind resolves. AI assistants were used under the author's direction to write and run the code and to keep the record. That workflow included verification passes, in which a separate read-only AI session re-checked the work: it checked claims against the record's documents and the code, or was asked to find what was wrong in a text before the text was accepted. The author made every decision. The record is deposited with restricted access at DOI `10.5281/zenodo.24226557`. The comments are left as written: they were not reworded in this release so that they read without the record.

## 4. The tests

`README.md` ("Tests") gives the command and says what the suite needs. Where it differs from the working repository's suite:

- **Test files not exported:** `tests/test_prereg_log.py` (10 tests), which exists only to check the pre-registration log in the working repository; `tests/test_export_idscan.py`, the tests of the second search; `tests/test_export_release.py`, the tests of the export script.
- **One test skips without the pre-registration log,** which is in the project record: `tests/test_config_audit.py::test_the_screen_reports_context_rather_than_filtering`, which screens the configuration's keys against the log's text. Its skip prints the reason.
- Every other test runs as it does in the working repository.

**Where the suite has been run, as released** (the command of `README.md`, the bundle and the record absent):

| environment | result |
|---|---|
| Linux (WSL 2 on a workstation), a fresh environment from `env/reference_environment.yml` | 2,265 passed, 1 skipped, 1 deselected |
| Linux (WSL 2 on a workstation), a fresh environment from `env/environment.yml` | 2,265 passed, 1 skipped, 1 deselected |
| Windows 11 (the same workstation), with `PYTHONUTF8=1` in the environment, a fresh environment from `env/environment.yml` | 2,264 passed, 2 skipped (the second is a test of a process start method that only POSIX systems have), 1 deselected |
| Windows 11 (the same workstation), with `PYTHONUTF8=1` in the environment, the environment the suite was developed in | 2,264 passed, 2 skipped (the same two), 1 deselected |
| Windows 11 (the same workstation), with `PYTHONUTF8` not set, the environment the suite was developed in | 2,264 passed, 2 skipped (the same two), 1 deselected |

## 5. How to reproduce

`README.md` ("Reproducing the reported numbers"): how many of the study's reported numbers are recovered at each coverage level, the three levels of reproduction, with the command for the second, and what a rerun from raw data must set. §8 here counts the files that rerun edits.

## 6. Where each claim of the study is checked

- **At which coverage level a number can be recovered from this release**: `RELEASE_NOTES.md` §2, one row for each reported number or group of numbers of the study; its §2.5 lists, by name, the analyses the workshop paper does not report.
- **Predictions and their scores**: `docs/prediction_ledger.json`, in this repository; `python -m src.analysis.prediction_ledger` derives the tallies.
- **What each column of the bundle's tables is**: `RELEASE_NOTES.md` §6.
- **The owner of a number; the order of a result file's commit and its declaration's; what was wrong and how it was corrected**: the project record (`README.md`, "What is released"). `RELEASE_NOTES.md` §1 gives the counts of the ordering check.

## 7. What the export changed

- **Identifiers replaced, and what is kept:** `RELEASE_NOTES.md` §3 has the substitution table and describes the two searches run on the result.
- **Hashes.** `release/export_hash_map.csv` gives the original and the exported sha256 of each file it covers. `RELEASE_NOTES.md` §3 says which files it covers and which documents state which hash.

## 8. Files that carry a path placeholder

The export wrote `$PROJECT`, `$SCRATCH` or `$HOME` in place of an absolute path in 124 files of code and configuration. `release/path_placeholders.tsv` lists them, one row per file, with the number of each placeholder the export added to it. By folder, with those numbers summed:

| folder | files | `$PROJECT` | `$SCRATCH` | `$HOME` |
|---|---:|---:|---:|---:|
| `canary/` | 1 | 40 | 0 | 0 |
| `configs/` | 10 | 10 | 20 | 0 |
| `env/` | 1 | 0 | 0 | 1 |
| `slurm/` | 106 | 103 | 81 | 2 |
| `src/` | 5 | 5 | 1 | 0 |
| `tests/` | 1 | 2 | 0 | 0 |

`README.md` (level 1) says what a rerun of the pipeline does with them.
