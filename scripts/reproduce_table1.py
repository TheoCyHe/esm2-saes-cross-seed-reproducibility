"""Amendment CT §CT18 (c): recompute the headline, Table 1 and the carrier
excess from `release/derived/` alone, and assert each against
`docs/NUMBERS_OF_RECORD.md` to the record's rounding.

    python scripts/reproduce_table1.py [--bundle release/derived] [--json OUT]

Exit 0 only if every bundle file matches its manifest hash and every check
passes under the run's TIER; exit 1 otherwise, with the failing checks named.

TIERS (§CT18-D4 item 3). REFERENCE -- Linux, the pinned versions of
env/reference_environment.yml (REFERENCE_ENV below), every BLAS pool at one
thread: every check is held to the record's rounding, the primary criterion.
PORTABLE -- any other environment: the BLAS library moves fitted values in the
last digits, so a fitted interval endpoint passes within PORTABLE_ENDPOINT_TOL_PP
of the record (reported PASS_BY_TOLERANCE); counted values and fixed effects are
held to the record's rounding. The script prints its tier and why. The data come from
the bundle and nothing else. The design-3 fit is this repository's own
(`src.analysis.design3_join.fit_arm`), because reimplementing a penalised
beta-binomial REML fit would test a different estimator; the carrier excess is
reimplemented here in plain numpy, independently of `carrier_support`.

WHAT IS CHECKED, and the NoR row each value is read from:
  headline mean r_f, theta = 0.70, n_compare = 39 ........ §1.1   0.495513
  threshold curve endpoints, theta = 0.30 and 0.90 ........ §1.1   0.805639, 0.088764
  mean r_f at theta = 0.70, n_compare = 19 ................ §1.1 q1 0.495815
  Table 1: binomial grid maximum, theta = 0.55 ............ §2.1   +0.3522 [+0.3208, +0.3836]
  Table 1: beta-binomial grid maximum, theta = 0.60 ....... §2.1   +0.3635 [+0.2397, +0.4873]
  Table 1: beta-binomial at the reference, theta = 0.70 ... §2.1   +0.2889 [+0.1531, +0.4248]
  carrier excess, single (firing-rate) arm, n_compare = 19  §20.17 +7.2219
  Table 1, all three intervals, theta = 0.60 and 0.70 ..... §20.4 (fixed effect, across-seed,
                                                            jackknife over the 40 dictionaries)
  the nine-arm table (Appendix K, Table K1), n_compare = 9: per arm the mean match
  rate at 0.70, FVU raw and standardised, the slope with its widest interval and
  that interval's name, and mean s_f; k = 16 and k = 128 have no primary slope and
  are checked as their labelled top-merged sensitivities ..  §19.1, §19.3, §20.27 (b),
                                                            §20.29, §20.30 (c), §20.32,
                                                            §20.33, §20.37
Slopes are percentage points of r_f per SD of log M, fixed effect, design 3,
`primary_hard_bin`; the record's rounding is the number of decimals NoR prints.
The expected values are written below from NoR, never computed from the bundle
(gotcha 32 (b)).

    python scripts/reproduce_table1.py --download BASE_URL [--bundle DIR]

fetches every file `MANIFEST.csv` names from `BASE_URL/<file name>` into the
bundle directory first, and refuses unless each hashes as the manifest in this
repository says: the manifest in git is the trust anchor, never the download.

The fits run serially in this process: the producers' process pools rely on
globals a forked worker inherits, which a spawned worker (Windows, macOS,
Python >= 3.14 on Linux) would not. RUN TIME: 34 to 76 minutes in the runs
recorded before this version of the script (about 42 minutes and 1.4 GB of
memory on Linux in the last two of them, 2026-11-13); the release's
RELEASE_NOTES.md lists the runs of this version. PROGRESS goes to standard
error while the fits run: one line per stage and one line per fit, with the
fit's number and the time elapsed. STANDARD OUTPUT carries the tier, one line
per check and the last line; with --download, one line before them says how
many files were fetched.

PYTORCH MUST BE INSTALLED (a CPU build suffices; no GPU is used): the fit's
module imports this repository's training module for its provenance helpers.
It is imported HERE, before any thread pool is limited, because on Windows a
PyTorch imported after threadpoolctl has loaded MKL's OpenMP runtime fails to
load its own (WinError 127, measured 2026-11-13).

BLAS RUNS SINGLE-THREADED, as the owners' fits and the build did
(`identity.omp_num_threads` = 1). Measured 2026-10-29 (`docs/RELEASE.md` §10):
on Linux, four BLAS threads moved 6 of the 85 numeric values by up to 2.5e-5 pp
and failed 2 checks at the record's rounding; one thread reproduced every owner
value bit for bit. On Windows (MKL) the fitted values differ from the owners by
up to 5.3e-5 pp at any thread count, and 3 checks fail at the record's rounding;
the count-based values (match rates, FVU, mean s_f, the carrier excess) are
identical on every platform run.
"""
from __future__ import annotations

import os

# before numpy is imported (a command-line run); run() also limits the pools at
# runtime, for a caller that imported numpy first
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import contextlib
import csv
import gzip
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# (name, NoR row, record value, decimals). Written down from NoR, not computed.
RECORD = {
    "headline_mean_rf_0.70_n39": ("§1.1", 0.495513, 6),
    "mean_rf_0.30_n39": ("§1.1 qualifier 3", 0.805639, 6),
    "mean_rf_0.90_n39": ("§1.1 qualifier 3", 0.088764, 6),
    "mean_rf_0.70_n19": ("§1.1 qualifier 1", 0.495815, 6),
    "binomial_0.55_fe_pp": ("§2.1", 0.3522, 4),
    "binomial_0.55_fe_ci_lo": ("§2.1", 0.3208, 4),
    "binomial_0.55_fe_ci_hi": ("§2.1", 0.3836, 4),
    "betabinomial_0.60_fe_pp": ("§2.1", 0.3635, 4),
    "betabinomial_0.60_fe_ci_lo": ("§2.1", 0.2397, 4),
    "betabinomial_0.60_fe_ci_hi": ("§2.1", 0.4873, 4),
    "betabinomial_0.70_fe_pp": ("§2.1", 0.2889, 4),
    "betabinomial_0.70_fe_ci_lo": ("§2.1", 0.1531, 4),
    "betabinomial_0.70_fe_ci_hi": ("§2.1", 0.4248, 4),
    "carrier_excess_single_pp": ("§20.17", 7.2219, 4),
    # Table 1's other two intervals (§20.4; the fixed effect's are §2.1's above)
    "betabinomial_0.60_across_seed_ci_lo": ("§20.4", 0.2590, 4),
    "betabinomial_0.60_across_seed_ci_hi": ("§20.4", 0.4698, 4),
    "betabinomial_0.60_jackknife_ci_lo": ("§20.4", 0.2252, 4),
    "betabinomial_0.60_jackknife_ci_hi": ("§20.4", 0.5018, 4),
    "betabinomial_0.70_across_seed_ci_lo": ("§20.4", 0.1560, 4),
    "betabinomial_0.70_across_seed_ci_hi": ("§20.4", 0.4237, 4),
    "betabinomial_0.70_jackknife_ci_lo": ("§20.4", 0.1222, 4),
    "betabinomial_0.70_jackknife_ci_hi": ("§20.4", 0.4557, 4),
}

# The nine-arm table (Appendix K), each cell from its NoR row. rate: mean r_f at
# 0.70 (six places, §19.1's rule, full values §20.27 (b) note, §20.29, §20.33);
# fvu / fvu_std: mean over the ten seeds (k64n9: seeds 0-9); slope: fixed effect
# and the widest interval, pp per SD of the arm's own log M (four places); sf:
# mean s_f over all cells (four places).
NINE_ARM = {
    "w2560": {"row": "§19.1, §19.3, §20.27 (b), §20.30 (c)", "rate": (0.601528, 6),
              "fvu": (0.461849, 6), "fvu_std": (0.480722, 6),
              "slope": (0.2524, -0.1542, 0.6591, "fixed_effect"), "sf": (0.7269, 4)},
    "w10240": {"row": "§19.1, §19.3, §20.27 (b), §20.30 (c)", "rate": (0.495794, 6),
               "fvu": (0.363886, 6), "fvu_std": (0.378671, 6),
               "slope": (0.2253, -0.0870, 0.5375, "jackknife"), "sf": (0.6845, 4)},
    "w20480": {"row": "§19.1, §19.3, §20.27 (b), §20.30 (c)", "rate": (0.426383, 6),
               "fvu": (0.321693, 6), "fvu_std": (0.334739, 6),
               "slope": (-0.9139, -1.1129, -0.7149, "jackknife"), "sf": (0.6556, 4)},
    "w40960": {"row": "§19.1, §19.3, §20.27 (b), §20.30 (c)", "rate": (0.336557, 6),
               "fvu": (0.284571, 6), "fvu_std": (0.295897, 6),
               "slope": (-1.6565, -1.7655, -1.5475, "jackknife"), "sf": (0.6171, 4)},
    "k64_w2560": {"row": "§20.29 (a), §20.33, §20.37, §20.30 (c)", "rate": (0.354479, 6),
                  "fvu": (0.41104, 5), "fvu_std": (0.42839, 5),
                  "slope": (0.5915, -0.1506, 1.3336, "jackknife"), "sf": (0.6030, 4)},
    "k64n9": {"row": "§20.33, §20.29 (b), §20.37", "rate": (0.215779, 6),
              "fvu": (0.32443, 5), "fvu_std": (0.33797, 5),
              "slope": (1.7016, 1.4288, 1.9745, "jackknife"), "sf": (0.5317, 4)},
    "k64_w40960": {"row": "§20.29 (a), §20.33, §20.37, §20.30 (c)", "rate": (0.111310, 6),
                   "fvu": (0.25591, 5), "fvu_std": (0.26625, 5),
                   "slope": (0.2998, 0.2362, 0.3634, "jackknife"), "sf": (0.4593, 4)},
    # no primary slope (§20.29, §20.32): the labelled top-merged sensitivity
    "k16_w10240": {"row": "§20.29 (b)-(c), §20.32, §20.37, §20.30 (c)", "rate": (0.660933, 6),
                   "fvu": (0.40242, 5), "fvu_std": (0.41712, 5),
                   "slope": (-0.1110, -0.4933, 0.2713, "jackknife"), "sf": (0.7546, 4),
                   "slope_label": "sensitivity; primary UNMEASURABLE (§CT5.1 cause 3)"},
    "k128_w10240": {"row": "§20.29 (b)-(c), §20.32, §20.37, §20.30 (c)", "rate": (0.086719, 6),
                    "fvu": (0.27570, 5), "fvu_std": (0.28754, 5),
                    "slope": (1.0315, 0.9320, 1.1310, "fixed_effect"), "sf": (0.3795, 4),
                    "slope_label": "sensitivity; primary UNMEASURABLE (§CT5.1 cause 3)"},
}
K64N9_SEEDS = [str(s) for s in range(10)]

# The two declared tiers (PREREGISTRATION.md §CT18-D4 item 3; the author, 2026-11-01).
# REFERENCE: the environment the record was reproduced in bit for bit (Bouchet jobs
# 31294938 and 31298208; env/reference_environment.yml). Every check is held to the
# record's rounding -- the primary criterion. PORTABLE: any other environment. The
# BLAS library and its version change the fitted values in the last digits (Windows,
# MKL: up to 5.27e-5 pp; docs/RELEASE.md §10), so fitted INTERVAL ENDPOINTS pass
# within PORTABLE_ENDPOINT_TOL_PP of the record; counted values and fixed effects
# are still held to the record's rounding.
REFERENCE_ENV = {"system": "Linux", "python": "3.11.15", "numpy": "2.4.6",
                 "scipy": "1.17.1", "h5py": "3.16.0", "blas": "openblas 0.3.33"}
PORTABLE_ENDPOINT_TOL_PP = 1e-3      # 19x the 5.27e-5 pp Windows-vs-Linux difference (the author: >= 10x)
ENDPOINT = re.compile(r"(_ci_(lo|hi)|_slope_widest_(lo|hi))$")


class BundleError(Exception):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_manifest(bundle: Path) -> list[dict]:
    """Every row's file exists and hashes as recorded; refuse otherwise."""
    man = bundle / "MANIFEST.csv"
    if not man.exists():
        raise BundleError(f"no manifest at {man}")
    rows = list(csv.DictReader(open(man, encoding="utf-8", newline="")))
    if not rows:
        raise BundleError("the manifest is empty")
    for r in rows:
        p = bundle / Path(r["path"]).name
        if not p.exists():
            raise BundleError(f"{r['path']} is in the manifest and absent")
        got = sha256_file(p)
        if got != r["sha256"]:
            raise BundleError(f"{r['path']}: sha256 {got} != manifest {r['sha256']}")
    return rows


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        return head, list(rd)


def download(base_url: str, bundle: Path, manifest: Path) -> list[str]:
    """Fetch every file the manifest names from `base_url`, verifying each
    against the manifest's sha256 before it is kept. A file already present
    with the right hash is not fetched again. Refuses on any mismatch."""
    import urllib.request
    rows = list(csv.DictReader(open(manifest, encoding="utf-8", newline="")))
    bundle.mkdir(parents=True, exist_ok=True)
    got = []
    for r in rows:
        name = Path(r["path"]).name
        dst = bundle / name
        if dst.exists() and sha256_file(dst) == r["sha256"]:
            continue
        tmp = dst.with_name(dst.name + ".part")
        url = base_url.rstrip("/") + "/" + name
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            for c in iter(lambda: resp.read(1 << 20), b""):
                f.write(c)
        h = sha256_file(tmp)
        if h != r["sha256"]:
            tmp.unlink()
            raise BundleError(f"{name} from {url}: sha256 {h} != manifest {r['sha256']}")
        tmp.replace(dst)
        got.append(name)
    return got


def check_value(name: str, row: str, rec: float, places: int, value) -> dict:
    """`check` for an expected value that is not in RECORD (the nine-arm table)."""
    if isinstance(rec, str):
        ok = value == rec
        return {"name": name, "nor_row": row, "record": rec, "recomputed": value,
                "verdict": "PASS" if ok else "FAIL"}
    ok = round(float(value), places) == round(rec, places)
    return {"name": name, "nor_row": row, "record": rec, "decimals": places,
            "recomputed": float(value), "recomputed_rounded": round(float(value), places),
            "verdict": "PASS" if ok else "FAIL"}


EXTENDED = (["sim_decoder_n39.csv.gz"]
            + [f"arm_{a}_{x}" for a in ("w2560", "w10240", "w20480", "w40960", "k64",
                                        "k64_w2560", "k64_w40960", "k16_w10240", "k128_w10240")
               for x in ("features.csv.gz", "sim_decoder.csv.gz", "fvu_per_seed.csv")])


def require_extended(bundle: Path) -> None:
    """The regeneration of Table 1's intervals and the nine-arm table needs the
    extended bundle (§CT18-F); a bundle without it is refused by name."""
    missing = [f for f in EXTENDED if not (bundle / f).exists()]
    if missing:
        raise BundleError(f"the extended bundle (§CT18-F) is incomplete: {len(missing)} file(s) "
                          f"absent, e.g. {missing[:3]}")


def table1_intervals(bundle: Path, b: dict, progress=None) -> list[dict]:
    """Table 1's across-seed and jackknife intervals at 0.60 and 0.70 (§20.4):
    the jackknife over the 40 dictionaries from `sim_decoder_n39`, through the
    record's own leave-one-dictionary-out functions, serially."""
    from src.analysis.release_derived_ext import read_sim, slope_block
    sim, anchors, cols = read_sim(bundle / "sim_decoder_n39.csv.gz", np.float32)
    if [int(a) for a in anchors] != b["anchors"]:
        raise BundleError("sim_decoder_n39's anchor order differs from the features'")
    log_phi = np.log10(np.maximum(b["fire_rate"], 1e-12))
    out = []
    for t in (0.60, 0.70):
        if progress is not None:
            progress.set(f"Table 1's intervals, theta = {t:.2f}: the full fit and one fit leaving out "
                         f"each of the {sim.shape[1]} dictionaries")
        blk = slope_block(sim, b["log_M"], log_phi, b["d_fs"], b["keep"], cols, theta=t)
        iv = blk["per_theta"]["intervals_95_pp"]
        for k, key in (("across_seed", "across_seed"), ("jackknife", "jackknife")):
            out.append(check(f"betabinomial_{t:.2f}_{key}_ci_lo", iv[k][0]))
            out.append(check(f"betabinomial_{t:.2f}_{key}_ci_hi", iv[k][1]))
    return out


def nine_arm(bundle: Path, progress=None) -> tuple[list[dict], list[dict]]:
    """The nine-arm table, regenerated from the bundle's arm files. Returns the
    checks and the regenerated rows."""
    from src.analysis.release_derived_ext import (k64n9_block, read_sim, read_table,
                                                  slope_block)
    checks, table = [], []
    for i_arm, (arm, rec) in enumerate(NINE_ARM.items(), 1):
        if progress is not None:
            progress.set(f"the nine-arm table, arm {i_arm} of {len(NINE_ARM)} ({arm}): the full fit and "
                         f"one fit leaving out each dictionary")
        src = "k64" if arm == "k64n9" else arm
        dt = {"log_M": np.float64, "fire_rate": np.float64, "d_fs": np.float64,
              "keep_a4a": bool, "s_f": np.float32}
        if arm == "k64n9":
            dt["s_f_use_0to9"] = np.float32
        f, anchors = read_table(bundle / f"arm_{src}_features.csv.gz", dt)
        head, rows = read_csv(bundle / f"arm_{src}_features.csv.gz")
        k70 = np.array([int(r[head.index("k_0.70")]) for r in rows]).reshape(len(anchors), -1)
        sim, sa, cols = read_sim(bundle / f"arm_{src}_sim_decoder.csv.gz", np.float32)
        if sa != anchors:
            raise BundleError(f"arm {src}: similarity and feature anchor orders differ")
        _, frows = read_csv(bundle / f"arm_{src}_fvu_per_seed.csv")
        fvu = {r[0]: (float(r[1]), float(r[2])) for r in frows}
        lp = np.log10(np.maximum(f["fire_rate"], 1e-12))
        if arm == "k64n9":
            idx = [anchors.index(s) for s in K64N9_SEEDS]
            blk = k64n9_block(sim, f["log_M"], lp, f["d_fs"], f["keep_a4a"], anchors, cols,
                              K64N9_SEEDS)
            own = blk["own"]
            iv = {"fixed_effect": own["fe_ci95_pp"], "across_seed": own["across_seed_ci95_pp"],
                  "jackknife": own["jackknife_ci95_pp"]}
            fe, widest = own["fe_pp"], own["widest"]
            rate = blk["mean_r_f"]
            sf = float(np.stack([f["s_f_use_0to9"][i] for i in idx]).mean())
            seeds = K64N9_SEEDS
        else:
            blk = slope_block(sim, f["log_M"], lp, f["d_fs"], f["keep_a4a"], cols)
            iv = blk["per_theta"]["intervals_95_pp"]
            fe, widest = blk["full_fit"]["fe_pp"], blk["per_theta"]["widest"]
            rate = float((k70.astype(np.float64) / (sim.shape[1] - 1)).ravel().mean())
            sf = float(f["s_f"].mean())
            seeds = sorted(fvu, key=int)
        fr = float(np.mean([fvu[s][0] for s in seeds]))
        fs = float(np.mean([fvu[s][1] for s in seeds]))
        r = rec["row"]
        checks += [check_value(f"{arm}_rate_0.70", r, rec["rate"][0], rec["rate"][1], rate),
                   check_value(f"{arm}_fvu_raw", r, rec["fvu"][0], rec["fvu"][1], fr),
                   check_value(f"{arm}_fvu_std", r, rec["fvu_std"][0], rec["fvu_std"][1], fs),
                   check_value(f"{arm}_slope_fe", r, rec["slope"][0], 4, fe),
                   check_value(f"{arm}_slope_widest_name", r, rec["slope"][3], 0, widest),
                   check_value(f"{arm}_slope_widest_lo", r, rec["slope"][1], 4, iv[widest][0]),
                   check_value(f"{arm}_slope_widest_hi", r, rec["slope"][2], 4, iv[widest][1]),
                   check_value(f"{arm}_mean_sf", r, rec["sf"][0], rec["sf"][1], sf)]
        table.append({"arm": arm, "rate_0.70": rate, "fvu_raw": fr, "fvu_std": fs,
                      "slope_fe_pp": fe, "slope_widest": widest,
                      "slope_widest_interval": iv[widest], "mean_sf": sf,
                      "slope_label": rec.get("slope_label", "primary")})
    return checks, table


def check(name: str, value: float) -> dict:
    row, rec, places = RECORD[name]
    ok = round(float(value), places) == round(rec, places)
    return {"name": name, "nor_row": row, "record": rec, "decimals": places,
            "recomputed": float(value), "recomputed_rounded": round(float(value), places),
            "verdict": "PASS" if ok else "FAIL"}


def carrier_excess(order: list[int], carrier: dict, detected: dict, has: dict,
                   fbin: dict) -> dict:
    """The single-arm carrier excess (AY7's event, BB's single arm), in plain
    numpy. `carrier[s]`, `detected[s]`: per pool concept; `has[(s, t)]`: per
    feature of s, an MNN partner in t; `fbin[s]`: per feature of s.

    Unordered pairs `s < t` in `order`; for each concept both seeds detect, A is
    "s's carrier has a partner in t", and its matched ambient is the rate of the
    same event among s's OTHER features in the carrier's firing bin (cells with
    fewer than two features are not usable). Excess = (A - matched) * 100.
    """
    a_hits = n_eval = 0
    matched_sum, matched_n = 0.0, 0
    for i, s in enumerate(order):
        n_in_bin = np.bincount(fbin[s])
        for t in order[i + 1:]:
            both = detected[s] & detected[t]
            cs = carrier[s][both]
            if cs.size == 0:
                continue
            h = has[(s, t)]
            a = h[cs]
            a_hits += int(a.sum())
            n_eval += int(cs.size)
            hit_in_bin = np.bincount(fbin[s], weights=h.astype(np.float64),
                                     minlength=n_in_bin.size).astype(np.int64)
            cc = fbin[s][cs]
            n_in, hit_in = n_in_bin[cc], hit_in_bin[cc]
            ok = n_in > 1
            if ok.any():
                num = hit_in[ok] - a[ok].astype(np.int64)
                matched_sum += float((num / (n_in[ok] - 1)).sum())
                matched_n += int(ok.sum())
    A = a_hits / n_eval
    matched = matched_sum / matched_n
    return {"A": A, "matched_ambient": matched, "n_cells": n_eval,
            "excess_pp": (A - matched) * 100.0}


def load_bundle(bundle: Path) -> dict:
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    head, rows = read_csv(bundle / "features_n39.csv.gz")
    anchors = [int(a) for a in meta["anchor_seed_order_features"]]
    n_a = len(anchors)
    n_f = len(rows) // n_a
    if n_a * n_f != len(rows):
        raise BundleError("feature table is not anchors x features")
    col = {h: i for i, h in enumerate(head)}

    def arr(name, dtype):
        return np.array([r[col[name]] for r in rows], dtype=dtype).reshape(n_a, n_f)
    seeds = arr("anchor_seed", np.int64)
    if not np.all(seeds == np.array(anchors)[:, None]):
        raise BundleError("feature rows are not in meta's anchor order")
    f = {"anchors": anchors, "n_features": n_f, "meta": meta,
         "k": {t: arr(f"k_{t:.2f}", np.int64) for t in meta["theta_grid"]},
         "log_M": arr("log_M", np.float64), "fire_rate": arr("fire_rate", np.float64),
         "fire_count_ap": arr("fire_count_ap", np.int64),
         "d_fs": arr("d_fs", np.float64), "keep": arr("keep_a4a", np.int64) == 1}

    ih, irows = read_csv(bundle / "mnn_indicator_theta070.csv.gz")
    cols = [int(h[2:]) for h in ih[2:]]
    ind = np.zeros((n_a, len(cols), n_f), dtype=bool)
    for r in irows:
        a, ft = anchors.index(int(r[0])), int(r[1])
        ind[a, :, ft] = [v == "1" for v in r[2:]]
    f["indicator"], f["columns"] = ind, cols

    ch, crows = read_csv(bundle / "concepts_carriers.csv.gz")
    cc = {h: i for i, h in enumerate(ch)}
    carrier, detected = {}, {}
    for r in crows:
        s = int(r[cc["anchor_seed"]])
        carrier.setdefault(s, []).append(int(r[cc["carrier_feature"]]))
        detected.setdefault(s, []).append(r[cc["detected"]] == "1")
    f["carrier"] = {s: np.array(v, dtype=np.int64) for s, v in carrier.items()}
    f["detected"] = {s: np.array(v, dtype=bool) for s, v in detected.items()}
    return f


@contextlib.contextmanager
def single_threaded():
    """Every BLAS / OpenMP pool limited to one thread for the block; yields the
    pools as they run inside it (an empty list if threadpoolctl is absent, when
    only the environment variables set at import apply)."""
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        yield []
        return
    with threadpool_limits(limits=1):
        yield [{k: p.get(k) for k in ("user_api", "internal_api", "version", "num_threads")}
               for p in threadpool_info()]


def environment(pools: list) -> dict:
    import platform
    import scipy
    import h5py
    blas = sorted({f"{p.get('internal_api')} {p.get('version')}" for p in pools
                   if p.get("user_api") == "blas"})
    return {"system": platform.system(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__,
            "blas": blas, "threads": sorted({p.get("num_threads") for p in pools})}


def tier(env: dict) -> tuple[str, list[str]]:
    """REFERENCE only if every pinned fact matches and every pool ran one thread;
    otherwise PORTABLE, with each fact that differed."""
    why = [f"{k} {env.get(k)!r} != {v!r}" for k, v in REFERENCE_ENV.items()
           if k != "blas" and env.get(k) != v]
    if env.get("blas") != [REFERENCE_ENV["blas"]]:
        why.append(f"blas {env.get('blas')!r} != [{REFERENCE_ENV['blas']!r}]")
    if env.get("threads") != [1]:
        why.append(f"BLAS threads {env.get('threads')!r} != [1]")
    return ("REFERENCE" if not why else "PORTABLE"), why


def apply_tier(checks: list[dict], tier_name: str) -> int:
    """In PORTABLE, a FAILED fitted interval endpoint within the stated tolerance
    of the record becomes PASS_BY_TOLERANCE; nothing else changes, and nothing
    changes in REFERENCE. Returns the number of checks still failing."""
    for c in checks:
        if (tier_name == "PORTABLE" and c["verdict"] == "FAIL" and ENDPOINT.search(c["name"])
                and not isinstance(c["record"], str)
                and abs(float(c["recomputed"]) - float(c["record"])) <= PORTABLE_ENDPOINT_TOL_PP):
            c["verdict"] = "PASS_BY_TOLERANCE"
            c["tolerance_pp"] = PORTABLE_ENDPOINT_TOL_PP
    return sum(c["verdict"] not in ("PASS", "PASS_BY_TOLERANCE") for c in checks)


def import_torch_first() -> str | None:
    """Import PyTorch before any thread pool is inspected or limited; returns its
    version, or None if it is not installed (the fit's import then fails with
    its own message). The order matters on Windows only -- see the docstring --
    and is the same on every platform so that one path is the tested one."""
    try:
        import torch
    except ImportError:
        return None
    return torch.__version__


class Progress:
    """One line per design-3 fit on standard error: the stage, the fit's number
    within the run and the time elapsed. It wraps `design3_join.fit_arm` for the
    block and puts the original back; arguments and results pass through
    untouched, so no computed value depends on it."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stderr
        self.stage, self.n, self.t0 = "", 0, 0.0

    def set(self, stage: str) -> None:
        self.stage = stage
        self._say(f"stage: {stage}")

    def _say(self, msg: str) -> None:
        import time
        e = int(time.monotonic() - self.t0)
        print(f"[reproduce_table1 {e // 60:>3d}:{e % 60:02d}] {msg}", file=self.stream, flush=True)

    def __enter__(self):
        import time
        import src.analysis.design3_join as d3
        self._d3, self._orig, self.t0 = d3, d3.fit_arm, time.monotonic()

        def fit_arm(*a, **k):
            self.n += 1
            self._say(f"fit {self.n} ({self.stage})")
            return self._orig(*a, **k)
        d3.fit_arm = fit_arm
        return self

    def __exit__(self, *exc):
        self._d3.fit_arm = self._orig
        self._say(f"done: {self.n} fits")
        return False


def run(bundle: Path) -> dict:
    torch_version = import_torch_first()
    with single_threaded() as pools:
        with Progress() as progress:
            out = _run(bundle, progress)
    out["torch"] = torch_version
    out["blas_pools"] = pools
    out["environment"] = environment(pools)
    out["tier"], out["tier_reasons"] = tier(out["environment"])
    out["portable_endpoint_tolerance_pp"] = PORTABLE_ENDPOINT_TOL_PP
    out["n_fail"] = apply_tier(out["checks"], out["tier"])
    return out


def _run(bundle: Path, progress: Progress | None = None) -> dict:
    progress = progress if progress is not None else Progress(io.StringIO())
    progress.set("verifying the bundle against its manifest, and reading it")
    manifest = verify_manifest(bundle)
    require_extended(bundle)
    b = load_bundle(bundle)
    meta, n = b["meta"], b["meta"]["n_compare"]
    out = {"bundle": str(bundle), "manifest_rows": len(manifest), "checks": []}
    C = out["checks"]
    C.append(check("headline_mean_rf_0.70_n39", b["k"][0.70].mean() / n))
    C.append(check("mean_rf_0.30_n39", b["k"][0.30].mean() / n))
    C.append(check("mean_rf_0.90_n39", b["k"][0.90].mean() / n))
    a_cols = [b["columns"].index(a) for a in b["anchors"]]
    k19 = b["indicator"][:, a_cols, :].sum(axis=1)
    C.append(check("mean_rf_0.70_n19", k19.mean() / (len(b["anchors"]) - 1)))

    from src.analysis.design3_join import fit_arm
    log_phi = np.log10(np.maximum(b["fire_rate"], 1e-12))
    fits = {}
    for t in (0.55, 0.60, 0.70):
        progress.set(f"Table 1, theta = {t:.2f} (1 fit)")
        arm = fit_arm(b["k"][t].astype(np.float64), b["log_M"], log_phi, b["d_fs"],
                      b["keep"], n, dispersion=True)
        fits[t] = {"binomial": arm["meta_marginal_effect"],
                   "betabinomial": arm["meta_marginal_effect_betabinomial"]}
    for lik, t in (("binomial", 0.55), ("betabinomial", 0.60), ("betabinomial", 0.70)):
        m = fits[t][lik]
        C.append(check(f"{lik}_{t:.2f}_fe_pp", 100.0 * m["fixed_effect"]))
        C.append(check(f"{lik}_{t:.2f}_fe_ci_lo", 100.0 * m["fixed_effect_ci95"][0]))
        C.append(check(f"{lik}_{t:.2f}_fe_ci_hi", 100.0 * m["fixed_effect_ci95"][1]))

    order = meta["carrier_pair_order_lexicographic"]
    edges = np.asarray(meta["firing_bin_edges_carrier"], dtype=np.float64)
    a_idx = {s: b["anchors"].index(s) for s in order}
    fbin = {s: np.clip(np.digitize(b["fire_count_ap"][a_idx[s]] / float(meta["n_rows_ap"]),
                                   edges[1:-1], right=False), 0, len(edges) - 2)
            for s in order}
    has = {(s, t): b["indicator"][a_idx[s], b["columns"].index(t)]
           for s in order for t in order if s != t}
    ex = carrier_excess(order, b["carrier"], b["detected"], has, fbin)
    out["carrier"] = ex
    C.append(check("carrier_excess_single_pp", ex["excess_pp"]))
    C.extend(table1_intervals(bundle, b, progress))
    nine, table = nine_arm(bundle, progress)
    C.extend(nine)
    out["nine_arm_table"] = table
    out["n_fail"] = sum(c["verdict"] != "PASS" for c in C)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", type=Path, default=REPO / "release" / "derived")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--download", default=None, metavar="BASE_URL",
                    help="fetch the manifest's files from BASE_URL first, each verified "
                         "against this repository's manifest")
    ap.add_argument("--manifest", type=Path, default=REPO / "release" / "derived" / "MANIFEST.csv",
                    help="the manifest the download is verified against (default: the "
                         "one in this repository)")
    a = ap.parse_args(argv)
    try:
        if a.download is not None:
            got = download(a.download, a.bundle, a.manifest)
            man = a.bundle / "MANIFEST.csv"
            if man.resolve() != a.manifest.resolve():
                man.write_bytes(a.manifest.read_bytes())
            print(f"downloaded {len(got)} file(s); every manifest hash verified")
        out = run(a.bundle)
    except BundleError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"TIER {out['tier']}" + (" (" + "; ".join(out["tier_reasons"]) + ")"
                                       if out["tier_reasons"] else ""))
    for c in out["checks"]:
        print(f"{c['verdict']}  {c['name']:<34} {c['nor_row']:<18} record "
              f"{c['record']!s:<12} recomputed {c['recomputed']!r}")
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    n_tol = sum(c["verdict"] == "PASS_BY_TOLERANCE" for c in out["checks"])
    print(f"{len(out['checks']) - out['n_fail']} of {len(out['checks'])} checks pass "
          f"(tier {out['tier']}" + (f"; {n_tol} fitted interval endpoint(s) only within "
                                     f"{PORTABLE_ENDPOINT_TOL_PP} pp" if n_tol else "") + ")")
    return 0 if out["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
