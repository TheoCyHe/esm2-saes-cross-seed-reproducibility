"""§CT27(b): rule 3 with the encoder similarity taken as the RAW dot product of
unnormalised encoder rows -- the per-comparator similarity, persisted.

Under the author's decision 1 (2026-10-23) the decoder half is unchanged: decoder
cosine, its own Hungarian assignment (rule 2's solve), threshold 0.70. The
encoder Hungarian is solved on the raw product, and a feature of anchor a has
a partner in comparator b when the two assignments agree AND its decoder
cosine is >= 0.70. The persisted similarity is then that feature's RAW encoder
product (-inf otherwise), so `sim >= theta_e` IS the variant's indicator and
§CT4's theta* rule rate-matches theta_e (`ct_robustness ct27`).

CONTROL, before anything is written: the same loop with the encoder rows
normalised rebuilds rule 3's similarity, which must equal
`ct20_rules_sim_ct.h5`'s `sim_rule3_hungarian_enc_and_dec` EXACTLY in every
(anchor, comparator, feature) cell (§CT27-F item 2).
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np

from src.analysis.rule_interior_mass import (DEFAULT_ANCHORS, load_checkpoints,
                                             pool_geometry)
from src.analysis.rule_sensitivity import (NEG, encoder_decoder_from_assignments,
                                           hungarian_assignment,
                                           partner_view_from_b)
from src.analysis.seed_axis import align_to_reference

THETA_DEC = 0.70
RULE3_DS = "sim_rule3_hungarian_enc_and_dec"


def variant_from_assignments(dec_partner, dec_sim, enc_partner, enc_sim,
                             theta_dec: float = THETA_DEC):
    """(partner, similarity): where the assignments agree and the decoder cosine
    is >= theta_dec, the partner and its RAW encoder product; -1 / -inf
    elsewhere."""
    keep = (dec_partner == enc_partner) & (dec_partner >= 0) & (dec_sim >= theta_dec)
    return np.where(keep, dec_partner, -1), np.where(keep, enc_sim, NEG)


def run(ckpt_root: Path, anchors: list[int], workers_note: str = "") -> dict:
    n_anchors, _ = pool_geometry(anchors)
    D, En, _c1 = load_checkpoints(ckpt_root, anchors)
    _D2, Er, _ = load_checkpoints(ckpt_root, anchors, normalise_encoder=False)
    n = D[anchors[0]].shape[1]
    pos = {s: i for i, s in enumerate(anchors)}
    var = np.full((n_anchors, n_anchors, n), NEG, dtype=np.float64)
    r3 = np.full((n_anchors, n_anchors, n), NEG, dtype=np.float64)
    per_pair = []
    t0 = time.time()
    for idx, (a, b) in enumerate(itertools.combinations(anchors, 2)):
        Cd = (D[a].T @ D[b]).numpy().astype(np.float64)
        p2, s2a = hungarian_assignment(Cd)
        # control: rule 3 as §CT20 computed it (normalised encoder rows)
        Cen = (En[a] @ En[b].T).numpy().astype(np.float64)
        pen, sen = hungarian_assignment(Cen)
        p3, s3a = encoder_decoder_from_assignments(p2, s2a, pen, sen)
        r3[pos[a], pos[b]] = s3a
        r3[pos[b], pos[a]] = partner_view_from_b(p3, s3a, n)
        # the variant: raw encoder rows
        Cer = (Er[a] @ Er[b].T).numpy().astype(np.float64)
        per, ser = hungarian_assignment(Cer)
        pv, sva = variant_from_assignments(p2, s2a, per, ser)
        var[pos[a], pos[b]] = sva
        var[pos[b], pos[a]] = partner_view_from_b(pv, sva, n)
        per_pair.append({"pair": [a, b],
                         "agree_frac_normalised": float(((p2 == pen) & (p2 >= 0)).mean()),
                         "agree_frac_raw": float(((p2 == per) & (p2 >= 0)).mean()),
                         "raw_encoder_assignment_equals_normalised":
                             float((per == pen).mean()),
                         "raw_product_max_assigned": float(ser.max()),
                         "raw_product_min_assigned": float(ser.min())})
        if idx % 20 == 0:
            print(f"  pair {idx + 1}  elapsed {time.time() - t0:.0f}s", flush=True)
    return {"var": var, "r3": r3, "per_pair": per_pair, "n_features": n,
            "elapsed_s": time.time() - t0}


def control_vs_stored(r3: np.ndarray, anchors: list[int], stored_h5: Path) -> dict:
    with h5py.File(stored_h5, "r") as hf:
        labs = [str(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
        s = hf[RULE3_DS][:]
    want = [str(a) for a in anchors]
    s = align_to_reference(labs, s, want)
    s = np.stack([align_to_reference(labs, s[i], want) for i in range(s.shape[0])])
    same = (r3 == s) | (np.isneginf(r3) & np.isneginf(s))
    return {"cells": int(same.size), "cells_differing": int((~same).sum()),
            "verdict": "EXACT" if bool(same.all()) else "DIFFERS"}


def main(argv=None) -> None:
    from src.sae.train import git_dirty, git_sha
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-root", type=Path, required=True)
    p.add_argument("--anchors", type=int, nargs="+", default=DEFAULT_ANCHORS)
    p.add_argument("--stored-rules-sim-h5", type=Path, required=True,
                   help="ct20_rules_sim_ct.h5, the control's reference")
    p.add_argument("--out-h5", type=Path, required=True)
    a = p.parse_args(argv)
    if a.out_h5.exists():
        raise SystemExit(f"FATAL: {a.out_h5} exists")
    anchors = [int(x) for x in a.anchors]
    res = run(a.ckpt_root, anchors)
    ctrl = control_vs_stored(res["r3"], anchors, a.stored_rules_sim_h5)
    print(f"[ct27] control vs stored rule 3: {ctrl}", flush=True)
    if ctrl["verdict"] != "EXACT":
        raise SystemExit(f"CONTROL FAILED: {ctrl}")
    a.out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out_h5, "w") as hf:
        hf.create_dataset("sim_ct27_raw_encoder_dec070", data=res["var"],
                          compression="gzip")
        hf.attrs["anchor_seed_labels"] = json.dumps([str(x) for x in anchors])
        hf.attrs["theta_decoder"] = THETA_DEC
        hf.attrs["definition"] = (
            "per (anchor, comparator, feature): the RAW encoder product of the "
            "partner where rule 2's decoder Hungarian assignment and the raw-"
            "encoder Hungarian assignment agree and the decoder cosine >= 0.70; "
            "-inf otherwise (amendment CT, §CT27(b), the author's decision 1)")
        hf.attrs["control_rule3_vs_stored"] = json.dumps(ctrl)
        hf.attrs["per_pair"] = json.dumps(res["per_pair"])
        hf.attrs["git_sha"] = git_sha()
        hf.attrs["git_dirty"] = json.dumps(git_dirty())
        hf.attrs["device_class"] = "cpu"
        hf.attrs["host"] = platform.node()
        hf.attrs["n_compare_seeds"] = len(anchors) - 1
        hf.attrs["elapsed_s"] = res["elapsed_s"]
    print(f"[ct27] wrote {a.out_h5}", flush=True)


if __name__ == "__main__":
    main()
