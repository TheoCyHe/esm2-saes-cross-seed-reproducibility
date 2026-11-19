"""The pooled raw null's seed order, recorded as a SIDECAR (amendment BS, part C3).

WHY A SIDECAR AND NOT AN ATTR ON THE ARTIFACT. `null_gain_raw_compliant.h5`
carries no `seeds` dataset, so its seed axis has to be recovered rather than
read. Writing the recovered order back into the artifact would make INFERRED
provenance indistinguishable from READ provenance -- the exact failure the
no-back-fill rule on device provenance exists to prevent (gotcha 15, and
`artifact_identity`'s `ABSENT_ON_BOTH` convention). **Nothing is back-filled.**
The sidecar states the mapping, the method, and the conditions under which the
method refuses, and it labels itself as recovered.

**THE LABEL IS "RECOVERED FROM THE REDUCTIONS' DATA", NOT "FROM FILENAMES", AND
THE DIFFERENCE IS NOT COSMETIC.** BS's brief described the recovery as being
"from a naming convention". It is not, and never was: `recover_seed_position`
(`classification_2x2.py`) matches the slab's max-over-concepts against
`null_pilot_compliant.h5`'s `null_g_*` by EXACT ARRAY EQUALITY, and the
reductions' own `derived_from_raw` attr names this raw file. No filename is
parsed anywhere in the path. Recording it as a filename convention would
understate the evidence -- an exact equality on 20 x 10240 values is
far stronger than a name match -- and would misdescribe what the two refusal
conditions mean. Corrections-log entry for BS.

THE REFUSAL CONDITIONS, which are the reason this is evidence at all: zero
matches means the reductions do not describe this array, and more than one
means the recovery cannot discriminate. Both raise `SeedOrderUnrecoverable`;
neither is papered over with a nearest match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .classification_2x2 import SeedOrderUnrecoverable, recover_seed_position

__all__ = ["recover_all", "build_sidecar"]

#: What this sidecar claims about the mapping's provenance. Stated as a
#: constant so the label cannot drift from the method in prose.
PROVENANCE = ("RECOVERED FROM THE REDUCTIONS' DATA BY EXACT MAX-OVER-CONCEPTS "
              "EQUALITY -- NOT READ FROM THE ARTIFACT, AND NOT FROM FILENAMES")


def _slab_max_over_concepts(dset, s: int) -> np.ndarray:
    """max over the CONCEPT axis of slab `s`, read one shift at a time.

    `dset` is (n_seeds, n_shifts, n_concepts, n_features). Reading a whole slab
    is 575 MB; a single shift is 29 MB, and the max is associative over the
    concept axis, so the per-shift read gives the identical result at a
    twentieth of the resident memory.

    THE RESULT IS IN THE RAW ARRAY'S STORED DTYPE (amendment CT, the same fix
    `classification_2x2.recover_raw_slab_seed` got at `d762121`). A max is
    exact in any dtype that holds its inputs, so taking it in float32 loses
    nothing; `recover_all` then compares in that dtype. Until CT this was a
    float64 accumulator compared in float64, which is correct only while
    `null_g_*` is float32-representable (every pooled null of record).
    """
    n_shifts = dset.shape[1]
    out = np.empty((n_shifts, dset.shape[3]), dtype=dset.dtype)
    for t in range(n_shifts):
        out[t] = np.asarray(dset[s, t]).max(axis=0)
    return out


def recover_all(raw_path: Path, reductions_path: Path) -> dict:
    """Recover the seed order independently on BOTH arms and check agreement.

    The two arms are separate arrays with separate null draws, so agreement
    between them is a genuine independent check and not a restatement -- if the
    recovery were matching on something incidental, the arms would not have to
    agree.
    """
    with h5py.File(raw_path, "r") as raw, h5py.File(reductions_path, "r") as red:
        if "seeds" in raw:
            raise RuntimeError(
                "REFUSING: the pooled raw null now HAS a `seeds` dataset. The "
                "recovery exists because it did not; if labels have been "
                "added, read them and delete this module rather than "
                "recovering an order that is now stored.")
        seed_labels = [s.decode() if isinstance(s, bytes) else str(s)
                       for s in red["seeds"][:]]
        n_seeds = raw["gain_a"].shape[0]
        if n_seeds != len(seed_labels):
            raise RuntimeError(
                f"seed-axis length disagreement: raw has {n_seeds} slabs, "
                f"reductions label {len(seed_labels)} seeds")

        per_arm: dict[str, list[int]] = {}
        for arm in ("a", "b"):
            # Compared IN THE RAW ARRAY'S STORED PRECISION. On a segmented-path
            # null `null_g` keeps the float64 max while the raw array keeps its
            # float32 cast, and a float64 comparison matches almost no cell.
            # Rounding to nearest is monotone, so max(f32(x)) == f32(max(x))
            # EXACTLY: this is still an identity, not a tolerance, and for a
            # float32-representable `null_g` the cast is lossless.
            stored = raw[f"gain_{arm}"].dtype
            g = np.asarray(red[f"null_g_{arm}"][:])
            if stored != np.float64:
                g = g.astype(stored)
            positions = []
            for s in range(n_seeds):
                m = _slab_max_over_concepts(raw[f"gain_{arm}"], s)
                positions.append(recover_seed_position(m, g))
            per_arm[arm] = positions

    agree = per_arm["a"] == per_arm["b"]
    pos_a = per_arm["a"]
    is_perm = sorted(pos_a) == list(range(n_seeds))
    return {
        "n_slabs": n_seeds,
        "positions_arm_a": pos_a,
        "positions_arm_b": per_arm["b"],
        "arms_agree": agree,
        "is_permutation": is_perm,
        "slab_to_seed_label": (
            {str(s): seed_labels[pos_a[s]] for s in range(n_seeds)}
            if agree and is_perm else None),
        "identity_ordering": (pos_a == list(range(n_seeds))) if is_perm else None,
    }


def build_sidecar(raw_path: Path, reductions_path: Path, git_sha: str,
                  git_dirty: bool) -> dict:
    rec = recover_all(raw_path, reductions_path)
    if not rec["arms_agree"]:
        raise SeedOrderUnrecoverable(
            f"the two arms recover DIFFERENT orders: a={rec['positions_arm_a']} "
            f"b={rec['positions_arm_b']}. The recovery is not trusted when its "
            "own independent check disagrees.")
    if not rec["is_permutation"]:
        raise SeedOrderUnrecoverable(
            f"the recovered positions {rec['positions_arm_a']} are not a "
            "permutation of range(n_slabs): two slabs matched the same "
            "labelled seed, so the mapping is not a bijection.")
    return {
        "identity": {
            "measurement": "seed_order_pooled_raw_null",
            "amendments": ["BS-C3"],
            "provenance": PROVENANCE,
            "raw_path": str(raw_path),
            "reductions_path": str(reductions_path),
            "method": (
                "For each slab s and each arm, max over the CONCEPT axis gives "
                "(n_shifts, n_features); this is matched by EXACT array "
                "equality, in the raw array's stored dtype, against each row "
                "of the reductions' null_g_<arm>, "
                "which is the max over concepts of this very array (the "
                "reductions' own `derived_from_raw` attr names this file). "
                "Exactly one row must match."),
            "refusal_conditions": [
                "ZERO matches -- the reductions do not describe this array; "
                "the seed axis cannot be recovered and must not be assumed.",
                "MORE THAN ONE match -- the recovery cannot discriminate "
                "between the candidates and is not trusted to.",
                "THE TWO ARMS DISAGREE -- the independent check failed.",
                "THE POSITIONS ARE NOT A PERMUTATION -- the mapping is not a "
                "bijection.",
                "THE RAW ARTIFACT GAINS A `seeds` DATASET -- the recovery is "
                "then obsolete and refuses rather than shadowing stored "
                "labels.",
            ],
            "not_written_into_the_artifact": (
                "DELIBERATE. Writing the recovered order into "
                "null_gain_raw_compliant.h5 would make inferred provenance "
                "indistinguishable from read provenance. Nothing is "
                "back-filled."),
            "git_sha": git_sha,
            "git_dirty": git_dirty,
        },
        "recovery": rec,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--reductions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--git-sha", default="UNKNOWN")
    ap.add_argument("--git-dirty", action="store_true")
    args = ap.parse_args()

    doc = build_sidecar(args.raw, args.reductions, args.git_sha, args.git_dirty)
    r = doc["recovery"]
    print(f"[sidecar] slabs={r['n_slabs']} arms_agree={r['arms_agree']} "
          f"is_permutation={r['is_permutation']} "
          f"identity_ordering={r['identity_ordering']}")
    print(f"[sidecar] positions (arm a): {r['positions_arm_a']}")
    print(f"[sidecar] positions (arm b): {r['positions_arm_b']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"[sidecar] wrote {args.out}")


if __name__ == "__main__":
    main()
