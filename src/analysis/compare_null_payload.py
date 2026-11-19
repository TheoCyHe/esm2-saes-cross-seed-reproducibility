"""
AC1 -- the pooled arm's regression test, on the RAW BIT PATTERN.

WHY BITS AND NOT `allclose`. Two float arrays can be `allclose` and not equal,
and `==` gets two cases wrong in opposite directions: NaN != NaN (so a
NaN-preserving refactor reads as a difference) and -0.0 == +0.0 (so a sign
change reads as agreement). Viewing each array as an unsigned integer of
matching width settles both -- NaN patterns compare equal to themselves, and
the two zeros compare unequal.

WHY BOTH FILES. The reductions are MAXIMA, and a maximum cannot detect changes
beneath it: a refactor could alter many per-(concept, feature) values and leave
`null_g_*` and `null_cmax_*` bit-identical. The raw file is the sensitive
channel. One pooled run writes both, so comparing both costs one script rather
than one extra run.

IT REPORTS THE DIFF, NOT A VERDICT. Count of differing elements, their
magnitude, and which (seed, shift) slices carry them -- a printed conclusion
has been wrong here while the computation was right (gotcha 20), and a
completed measurement has been reported as FAILED because a summary print died
on a legitimately undefined value (BA17), so every formatter below is total.

Attributes are compared and REPORTED SEPARATELY, never folded into the
verdict: `git_sha` differs by construction on any re-run, and a bit-identity
criterion applied to a provenance key can only ever fail (gotcha 25a).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.sae.train import git_dirty, git_sha

# Keys that differ by construction on a re-run. Exempted from the ATTRIBUTE
# report's "unexpected" list by FULL NAME -- gotcha 25a: exempting by leaf
# name would also exempt keys that must stay compared.
PROVENANCE_ATTRS = ("git_sha", "git_dirty", "labels", "raw_gain_path",
                    "reductions_path", "created_utc", "note",
                    "attr_schema_version")


def bit_view(a: np.ndarray) -> np.ndarray:
    """Raw bit pattern as an unsigned integer of matching width."""
    if a.dtype.kind == "f":
        return a.view({4: np.uint32, 8: np.uint64}[a.dtype.itemsize])
    return a


def compare_array(name: str, got: np.ndarray, ref: np.ndarray) -> dict:
    """Elementwise bit comparison plus the magnitude of any disagreement."""
    if got.shape != ref.shape:
        return {"dataset": name, "status": "SHAPE_DIFFERS",
                "got_shape": list(got.shape), "ref_shape": list(ref.shape)}
    diff = bit_view(got) != bit_view(ref)
    n_diff = int(diff.sum())
    out = {"dataset": name, "status": "BIT_IDENTICAL" if not n_diff else "DIFFERS",
           "n_elements": int(got.size), "n_differing": n_diff,
           "dtype": str(got.dtype), "shape": list(got.shape)}
    if not n_diff:
        return out
    if got.dtype.kind == "f":
        g, r = got[diff].astype(np.float64), ref[diff].astype(np.float64)
        finite = np.isfinite(g) & np.isfinite(r)
        out["max_abs_difference"] = (float(np.abs(g[finite] - r[finite]).max())
                                     if finite.any() else None)
        out["n_nonfinite_involved"] = int((~finite).sum())
    # Which (seed, shift) slices, when the leading axes are those.
    if got.ndim >= 2:
        lead = diff.reshape(got.shape[0], got.shape[1], -1).any(axis=2)
        idx = np.argwhere(lead)
        out["n_leading_slices_affected"] = int(lead.sum())
        out["leading_slices_affected"] = [
            [int(i), int(j)] for i, j in idx[:40]]
        if len(idx) > 40:
            out["leading_slices_truncated_after"] = 40
    return out


def compare_attrs(got: h5py.File, ref: h5py.File) -> dict:
    def norm(v):
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    g = {k: norm(v) for k, v in got.attrs.items()}
    r = {k: norm(v) for k, v in ref.attrs.items()}
    differing, only_new, only_ref = {}, sorted(set(g) - set(r)), sorted(set(r) - set(g))
    for k in sorted(set(g) & set(r)):
        if g[k] != r[k]:
            differing[k] = {"reference": r[k], "now": g[k]}
    unexpected = sorted(k for k in differing if k not in PROVENANCE_ATTRS)
    return {"n_differing": len(differing), "differing": differing,
            "only_in_new": only_new, "only_in_reference": only_ref,
            "differing_beyond_provenance": unexpected,
            "note": "Attribute differences are REPORTED, never folded into the "
                    "payload verdict. git_sha differs by construction."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Payload datasets to compare.")
    p.add_argument("--seed-axis-datasets", nargs="*", default=[],
                   help="Datasets whose FIRST axis is the seed axis; the "
                        "reference is sliced to the new file's seed count so a "
                        "declared subset can be compared against a full run.")
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    results, missing = [], []
    with h5py.File(args.new, "r") as gf, h5py.File(args.reference, "r") as rf:
        n_new_seeds = None
        if "seeds" in gf:
            n_new_seeds = len(gf["seeds"])
        # Can the seed alignment be VERIFIED, or only assumed? Both files must
        # label the axis. An absent label is reported absent, never treated as
        # agreement (I3).
        seeds_verifiable = ("seeds" in gf) and ("seeds" in rf)
        if not seeds_verifiable:
            print(f"  NOTE: seed labels present in new={'seeds' in gf} "
                  f"reference={'seeds' in rf} -- where absent, any seed-axis "
                  f"alignment below is POSITIONAL and UNVERIFIED (gotcha 29).")
        for name in args.datasets:
            if name not in gf or name not in rf:
                missing.append({"dataset": name, "in_new": name in gf,
                                "in_reference": name in rf})
                print(f"  [ABSENT] {name}: new={name in gf} ref={name in rf}")
                continue
            got = gf[name][:]
            ref = rf[name][:]
            sliced_positionally = False
            if name in args.seed_axis_datasets:
                # A DECLARED SUBSET is compared against the leading rows of a
                # full-length reference. Where BOTH files label their seed
                # axis, the `seeds` dataset is itself compared and a
                # disagreement in order is caught. Where the reference does
                # NOT label it -- the raw null had a seed axis and no labels,
                # gotcha 29's named hazard -- the alignment is POSITIONAL and
                # unverifiable, so it is recorded as such rather than done
                # silently.
                n = n_new_seeds if n_new_seeds is not None else got.shape[0]
                if ref.shape and ref.shape[0] != got.shape[0]:
                    ref = ref[:n]
                    sliced_positionally = not seeds_verifiable
            if got.dtype.kind in "OSU":
                same = [a if isinstance(a, str) else a.decode() for a in got.tolist()] == \
                       [b if isinstance(b, str) else b.decode() for b in ref.tolist()]
                r = {"dataset": name,
                     "status": "IDENTICAL" if same else "DIFFERS",
                     "n_elements": int(got.size), "n_differing": 0 if same else -1,
                     "dtype": "string"}
            else:
                r = compare_array(name, got, ref)
            if sliced_positionally:
                r["reference_aligned_POSITIONALLY_unverified"] = True
            results.append(r)
            print(f"  [{r['status']}] {name}: {r.get('n_differing', '?')} of "
                  f"{r.get('n_elements', '?')} elements differ"
                  + (f", max |diff| {r['max_abs_difference']:.6g}"
                     if r.get("max_abs_difference") is not None else ""))
            for sl in r.get("leading_slices_affected", [])[:10]:
                print(f"           affected (seed, shift): {tuple(sl)}")
        attrs = compare_attrs(gf, rf)

    n_bad = sum(1 for r in results if r["status"] not in ("BIT_IDENTICAL", "IDENTICAL"))
    verdict = ("PAYLOAD_BIT_IDENTICAL" if not n_bad and not missing
               else "PAYLOAD_DIFFERS" if n_bad
               else "PAYLOAD_INCOMPLETE_DATASET_ABSENT")
    payload = {
        "artifact": "ac1_null_payload_comparison",
        "label": args.label,
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "new": str(args.new), "reference": str(args.reference),
        "convention": "raw bit pattern via an unsigned view; NaN==NaN, "
                      "+0.0 != -0.0",
        "verdict": verdict,
        "n_datasets_compared": len(results),
        "n_datasets_differing": n_bad,
        "datasets": results,
        "datasets_absent": missing,
        "attributes": attrs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"\n  attributes: {attrs['n_differing']} differ "
          f"({len(attrs['differing_beyond_provenance'])} beyond provenance"
          + (f": {attrs['differing_beyond_provenance']}"
             if attrs["differing_beyond_provenance"] else "") + ")")
    print(f"  only in new: {attrs['only_in_new']}")
    print(f"wrote {args.out}")
    print(f"VERDICT [{args.label}]: {verdict} "
          f"({n_bad} of {len(results)} datasets differ)")
    if verdict != "PAYLOAD_BIT_IDENTICAL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
