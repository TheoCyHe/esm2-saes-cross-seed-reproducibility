"""The manuscript's figures. THREE FLOATS, built from artifacts rather than tables.

=============================================================================
WHY THREE AND NOT ONE — the narrative moved, and the figure set moved with it
=============================================================================
The combined float this module first produced put THREE matching rules in the
measurement section, where **only one rule exists**. A reader there had to
ignore two thirds of the figure or jump forward three sections. The set was
deliberately left undeclared so the narrative could move it; it moved.

  * **Figure 1 — the measurement section.** This project's own measurement under
    its own rule, and nothing else. **No other rule appears in it, in any form,
    under any flag.**
  * **Figure 2 — the measurement-validity section.** The three rules and their
    difference. The old float's left half, repaired.
  * **Figure 3 — generated, and the drafting side decides whether it ships.**
    The reconstruction spread. That section may carry it in prose instead.

=============================================================================
THE DENOMINATOR, WHICH IS THE THING MOST LIKELY TO GO WRONG HERE
=============================================================================
**FIGURE 1 IS ENTIRELY AN `n_compare = 39` OBJECT. FIGURE 2 IS ENTIRELY AN
`n_compare = 19` OBJECT.** Both panels of figure 1 agree with each other; both
panels of figure 2 agree with each other; **the two figures do not agree with
each other and must never be read across.**

**AND THE COLLISION IS WORSE THAN "TWO NUMBERS THAT LOOK ALIKE" — MEASURED, NOT
FEARED.** The 39 curve and the 19 curve agree to **four significant figures at
every one of the thirteen shared grid points**:

    theta        39 (figure 1)      19 (figure 2, rule 1)
    0.30         0.8056391          0.8056209
    0.70         0.4955127          0.4958141
    0.90         0.0887644          0.0889700

The largest gap anywhere on the grid is at `theta = 0.85`, and it is 4.8e-04.
**A reader cannot tell these two curves apart by inspection, and nothing about
either number announces which comparison set it belongs to.** That is why every
caption this module emits names its comparison set, and why the two figures use
visibly different y-axis labels rather than the same words.

**FIGURE 1's PANEL A IS READ FROM `design3_join_aw_nofeat.json`'s
`theta_curve`** — the same artifact and the same curve `PREREGISTRATION.md`
§CD-a reads the headline `0.495513` from, whose `identity.n_compare_seeds` is
**39**. It is NOT read from the rule-sensitivity artifact, which is 19.

=============================================================================
FIGURE 2's FOUR REPAIRS, EACH AGAINST A REPORTED DEFECT
=============================================================================
1. **The legend is OUTSIDE the axes**, horizontal, below the difference strip.
   The previous version put it inside panel A, where its text crossed the
   rule-3 curve. **No text overlaps any data in either panel.**
2. **Margins are explicit and generous**, and `savefig.bbox` is NOT `"tight"` --
   that renegotiates the figure size at save time and the width is a venue
   constraint, not a preference.
3. **The strip says what it is.** `Hungarian - MNN` does not say WHICH
   Hungarian and a bare `Delta (pp)` does not say delta of what. The axis now
   names the quantity and the panel states the sign convention in full, and
   **the interval the ordering reverses in is marked** rather than left to be
   found.
   > **CHANGED: IT WAS A POINT AND IT IS NOW AN INTERVAL.** The previous
   > version interpolated a crossing abscissa between the bracketing grid
   > points and drew a marker labelled `theta ~= 0.54`. **The grid steps are
   > 0.05, so no crossing point is measured** -- 0.54 is a rendering artefact,
   > it is in no artifact and no committed section, and it disagreed with the
   > manuscript's own *"between 0.50 and 0.55"*. A shaded band over
   > `[0.50, 0.55]` is what the data supports. `crossing_bracket` is the one
   > locator, read by the band, its label AND `check_against_record`, so the
   > three cannot drift apart.
4. **Every panel has its own letter**, in all three figures. The previous float
   gave one letter to two panels.

=============================================================================
FIGURE 3 — AND THE PROHIBITION THAT DEFINES IT
=============================================================================
**Figure 3 is NOT half of a function-versus-basis contrast, and it must never be
placed beside the across-pair spread of the match rate** (`NUMBERS_OF_RECORD`
§15's prohibition). Those are **two aggregates**, and pairing them contradicts
§CF7a rather than illustrating it -- measured: relative across-seed spread of
FVU is 0.31%, relative across-pair spread of the match rate is 1.11%, the same
order of magnitude. **The identity side of that contrast is FIGURE 1's PANEL B**,
which is a distribution and not an aggregate.

Both spaces travel, every time. **Raw is never shown alone**: it is the more
flattering number in 20 of 20 seeds by ~0.0148 (`NUMBERS_OF_RECORD` §6.2).

=============================================================================
WHAT IS KEPT FROM THE VERSION THAT WORKED
=============================================================================
Vector PDF with embedded TrueType (`pdf.fonttype = 42`; Type 3 trips submission
checkers), explicit margins rather than a tight box, 8 pt serif tick and axis
labels, Okabe-Ito run through the validator rather than eyeballed (all six
checks PASS, worst adjacent pair deltaE 11.0 under deuteranopia), **line style
as the primary encoding so the figures survive greyscale**, no style-file
changes of any kind.

**AND THE RECORD ASSERTIONS, EXTENDED TO EVERY NEW QUANTITY ANY PANEL DRAWS** --
the 39 theta curve's three endpoints and its grid length are new here and are
asserted. **A figure that cannot verify its own numbers refuses rather than
renders**, exit 1. Both directions are mutation-tested; see the module's own
`--expect-*` overrides.

=============================================================================
RENDER THIS LOCALLY, NOT ON BOUCHET -- THE FONT FALLS THROUGH THERE
=============================================================================
**MEASURED 2026-08-31, by rendering in both places and reading `/BaseFont` out
of the PDFs.** `font.serif` lists five faces; **the compute node has only the
LAST one.** `Times New Roman`, `Nimbus Roman No9 L`, `Liberation Serif` and
`STIX Two Text` are all absent from the cluster env, so a cluster render comes
out in **DejaVu Serif** -- silently, because a font fallback is not an error.
The Windows checkout has **Times New Roman**, the first in the chain.

**So the figures are rendered LOCALLY, from artifacts copied down** (they total
~2 MB: three JSONs and one small h5, `--d3 --cj --fvu --rf`). **Nothing is
computed here** -- the script reads stored numbers and draws them -- so this is
a typography decision and NOT a gotcha 36 device-class question.

**THE CHECK, if the chain is ever edited:** `/BaseFont` in the output PDF must
name a Times face. A submission whose body text is Times and whose figures are
DejaVu is the failure this note exists to prevent, and it is invisible in a
thumbnail.

**THE SCRIPT LIVES IN `src/analysis/` AND NOT `src/figures/`.** `.gitignore`
line 12 is `figures/` with no leading slash, which matches a directory of that
name at ANY depth -- a script there is invisible to git and lost at the next
clone while appearing delivered. Output still goes to `figures/`, which is
ignored on purpose: the PDFs are derived artefacts and this script regenerates
them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- the cluster's literal paths. NEVER os.path.expandvars("$PROJECT/...") ----
PROJECT_ANALYSIS = "$PROJECT/analysis"
DEFAULT_D3 = f"{PROJECT_ANALYSIS}/design3_join_aw_nofeat.json"
DEFAULT_CJ = f"{PROJECT_ANALYSIS}/cj_rule_sensitivity.json"
DEFAULT_FVU = f"{PROJECT_ANALYSIS}/fvu_eval_cf.json"
DEFAULT_RF = f"{PROJECT_ANALYSIS}/rf_counts_theta070_n39.h5"

# --- Okabe-Ito, validated: all six checks PASS, worst adjacent CVD dE 11.0 ----
C_MNN = "#0072B2"     # blue
C_HUNG = "#D55E00"    # vermillion
C_ENCDEC = "#009E73"  # bluish green
C_INK = "#1a1a1a"
C_MUTED = "#6b6b6b"
C_FAINT = "#c9c9c9"

# Line style is the PRIMARY encoding; colour is redundant with it by design.
STYLE = {
    "rule1": dict(color=C_MNN, linestyle="-", linewidth=1.3),
    "rule2": dict(color=C_HUNG, linestyle=(0, (4.0, 1.6)), linewidth=1.3),
    "rule3": dict(color=C_ENCDEC, linestyle=(0, (1.2, 1.2)), linewidth=1.5),
}
LABEL = {
    "rule1": "mutual nearest neighbours, decoder cosine",
    "rule2": "Hungarian one-to-one, decoder cosine",
    "rule3": "Hungarian, encoder AND decoder agree",
}

REFERENCE_THETA = 0.70

# --- what the committed record says these artifacts contain -------------------
# Tolerances are TIGHT on purpose: an assertion that cannot fail is worse than
# no assertion (gotcha 13, gotcha 25a).
EXPECT = {
    # NUMBERS_OF_RECORD 1.1, owner PREREGISTRATION CD-a -- the n_compare=39 curve
    "d3_theta030": 0.805639,
    "d3_theta070": 0.495513,
    "d3_theta090": 0.088764,
    "d3_n_grid": 13,
    "d3_n_compare": 39,
    "d3_n_anchors": 20,
    # NUMBERS_OF_RECORD 13.1, owner PREREGISTRATION CJ9a -- the n_compare=19 rules
    "rule1_at_ref": 0.4958,
    "rule2_at_ref": 0.4908,
    "rule3_at_ref": 0.3038,
    "rule2_minus_rule1_at_030_pp": 11.2,
    "n_pairs": 190,
    # THE ONE ENTRY WHOSE SOURCE IS NOT A NUMBERS_OF_RECORD SECTION, and it is
    # labelled rather than left to look like the others. NUMBERS_OF_RECORD 13.1
    # qualifier 1 states the reversal ("at 0.30 rule 2 sits 11.2 pp above rule
    # 1; by 0.70 it has fallen below it") but NOT which grid interval it falls
    # in. The pair below is the MANUSCRIPT's sentence, written down here so the
    # artifact can contradict it -- which is the direction that matters.
    "crossing_bracket": (0.50, 0.55),
    # NUMBERS_OF_RECORD 6.2, owner PREREGISTRATION CF7a
    "fvu_raw_mean": 0.363821,
    "fvu_std_mean": 0.378630,
    "fvu_raw_sd": 0.001118,
    "fvu_std_sd": 0.001162,
    # NUMBERS_OF_RECORD 14, owner PREREGISTRATION AL6 rows A4 and C1
    "p_k0_n39": 0.095146,
    "p_kN_n39": 0.039585,
    "interior_n39": 0.865269,
    "mean_rf_n39": 0.495513,
}


def _style_rcparams() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Liberation Serif",
                       "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.edgecolor": C_INK,
        "text.color": C_INK,
        "axes.labelcolor": C_INK,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "figure.dpi": 300,
        # NOT bbox="tight": the width is a venue constraint, not a preference.
        "savefig.bbox": None,
    })


def _recede(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", axis="both", color="#e2e2e2",
            linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def _letter(ax, tag, dx=-0.20, dy=1.13, va="top"):
    """Panel letter. In a STACKED figure it goes ABOVE the axes, not beside
    them: a two-line y-label occupies the left margin and the letter lands on
    top of it."""
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va=va, ha="left", color=C_INK)


# =============================================================================
# INPUTS
# =============================================================================

def load_d3(path: str) -> dict:
    """The n_compare = 39 theta curve -- the SAME curve CD-a reads the headline
    from. Not the rule-sensitivity artifact, which is 19."""
    with open(path) as fh:
        d = json.load(fh)
    rows = d["theta_curve"]["rows"]
    return {
        "theta": np.asarray([r["theta"] for r in rows], dtype=float),
        "mean_r_f": np.asarray([r["mean_r_f"] for r in rows], dtype=float),
        "n_compare": int(d["identity"]["n_compare_seeds"]),
        "n_anchors": int(d["identity"]["n_anchors"]),
        "reference_theta": float(d["identity"]["reference_theta"]),
    }


def load_cj(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    grid = np.asarray(d["theta_grid"], dtype=float)
    out = {"theta": grid, "n_pairs": int(d["n_pairs"]),
           "device_class": d["device_class"],
           "reference_theta": float(d["reference_theta"])}
    for key, src in (("rule1", "rule1_mnn"), ("rule2", "rule2_hung"),
                     ("rule3", "rule3_encdec")):
        out[key] = np.asarray([d["across_theta_grid"][src][f"{t:g}"] for t in grid],
                              dtype=float)
    return out


def load_fvu(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    return {
        "k": int(d["k"]),
        "raw": np.asarray(list(d["spread_raw"]["values_by_seed"].values()), float),
        "std": np.asarray(list(d["spread_standardised"]["values_by_seed"].values()), float),
        "raw_stats": d["spread_raw"],
        "std_stats": d["spread_standardised"],
    }


def load_rf(path: str) -> dict:
    with h5py.File(path, "r") as h:
        mc = h["match_count"][:]
        n_compare = int(h.attrs["n_compare_seeds"])
        n_anchors = int(h.attrs["n_anchors"])
        theta = float(h.attrs["theta"])
    n = mc.size
    return {
        "n_cells": n, "n_compare": n_compare, "n_anchors": n_anchors, "theta": theta,
        "p_k0": float((mc == 0).sum() / n),
        "p_kN": float((mc == n_compare).sum() / n),
        "interior": float(1.0 - (mc == 0).sum() / n - (mc == n_compare).sum() / n),
        "mean_rf": float((mc / n_compare).mean()),
    }


def crossing_bracket(theta, diff) -> tuple[float, float]:
    """The two ADJACENT GRID POINTS the rule-2-minus-rule-1 sign change falls
    between. Returns grid values, never an interpolated abscissa.

    THE OUTCOME SPACE IS ENUMERATED IN THE PRIMITIVE THE DATA HAS (gotcha 39),
    and the primitive here is a SIGN SEQUENCE WITH TIES IN IT, not a sequence
    of strictly-signed values. `diff` is sampled on a 0.05 grid, so:

      no sign change   -- the reversal this panel exists to show is absent;
      more than one    -- "the ordering reverses" is false as written, and one
                          band would hide the others;
      a sign change SEPARATED BY ONE OR MORE EXACT TIES -- the reversal passes
                          through zero AT a grid point, so it is a value of
                          record and not an interval, and the band would
                          understate what the artifact contains.

    All three refuse. **A TIE THAT DOES NOT SEPARATE OPPOSITE SIGNS IS NOT A
    CROSSING AND IS IGNORED**, and that case is real rather than hypothetical:
    the record has rules 1 and 2 COINCIDING at theta = 0.95 (both 2.6199%,
    `NUMBERS_OF_RECORD` 13.1 qualifier 1 as repaired at CO9b). Testing `diff ==
    0` anywhere would refuse on the actual artifact -- which is what an earlier
    draft of this function did, caught by running it.
    """
    theta = np.asarray(theta, dtype=float)
    diff = np.asarray(diff, dtype=float)
    nz = np.where(diff != 0.0)[0]
    if nz.size == 0:
        raise SystemExit("REFUSING: the difference is identically zero; there "
                         "is no ordering to reverse.")
    sgn = np.sign(diff[nz])
    flips = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if flips.size == 0:
        raise SystemExit("REFUSING: the difference strip has no sign change; "
                         "the reversal this panel exists to show is absent.")
    if flips.size > 1:
        brackets = [(float(theta[nz[m]]), float(theta[nz[m + 1]]))
                    for m in flips]
        raise SystemExit(
            f"REFUSING: the difference changes sign {flips.size} times, in "
            f"{brackets}. The panel says 'the ordering reverses' in the "
            f"singular and one band would hide the rest.")
    a, b = int(nz[int(flips[0])]), int(nz[int(flips[0]) + 1])
    if b != a + 1:
        raise SystemExit(
            f"REFUSING: the sign change spans theta={theta[a]:.4f} to "
            f"{theta[b]:.4f} with exact ties at "
            f"{theta[a + 1:b].tolist()} in between. The reversal is measured "
            f"AT a grid point, so it is a value of record and not an interval.")
    return float(theta[a]), float(theta[b])


def check_against_record(d3, cj, fvu, rf) -> list[str]:
    """Refuse on mismatch. Returns the lines verified, so a silent pass is
    visible as output rather than as absence of output (gotcha 28)."""
    lines: list[str] = []

    def near(name, got, want, tol):
        if abs(got - want) > tol:
            raise SystemExit(
                f"REFUSING: {name} is {got!r}, record says {want!r} "
                f"(tolerance {tol}). The artifact and the record disagree; "
                f"that is a finding, not something to plot around.")
        lines.append(f"  OK  {name:<40} {got:.6f}  (record {want})")

    def eq(name, got, want):
        if got != want:
            raise SystemExit(f"REFUSING: {name} is {got}, record says {want}.")
        lines.append(f"  OK  {name:<40} {got}")

    # --- figure 1 panel A: the n_compare = 39 curve (NEW in this version) ----
    eq("d3 n_compare_seeds", d3["n_compare"], EXPECT["d3_n_compare"])
    eq("d3 n_anchors", d3["n_anchors"], EXPECT["d3_n_anchors"])
    eq("d3 qualifying grid points", d3["theta"].size, EXPECT["d3_n_grid"])
    i30 = int(np.argmin(np.abs(d3["theta"] - 0.30)))
    i70 = int(np.argmin(np.abs(d3["theta"] - REFERENCE_THETA)))
    i90 = int(np.argmin(np.abs(d3["theta"] - 0.90)))
    near("d3 mean r_f @ 0.30 (n_compare=39)", d3["mean_r_f"][i30], EXPECT["d3_theta030"], 5e-6)
    near("d3 mean r_f @ 0.70 (n_compare=39)", d3["mean_r_f"][i70], EXPECT["d3_theta070"], 5e-6)
    near("d3 mean r_f @ 0.90 (n_compare=39)", d3["mean_r_f"][i90], EXPECT["d3_theta090"], 5e-6)

    # --- figure 2: the n_compare = 19 rules ---------------------------------
    j = int(np.argmin(np.abs(cj["theta"] - REFERENCE_THETA)))
    near("rule 1 @ 0.70 (n_compare=19)", cj["rule1"][j], EXPECT["rule1_at_ref"], 5e-5)
    near("rule 2 @ 0.70 (n_compare=19)", cj["rule2"][j], EXPECT["rule2_at_ref"], 5e-5)
    near("rule 3 @ 0.70 (n_compare=19)", cj["rule3"][j], EXPECT["rule3_at_ref"], 5e-5)
    near("rule2-rule1 @ 0.30 (pp)", 100.0 * (cj["rule2"][0] - cj["rule1"][0]),
         EXPECT["rule2_minus_rule1_at_030_pp"], 0.05)
    eq("cj n_pairs", cj["n_pairs"], EXPECT["n_pairs"])
    # The band panel B draws. Asserted HERE as well as located there, so a
    # reversal that moved to a different interval refuses at the same gate as
    # every other number rather than silently relabelling the annotation.
    got_lo, got_hi = crossing_bracket(cj["theta"], cj["rule2"] - cj["rule1"])
    want_lo, want_hi = EXPECT["crossing_bracket"]
    # A tolerance, not `==`: these are grid points that arrived through JSON and
    # a float compare on 0.55 is a trap. Half a grid step is far tighter than
    # the thing being checked -- the neighbouring bracket is a full step away,
    # so this still fails if the reversal moves at all (verified by control).
    if abs(got_lo - want_lo) > 0.005 or abs(got_hi - want_hi) > 0.005:
        raise SystemExit(
            f"REFUSING: the rule2-rule1 reversal falls between theta="
            f"{got_lo:.4f} and {got_hi:.4f}; the manuscript says between "
            f"{want_lo} and {want_hi}. Figure 2's band and the prose would "
            f"disagree, which is what this annotation was changed to fix.")
    lines.append(f"  OK  {'rule2-rule1 reversal bracket':<40} "
                 f"[{got_lo:.2f}, {got_hi:.2f}]  (manuscript "
                 f"[{want_lo:.2f}, {want_hi:.2f}])")

    # --- figure 3 ------------------------------------------------------------
    near("FVU raw mean", float(fvu["raw_stats"]["mean"]), EXPECT["fvu_raw_mean"], 5e-6)
    near("FVU std mean", float(fvu["std_stats"]["mean"]), EXPECT["fvu_std_mean"], 5e-6)
    near("FVU raw sd", float(fvu["raw_stats"]["sd"]), EXPECT["fvu_raw_sd"], 5e-6)
    near("FVU std sd", float(fvu["std_stats"]["sd"]), EXPECT["fvu_std_sd"], 5e-6)
    eq("FVU k", fvu["k"], 32)

    # --- figure 1 panel B ----------------------------------------------------
    near("P(k=0), n_compare=39", rf["p_k0"], EXPECT["p_k0_n39"], 5e-6)
    near("P(k=39), n_compare=39", rf["p_kN"], EXPECT["p_kN_n39"], 5e-6)
    near("interior mass, n_compare=39", rf["interior"], EXPECT["interior_n39"], 5e-6)
    near("mean r_f, n_compare=39", rf["mean_rf"], EXPECT["mean_rf_n39"], 5e-6)
    eq("rf n_compare", rf["n_compare"], 39)

    # --- the cross-figure invariant the denominators depend on ---------------
    if d3["n_compare"] != rf["n_compare"]:
        raise SystemExit(
            f"REFUSING: figure 1's two panels disagree on the comparison set "
            f"({d3['n_compare']} against {rf['n_compare']}). The caption's "
            f"denominator claim would be false.")
    lines.append(f"  OK  {'figure 1 panels share n_compare':<40} {d3['n_compare']}")
    return lines


# =============================================================================
# FIGURE 1 -- the measurement section. ENTIRELY n_compare = 39.
# =============================================================================

def figure1(d3: dict, rf: dict):
    fig = plt.figure(figsize=(5.4, 2.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.42,
                          left=0.105, right=0.985, bottom=0.215, top=0.87)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # --- A: the MNN curve ALONE. No other rule, in any form. ----------------
    axa.plot(d3["theta"], 100 * d3["mean_r_f"], zorder=3, **STYLE["rule1"])
    axa.plot(d3["theta"], 100 * d3["mean_r_f"], linestyle="none", marker="o",
             markersize=2.2, markerfacecolor=C_MNN, markeredgecolor="none",
             zorder=4)
    axa.axvline(REFERENCE_THETA, color=C_MUTED, linewidth=0.6,
                linestyle=(0, (2, 2)), zorder=2)
    i70 = int(np.argmin(np.abs(d3["theta"] - REFERENCE_THETA)))
    axa.annotate("reference", xy=(REFERENCE_THETA, 100 * d3["mean_r_f"][i70]),
                 xytext=(0.775, 78), fontsize=6.5, color=C_MUTED,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=C_MUTED, linewidth=0.5,
                                 shrinkA=0, shrinkB=2))
    axa.set_xlabel(r"matching threshold $\theta$ (decoder cosine)")
    axa.set_ylabel("mean reproducibility\n" r"$r_f$ (%), $n_{compare}=39$")
    axa.set_xlim(0.27, 0.93)
    axa.set_ylim(-2, 92)
    axa.set_xticks([0.3, 0.5, 0.7, 0.9])
    axa.set_yticks([0, 25, 50, 75])
    _recede(axa)
    _letter(axa, "A", dx=-0.235)

    # --- B: the identity distribution, same comparison set ------------------
    shares = [rf["p_k0"], rf["interior"], rf["p_kN"]]
    for i, (s, c) in enumerate(zip(shares, [C_FAINT, C_MNN, C_FAINT])):
        axb.bar(i, 100 * s, width=0.66, color=c, edgecolor=C_INK,
                linewidth=0.5, zorder=3)
        axb.text(i, 100 * s + 2.5, f"{100*s:.1f}", ha="center", va="bottom",
                 fontsize=6.5, color=C_INK)
    axb.set_xlim(-0.62, 2.62)
    axb.set_xticks([0, 1, 2])
    axb.set_xticklabels(["0", "1-38", "39"])
    axb.set_ylim(0, 100)
    axb.set_yticks([0, 25, 50, 75, 100])
    axb.set_xlabel("dictionaries matched (of 39)")
    axb.set_ylabel("share of feature-seed cells (%)")
    _recede(axb)
    _letter(axb, "B", dx=-0.34)
    return fig


# =============================================================================
# FIGURE 2 -- the measurement-validity section. ENTIRELY n_compare = 19.
# =============================================================================

def figure2(cj: dict):
    fig = plt.figure(figsize=(5.4, 2.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.15, 1.0], hspace=0.32,
                          left=0.135, right=0.985, bottom=0.375, top=0.895)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0], sharex=axa)
    th = cj["theta"]

    handles = []
    for key in ("rule3", "rule2", "rule1"):
        (ln,) = axa.plot(th, 100 * cj[key], label=LABEL[key], zorder=3, **STYLE[key])
        handles.append(ln)
    axa.axvline(REFERENCE_THETA, color=C_MUTED, linewidth=0.6,
                linestyle=(0, (2, 2)), zorder=2)
    axa.set_ylabel("features matched (%)\n" r"$n_{compare}=19$")
    axa.set_xlim(0.27, 0.98)
    axa.set_ylim(-3, 100)
    axa.set_xticks([0.3, 0.5, 0.7, 0.9])
    axa.set_yticks([0, 25, 50, 75, 100])
    axa.tick_params(labelbottom=False)
    _recede(axa)
    _letter(axa, "A", dx=-0.018, dy=1.03, va="bottom")

    # --- B: the difference strip, named in full -----------------------------
    diff = 100.0 * (cj["rule2"] - cj["rule1"])
    axb.axhline(0.0, color=C_INK, linewidth=0.6, zorder=2)
    axb.plot(th, diff, color=C_MUTED, linewidth=1.1, zorder=3)
    axb.plot(th, diff, linestyle="none", marker="o", markersize=1.9,
             markerfacecolor=C_MUTED, markeredgecolor="none", zorder=4)
    axb.axvline(REFERENCE_THETA, color=C_MUTED, linewidth=0.6,
                linestyle=(0, (2, 2)), zorder=1)

    # Mark the INTERVAL the reversal happens in, not a point. The grid steps
    # are 0.05 and nothing was measured between two grid points, so an
    # interpolated crossing abscissa is a rendering artefact rather than a
    # number of record -- the previous version drew a marker at
    # `theta ~= 0.54`, which no artifact contains and which disagreed with the
    # manuscript's own "between 0.50 and 0.55". `crossing_bracket` is the
    # shared locator, so the band, its label and `check_against_record` all
    # read the same two grid points.
    lo, hi = crossing_bracket(th, diff)
    axb.axvspan(lo, hi, color=C_FAINT, alpha=0.55, linewidth=0, zorder=0)
    # ONE line, and short. Panel B is ~0.41 in tall spanning 16.9 pp, so the
    # gap between this label at y = 6.4 and the panel's own caption at y = 12.6
    # is about 11 pt -- less than two 6.5 pt lines. A two-line version was
    # drawn and it overlapped the caption, which is repair 1's constraint
    # ("no text overlaps any data in either panel") failing on the annotation
    # that was added to fix a different defect. The interval carries no
    # "theta" of its own because the x axis directly below already names it.
    axb.annotate(f"reverses in [{lo:.2f}, {hi:.2f}]",
                 xy=(0.5 * (lo + hi), 0.0),
                 xytext=(hi + 0.035, 6.4), fontsize=6.5, color=C_INK,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=C_INK, linewidth=0.5,
                                 shrinkA=0, shrinkB=3))

    axb.set_xlim(0.27, 0.98)
    axb.set_ylim(-3.4, 13.5)
    axb.set_xticks([0.3, 0.5, 0.7, 0.9])
    axb.set_yticks([0, 5, 10])
    axb.set_xlabel(r"matching threshold $\theta$ (decoder cosine)")
    axb.set_ylabel("difference (pp)")
    axb.text(0.955, 12.6, "Hungarian one-to-one $-$ mutual nearest neighbours",
             ha="right", va="top", fontsize=6.5, color=C_MUTED)
    _recede(axb)
    _letter(axb, "B", dx=-0.018, dy=1.06, va="bottom")

    # --- the legend, OUTSIDE the axes, horizontal, below everything ---------
    # ncol=1, NOT the horizontal three-across the brief suggested: measured,
    # the three criterion names total 113 characters and need ~6.1 in on one
    # row against 4.64 in of usable width. Three across fits only by
    # abbreviating the criteria, which is the one thing the legend is for.
    fig.legend(handles=handles[::-1], loc="lower center", ncol=1,
               bbox_to_anchor=(0.55, 0.008), frameon=False, handlelength=2.2,
               labelspacing=0.22, handletextpad=0.5, columnspacing=1.2,
               fontsize=7)
    return fig


# =============================================================================
# FIGURE 3 -- one panel. Generated; the drafting side decides whether it ships.
# =============================================================================

def figure3(fvu: dict):
    fig = plt.figure(figsize=(2.7, 2.2))
    ax = fig.add_axes([0.295, 0.175, 0.67, 0.795])
    rng = np.random.default_rng(0)   # jitter only; deterministic
    for i, (vals, col) in enumerate([(fvu["raw"], C_MNN), (fvu["std"], C_HUNG)]):
        x = i + (rng.random(vals.size) - 0.5) * 0.30
        ax.scatter(x, vals, s=6.0, facecolors="none", edgecolors=col,
                   linewidths=0.6, zorder=3, clip_on=False)
        ax.hlines(vals.mean(), i - 0.30, i + 0.30, color=C_INK,
                  linewidth=1.0, zorder=4)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "standardized"])
    ax.set_ylim(0.358, 0.386)
    ax.set_yticks([0.36, 0.37, 0.38])
    ax.set_xlabel("measurement space")
    ax.set_ylabel("FVU on the evaluation split")
    _recede(ax)
    # NO panel letter: this is a single-panel float. A lone "A" on one panel
    # labels nothing, and the lettering requirement is about the two
    # multi-panel figures.
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--d3", default=DEFAULT_D3)
    p.add_argument("--cj", default=DEFAULT_CJ)
    p.add_argument("--fvu", default=DEFAULT_FVU)
    p.add_argument("--rf", default=DEFAULT_RF)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--expect-interior", type=float, default=None,
                   help="override the expected interior mass; used ONLY to "
                        "demonstrate that the record check can fail.")
    p.add_argument("--expect-d3-theta070", type=float, default=None,
                   help="override the expected n_compare=39 curve value at the "
                        "reference; used ONLY to demonstrate the same.")
    args = p.parse_args()

    _style_rcparams()
    if args.expect_interior is not None:
        EXPECT["interior_n39"] = args.expect_interior
    if args.expect_d3_theta070 is not None:
        EXPECT["d3_theta070"] = args.expect_d3_theta070

    d3 = load_d3(args.d3)
    cj = load_cj(args.cj)
    fvu = load_fvu(args.fvu)
    rf = load_rf(args.rf)

    print("checking artifacts against the committed record:")
    for line in check_against_record(d3, cj, fvu, rf):
        print(line)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    built = [("fig1_reproducibility", figure1(d3, rf)),
             ("fig2_matching_rules", figure2(cj)),
             ("fig3_reconstruction_spread", figure3(fvu))]
    print()
    for name, fig in built:
        fig.savefig(outdir / f"{name}.pdf")
        fig.savefig(outdir / f"{name}.png", dpi=400)
        w, h = fig.get_size_inches()
        plt.close(fig)
        print(f"wrote {outdir / (name + '.pdf')}  ({w:g} x {h:g} in)")
        print(f"wrote {outdir / (name + '.png')}")

    print("\nDENOMINATORS, so a caption cannot get them wrong:")
    print(f"  figure 1 (both panels): n_compare = {d3['n_compare']}, "
          f"{d3['n_anchors']} anchors")
    print(f"  figure 2 (both panels): n_compare = 19, {cj['n_pairs']} unordered "
          f"anchor pairs, device {cj['device_class']}")
    print(f"  figure 3:               {fvu['k']=}, 20 anchors, both spaces")
    i70 = int(np.argmin(np.abs(d3["theta"] - REFERENCE_THETA)))
    j70 = int(np.argmin(np.abs(cj["theta"] - REFERENCE_THETA)))
    print(f"\n  the collision, measured: at the reference the 39 curve is "
          f"{d3['mean_r_f'][i70]:.7f} and the 19 curve is "
          f"{cj['rule1'][j70]:.7f} -- same four significant figures, "
          f"different objects.")


if __name__ == "__main__":
    main()
