"""Amendment CT §CT21 (the brief's CT13): the alignment slope at each of the 13
grid θ under decoder cosine (§CT3's three intervals) and under activation
correlation (§CT4's two), on ONE axis, with the 1 pp per SD line drawn.

Data only from `ct21_figure_data_ct.json` (`ct_robustness ct21`, itself read from
§CT3's `jackknife_ct.json` and §CT4's `activation_vs_decoder_slope_curve_ct.json`);
no new computation. Style, fonts and the decoder colour are `paper_figures`'s,
so the figure sits with figures 1-3: decoder cosine keeps figure 2's rule-1
blue, and activation correlation -- a different matching criterion -- takes the
next Okabe-Ito hue rather than any colour figure 2 gave another rule.

Palette validated (dataviz `validate_palette.js`, light): CVD dE 9.6, normal
dE 24.4, all checks PASS; the purple is 2.98:1 against the surface, so both
series are DIRECT-LABELLED and differ in marker and line style as well as hue.

The intervals are drawn as short vertical bars dodged around each θ, one per
interval type, in the type's own line style; the legend names each.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.analysis.paper_figures import C_INK, C_MNN, C_MUTED, _recede, _style_rcparams

C_ACT = "#CC79A7"          # Okabe-Ito reddish purple
IV_STYLE = {"fixed_effect": "-", "across_seed": (0, (1.0, 1.0)),
            "jackknife": (0, (3.0, 1.2))}
IV_LABEL = {"fixed_effect": "fixed effect", "across_seed": "across-seed",
            "jackknife": "jackknife (40 dictionaries)"}
# what the record says (NoR §20.1 / §20.4; §CT12.4-R(b)); tight on purpose
EXPECT = {"decoder_0.70": 0.2889, "activation_0.70": 2.5656, "n_theta": 13}


def check(rows: list) -> None:
    th = [r["theta"] for r in rows]
    if len(th) != EXPECT["n_theta"] or th != sorted(th):
        raise SystemExit(f"FATAL: expected 13 ascending θ, got {th}")
    r70 = [r for r in rows if abs(r["theta"] - 0.70) < 1e-9][0]
    for k, v in (("decoder_fe_pp", EXPECT["decoder_0.70"]),
                 ("activation_fe_pp", EXPECT["activation_0.70"])):
        if round(r70[k], 4) != v:
            raise SystemExit(f"FATAL: {k} at 0.70 is {r70[k]}, the record says {v}")


def figure(rows: list):
    _style_rcparams()
    th = np.array([r["theta"] for r in rows])
    fig = plt.figure(figsize=(3.5, 2.6))
    ax = fig.add_axes([0.16, 0.30, 0.80, 0.64])
    _recede(ax)
    ax.axhline(1.0, color=C_MUTED, linewidth=0.7, linestyle=(0, (6, 2, 1, 2)), zorder=1)
    ax.text(0.905, 1.06, "1 pp per SD", fontsize=6.5, color=C_MUTED, ha="right", va="bottom")
    ax.axhline(0.0, color=C_INK, linewidth=0.5, zorder=1)
    series = (("decoder", C_MNN, "o", "-", ("fixed_effect", "across_seed", "jackknife")),
              ("activation", C_ACT, "s", (0, (4.0, 1.6)), ("fixed_effect", "across_seed")))
    for name, col, mk, ls, ivs in series:
        y = np.array([r[f"{name}_fe_pp"] for r in rows])
        ax.plot(th, y, color=col, linestyle=ls, linewidth=1.2, zorder=3)
        ax.plot(th, y, linestyle="none", marker=mk, markersize=3.0, color=col,
                markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        offs = np.linspace(-0.009, 0.009, len(ivs)) if len(ivs) > 1 else [0.0]
        for off, iv in zip(offs, ivs):
            for r in rows:
                lo, hi = r[f"{name}_intervals_95_pp"][iv]
                ax.plot([r["theta"] + off] * 2, [lo, hi], color=col, linewidth=0.7,
                        linestyle=IV_STYLE[iv], zorder=2, solid_capstyle="butt")
    # direct labels (the purple needs them, §'s palette note)
    ax.text(0.305, 1.13, "activation correlation", color=C_INK, fontsize=6.8, ha="left")
    ax.text(0.31, 0.44, "decoder cosine", color=C_INK, fontsize=6.8, ha="left")
    ax.set_xlim(0.27, 0.93)
    ax.set_xticks(np.round(np.arange(0.30, 0.91, 0.10), 2))
    ax.set_xlabel("matching threshold θ")
    ax.set_ylabel("alignment slope\n(pp of $r_f$ per SD of log $M$)")
    handles = [Line2D([], [], color=C_MNN, marker="o", markersize=3, linestyle="-",
                      linewidth=1.2, label="decoder cosine (MNN)"),
               Line2D([], [], color=C_ACT, marker="s", markersize=3,
                      linestyle=(0, (4.0, 1.6)), linewidth=1.2,
                      label="activation correlation (MNN)")]
    handles += [Line2D([], [], color=C_MUTED, linewidth=0.8, linestyle=IV_STYLE[k],
                       label=f"95% interval: {IV_LABEL[k]}") for k in IV_STYLE]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=6.0, bbox_to_anchor=(0.55, 0.0), handlelength=2.4,
               columnspacing=1.0)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("figures"))
    args = p.parse_args()
    rows = json.loads(args.data.read_text(encoding="utf-8"))["rows"]
    check(rows)
    fig = figure(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(args.outdir / f"fig_ct21_slope_across_theta.{ext}", **kw)
    print(f"wrote {args.outdir}/fig_ct21_slope_across_theta.{{pdf,png}}")


if __name__ == "__main__":
    main()
