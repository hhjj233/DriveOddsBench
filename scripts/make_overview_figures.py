"""Overview figures from the frozen TEST analyses: property scores (ranked summary and radar; higher is better)
and the skew profile (mean PIT - 1/2 for every tested quantity).

Usage: DOB_ROOT=<working directory> [DOB_FIGURE_OUT=<output dir, default figure_out>] python scripts/make_overview_figures.py
Reads only analysis outputs: results.json (frozen analysis v1), results-v12.json (frozen v1.2 secondary quantities), the
frozen whole-trajectory results (extension 2) and the continuation breaks. Writes property_scores.json, property_scores.tex,
radar_axis_ranges.json, overview_h2.pdf and calibration_profile.pdf to $DOB_FIGURE_OUT.
"""
import argparse, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()  # --help only
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import blended_transform_factory

B = os.environ.get("BENCH_DIR", os.path.join(os.environ.get("DOB_ROOT", "."), "outputs/wm-calibration/bench-v1"))
OUT = os.environ.get("DOB_FIGURE_OUT", "figure_out"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42})
DRV, GEN, PHY, REF = "#159B8E", "#6267D8", "#DF8A46", "#C0392B"   # categories as in VehDyn: driving, general, physical
SINGLE = {"Vista", "Wan2.2-TI2V-5B"}   # single-frame conditioning (case study)
# name, group, colour, line style, v1 analysis dir, v1.2 analysis dir (the order of Table 3 of the paper, with Vista last)
MODELS = [("Orbis 2", "driving", DRV, "-", "orbis2/analysis-test-v1", "orbis2/analysis-v12-test"),
          ("Cosmos3-Nano", "general", PHY, "-", "analysis-test-v1", "analysis-v12-test"),
          ("ReWorld-Video", "driving", "#2A9D8F", (0, (1, 1)), "reworld-video/stage2-p84/analysis-test-v1", "reworld-video/stage2-p84/analysis-v12-test"),
          ("Wan2.1-VACE", "general", "#9B7BC4", "--", "wan21-vace13b/test-v1/analysis-test-v1", "wan21-vace13b/test-v1/analysis-v12-test"),
          ("DrivingWorld", "driving", "#4FBFB2", (0, (6, 1.5)), "drivingworld/stage2-p84/analysis-test-v1", "drivingworld/stage2-p84/analysis-v12-test"),
          ("Cosmos3-Edge", "general", "#F0B27A", "--", "edge/analysis-test-v1", "edge/analysis-v12-test"),
          ("Epona", "driving", "#0B6B62", "--", "epona/analysis-test-v1", "epona/analysis-v12-test"),
          ("LTX-2.5", "general", GEN, "-", "ltx25d-2s/test/analysis-test-v1", "ltx25d-2s/test/analysis-v12-test"),
          ("VaViM", "driving", "#1F7A70", (0, (3, 1, 1, 1)), "vavim/stage2-p84/analysis-test-v1", "vavim/stage2-p84/analysis-v12-test"),
          ("DriveLaW-Video", "driving", "#0B4F49", ":", "drivelaw-video/stage2-p84/analysis-test-v1", "drivelaw-video/stage2-p84/analysis-v12-test"),
          ("Vista", "driving", "#7FCDBB", "-.", "vista/analysis-test-v1", "vista/analysis-v12-test")]
FAMS = ["lead", "lead_stop", "cutin", "oncoming", "ego_inter"]
# models drawn as 'pending' rows without values (name, group); empty: every model has its frozen TEST analysis
PENDING = []
R = {m[0]: json.load(open(f"{B}/{m[4]}/results.json")) for m in MODELS}
V = {m[0]: json.load(open(f"{B}/{m[5]}/results-v12.json")) for m in MODELS}

def outside(stats):
    return 100 * (stats["above_all"]["share"] + stats["below_all"]["share"])

# ---------- Figure: property scores (higher is better), (a) summary ranking, (b) radar ----------
# Scores in [0, 1], 1 = ideal; descriptive summaries of the frozen TEST results (Table 3 of the paper):
#   validity  = scorable share x (1 - share of generated clips with a boundary break at t0, J > 3); the scorable share is
#               the share of (history, family) cases over the five scored families in which the real and all 16
#               generated readings are valid; a family that cannot be read on the model's own reconstruction of real
#               video (not tested for that model: DrivingWorld's ego family, VaViM's cut-in, oncoming and ego families)
#               counts as unusable, since none of its futures can then be read for it;
#   coverage  = mean over the primary horizon and the whole trajectory of c(T, p) = 1 - |T - p| / (1 - p), with
#               T = H2 and p = 2/17, and T = least-central share and p = 1/17 (penalizes too narrow and too wide);
#   centering = 1 - 2 * mean |PIT - 1/2| over the primary quantities of the model's scored families (five for most models);
#   informativeness = mean of the point (CRPS) and trajectory (energy score) skill against climatology over the model's
#               scored families, clipped to [0, 1] (share of the climatological error that the model removes);
#   summary   = mean of the four.
TRAJ = json.load(open(f"{B}/cross-model/trajectory-calibration-v1-ext2/run-20261115-v1/results.json"))["models"]
CLIM = json.load(open(f"{B}/cross-model/trajectory-calibration-v1-ext2/run-20261115-v1/addendum-1-climatology.json"))["models"]
CB = {"Orbis 2": ("orbis2/continuation-break/orbis2-test", "Orbis2"), "Cosmos3-Nano": ("exploratory/continuation-break-v1", "Cosmos3-Nano"),
      "Cosmos3-Edge": ("exploratory/continuation-break-v1", "Cosmos3-Edge"), "Epona": ("epona/continuation-break/epona-test", "Epona"),
      "LTX-2.5": ("ltx25d-2s/continuation-break/ltx25d-2s-test", "LTX2.5-two-stage"), "Vista": ("vista/continuation-break/vista-test", "Vista"),
      "DriveLaW-Video": ("drivelaw-video/continuation-break/drivelaw-test", "DriveLaW-Video"),
      "DrivingWorld": ("drivingworld/continuation-break/drivingworld-test", "DrivingWorld"),
      "ReWorld-Video": ("reworld-video/continuation-break/reworld-test", "ReWorld-Video"),
      "Wan2.1-VACE": ("wan21-vace13b/test-v1/continuation-break/t20", "Wan2.1-VACE"),
      "VaViM": ("vavim/continuation-break-2hz/vavim-test", "VaViM")}   # VaViM on its own 2 Hz step (continuation_break_2hz.py)
# VaViM's 10 Hz canvas repeats the t0 frame until its first 2 Hz frame (frame 23), so the frame-21 test does not apply; its
# jump compares that frame with t0 against its own 0.5 s history steps
F4 = ["lead", "lead_stop", "cutin", "ego_inter"]
AXES = ["Validity", "Coverage", "Centering", "Informativeness"]
SC = {}
# DrivingWorld: its reference clips failed the frozen odometry gate D2, so its ego family is not scored and its scores use
# the other four families, as its H2 pool does (main-table footnote)
SCORED = {m[0]: [f for f in FAMS if R[m[0]]["families_results"][f]["stats"].get("n", 0) > 0] for m in MODELS}
for m in MODELS:
    r, t, fams = R[m[0]], TRAJ[m[0]]["result"], SCORED[m[0]]
    jump = json.load(open(f"{B}/{CB[m[0]][0]}/continuation-break.json"))["summary"]["groups"][CB[m[0]][1]]["boundary_break"] if m[0] in CB else 0.0
    c = lambda T, p: 1 - abs(T - p) / (1 - p)
    cov = 0.5 * (c(r["confirmatory"]["H2"]["T"], 2 / 17) + c(t["pooled"]["bd_least_central"]["T"], 1 / 17))
    scor = sum(r["scorability"][f]["kept"] for f in fams) / sum(r["scorability"][f]["windows_with_family_outcome"] for f in FAMS)   # unreadable families count as unusable
    cen = 1 - 2 * np.mean([abs(r["families_results"][f]["stats"]["mean_pit_minus_half"]) for f in fams])
    pt = np.mean([r["families_results"][f]["baselines"]["baselines"]["climatology"]["crpss"] for f in fams])
    tj = np.mean([CLIM[m[0]][f]["skill"] for f in fams])
    inf = float(np.clip(0.5 * (pt + tj), 0, 1))
    SC[m[0]] = dict(Validity=scor * (1 - jump), scorable=scor, jump=jump if m[0] in CB else None, scored_families=fams, Coverage=cov, Centering=cen, Informativeness=inf, informativeness_point=float(pt), informativeness_traj=float(tj))
    SC[m[0]]["Summary"] = float(np.mean([SC[m[0]][a] for a in AXES]))
json.dump(SC, open(os.path.join(OUT, "property_scores.json"), "w"), indent=1)
TYPE = {m[0]: m[1] for m in MODELS}
tab = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
       r"\caption{\textbf{Property scores} (TEST; definitions in \cref{app:scores}; 1 is ideal). Rows sorted by the summary score.}",
       r"\label{tab:scores}", r"\begin{tabular}{@{}l l r r r r r@{}}", r"\toprule",
       r"Model & Type & $S_{\mathrm{val}}$ & $S_{\mathrm{cov}}$ & $S_{\mathrm{cen}}$ & $S_{\mathrm{inf}}$ & $S$ \\", r"\midrule"]
for m in sorted(SC, key=lambda k: -SC[k]["Summary"]):
    v = SC[m]
    tab.append(f"{m}{' (1 frame)' if m in SINGLE else ''} & {TYPE[m]} & {v['Validity']:.3f} & {v['Coverage']:.3f} & {v['Centering']:.3f} & {v['Informativeness']:.3f} & {v['Summary']:.3f} \\\\")
tab += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "property_scores.tex"), "w").write("\n".join(tab) + "\n")

fig = plt.figure(figsize=(17 / 2.54, 7.4 / 2.54))
ax = fig.add_axes([0.20, 0.19, 0.245, 0.72])
rows = sorted(MODELS, key=lambda m: -SC[m[0]]["Summary"])
labels, ranks = ["Ideal"], ["–"]
ax.barh(0, 1.0, height=0.62, color="white", edgecolor="#7A8290", hatch="////", linewidth=0.5)
ax.text(1.0 + 0.015, 0, "1.00", va="center", fontsize=6.5, color="#5B6577")
for i, m in enumerate(rows, start=1):
    v = SC[m[0]]["Summary"]
    if i % 2 == 1:
        ax.axhspan(i - 0.5, i + 0.5, color="#F1F3F6", zorder=0, lw=0)
    ax.barh(i, v, height=0.62, color=m[2], alpha=0.9, zorder=2)
    ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=6.5, fontweight="bold", zorder=3)
    labels.append(m[0] + (" (1 frame)" if m[0] in SINGLE else "")); ranks.append(str(i))
for k, (name, g) in enumerate(PENDING, start=len(rows) + 1):
    ax.text(0.02, k, "pending", va="center", fontsize=6.2, color="#9AA3AF", style="italic")
    labels.append(name + (" (1 frame)" if name in SINGLE else "")); ranks.append("–")
ax.axvline(1.0, color="#253247", lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.tick_params(axis="y", length=0, pad=3)
tr = blended_transform_factory(fig.transFigure, ax.transData)
for k, r in enumerate(ranks):
    ax.text(0.03, k, r, transform=tr, ha="center", va="center", fontsize=6.5)
ax.text(0.03, -0.95, "Rank", transform=tr, ha="center", fontweight="bold", fontsize=6.5)
ax.text(-0.03, -0.95, "World model", transform=ax.get_yaxis_transform(), ha="right", fontweight="bold", fontsize=6.5)
ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylim(len(labels) - 0.5, -0.5)
ax.set_xlabel("summary score (mean of the four scores), higher is better", fontsize=6.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.text(0.32, 0.012, "(a)", ha="center", fontsize=7.5)

axr = fig.add_axes([0.59, 0.20, 0.20, 0.62], projection="polar")
axr.set_theta_offset(np.pi / 2); axr.set_theta_direction(-1)
ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
close = lambda v: np.r_[v, v[:1]]
# each axis is scaled separately, from its lowest to its highest model value rounded outwards to 0.05,
# so that small differences (e.g. validity 0.85-1.00) stay visible; the inner ring is that axis's low end
R0 = 0.12
LIM = {}
for a in AXES:
    vals = [SC[m[0]][a] for m in MODELS]
    lo, hi = np.floor(min(vals) * 20 + 1e-9) / 20, min(1.0, np.ceil(max(vals) * 20 - 1e-9) / 20)
    LIM[a] = (lo, max(hi, lo + 0.05))
rad = lambda a, v: R0 + (1 - R0) * (v - LIM[a][0]) / (LIM[a][1] - LIM[a][0])
for m in MODELS:
    v = np.array([rad(a, SC[m[0]][a]) for a in AXES])
    axr.plot(close(ang), close(v), color=m[2], ls=m[3], lw=0.9, marker="o", ms=1.8, zorder=3)
    axr.fill(close(ang), close(v), color=m[2], alpha=0.05, zorder=2)
axr.set_ylim(0, 1.0); axr.set_yticks([R0, (1 + R0) / 2, 1.0]); axr.set_yticklabels([])
for th, a in zip(ang, AXES):
    lo, hi = LIM[a]
    for rr, dth, val in ((R0 + 0.02, 0.55, lo), (1.0, 0.11, hi)):
        axr.text(th + dth, rr, f"{val:.2f}", fontsize=4.8, color="#5B6577", ha="center", va="center", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
axr.set_xticks(ang)
axr.set_xticklabels(AXES, fontsize=6.8)
for lab, th in zip(axr.get_xticklabels(), ang):
    if abs(np.sin(th)) > 0.5:
        lab.set_horizontalalignment("left" if np.sin(th) > 0 else "right")
axr.tick_params(axis="x", pad=4)
axr.grid(lw=0.4, color="#C9CED6"); axr.spines["polar"].set_linewidth(0.4)
fig.text(0.69, 0.012, "(b)", ha="center", fontsize=7.5)
groups = [("Driving", "driving"), ("Physical", "physical"), ("General", "general")]
y = 0.90
for title, g in groups:
    ms = [m for m in MODELS if m[1] == g]
    fig.text(0.845, y, title, fontsize=6.3, fontweight="bold")
    fig.legend([Line2D([], [], color=m[2], ls=m[3], marker="o", ms=1.8, lw=0.9) for m in ms], [m[0] + (" (1 frame)" if m[0] in SINGLE else "") for m in ms], loc="upper left",
               bbox_to_anchor=(0.838, y - 0.005), frameon=False, fontsize=6, handlelength=2.2, borderaxespad=0, labelspacing=0.3)
    y -= 0.10 + 0.064 * len(ms)
json.dump({a: list(LIM[a]) for a in AXES}, open(os.path.join(OUT, "radar_axis_ranges.json"), "w"))
fig.savefig(f"{OUT}/overview_h2.pdf")

# ---------- Figure: calibration profile, mean PIT - 1/2 (x100) for every tested quantity ----------
COLS = [("Lead", [("v1", "lead", "gap 4 s"), ("v12", "lead", "thw_4s", "headway 4 s"),
                  ("v12", "lead", "pet_follow", "PET (min)"), ("v12", "lead", "pet_t0", "PET (t0)")]),
        ("Brake/stop", [("v1", "lead_stop", "gap 4 s"), ("v12", "lead_stop", "thw_4s", "headway 4 s"),
                        ("v12", "lead_stop", "closing_speed_4s", "closing speed 4 s")]),
        ("Cut-in", [("v1", "cutin", "|lateral| 2 s"), ("v12", "cutin", "lat_speed_1_2s", "lateral speed 1-2 s")]),
        ("Oncoming", [("v1", "oncoming", "position 1 s"), ("v12", "oncoming", "exit_time", "exit time")]),
        ("Ego", [("v1", "ego_inter", "advance 0-4 s"), ("v12", "ego", "ego_dx_0_2s", "advance 0-2 s"),
                 ("v12", "ego", "ego_dx_0_6s", "advance 0-6 s"), ("v12", "ego", "ego_speed_2s", "speed 2 s"),
                 ("v12", "ego", "ego_speed_4s", "speed 4 s"), ("v12", "ego", "ego_max_decel_0_4s", "max. decel. 0-4 s")])]
# binary quantities (ego stop by 4 s) are analysed as rates, not ranks, and are left out
cols = [c for _, cs in COLS for c in cs]
ROWS = [m[0] for m in MODELS if m[0] not in SINGLE] + [p[0] for p in PENDING if p[0] not in SINGLE] + \
       [m[0] for m in MODELS if m[0] in SINGLE] + [p[0] for p in PENDING if p[0] in SINGLE]
n_video = sum(1 for r in ROWS if r not in SINGLE)
val = np.full((len(ROWS), len(cols)), np.nan); sig = np.zeros_like(val, dtype=bool)
CAVEAT = {("DrivingWorld", "ego")}   # v1.2 ego quantities shown with the D2 caveat (caption)
for m in MODELS:
    i = ROWS.index(m[0])
    tested = {(t["family"], t["quantity"]): t for t in V[m[0]]["tested"]}
    for j, c in enumerate(cols):
        if c[0] == "v1" and c[1] not in SCORED[m[0]]:
            continue                                       # not scored (DrivingWorld's ego family): n/t
        if c[0] == "v1":
            val[i, j] = 100 * R[m[0]]["families_results"][c[1]]["stats"]["mean_pit_minus_half"]
            sig[i, j] = R[m[0]]["secondary_bh"]["mean_pit"][c[1]]["significant_q_0_10"]
        elif (c[1], c[2]) in tested:
            t = tested[(c[1], c[2])]
            val[i, j] = 100 * t["stats"]["mean_pit_minus_half"]; sig[i, j] = t["bh_q"]["mean_pit"] <= 0.10
fig, ax = plt.subplots(figsize=(17 / 2.54, 8.0 / 2.54))
pending_rows = {ROWS.index(p[0]) for p in PENDING}
lim = 50
im = ax.imshow(np.clip(val, -lim, lim), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for i in range(val.shape[0]):
    for j in range(val.shape[1]):
        if i in pending_rows:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#F4F5F7", lw=0))
            if j == len(cols) // 2:
                ax.text(j, i, "pending", ha="center", va="center", fontsize=5.6, color="#9AA3AF", style="italic")
        elif np.isnan(val[i, j]):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#ECEEF1", lw=0))
            ax.text(j, i, "n/t", ha="center", va="center", fontsize=5, color="#9AA3AF")
        else:
            v = val[i, j]
            ax.text(j, i, ("0" if round(v) == 0 else f"{v:+.0f}").replace("-", "−") + ("†" if (ROWS[i], cols[j][1]) in CAVEAT else ""), ha="center", va="center", fontsize=5.6,
                    fontweight="bold" if sig[i, j] else "normal",
                    color=("white" if abs(v) > 30 else "#1C2229") if sig[i, j] else "#6B7480")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[-1] for c in cols], rotation=40, ha="right", fontsize=6)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS, fontsize=6.5)
for i in pending_rows:
    ax.get_yticklabels()[i].set_color("#7A8290")
ax.tick_params(length=0)
x = -0.5
for g, cs in COLS:
    ax.text(x + len(cs) / 2, -0.75, g, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    x += len(cs)
    if x < len(cols) - 0.5:
        ax.axvline(x, color="black", lw=0.8)
ax.axhline(n_video - 0.5, color="black", lw=0.8)
ax.text(len(cols) - 0.4, (n_video + len(ROWS) - 1) / 2, " single\n frame", va="center", fontsize=5.6, color="#5B6577")
for s in ax.spines.values():
    s.set_linewidth(0.5)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.075)
cb.set_label("mean PIT − ½ (×100)", fontsize=6); cb.ax.tick_params(labelsize=5.5, width=0.4, length=2); cb.outline.set_linewidth(0.4)
fig.subplots_adjust(left=0.13, right=0.935, top=0.90, bottom=0.24)
fig.savefig(f"{OUT}/calibration_profile.pdf")
print("max |value|", np.nanmax(np.abs(val)).round(1), "tested cells", int(np.sum(~np.isnan(val))), "significant", int(sig.sum()))
