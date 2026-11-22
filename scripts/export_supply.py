#!/usr/bin/env python3
"""Export the numbers behind the paper's figures and tables: one CSV per dataset, each with a Markdown description.

Run from the repository root:   DOB_ROOT=<working directory> [DOB_FIGURE_OUT=<dir>] python scripts/export_supply.py [--frames]
Inputs: the frozen analysis outputs under $DOB_ROOT/outputs/wm-calibration/bench-v1/, and property_scores.json and
radar_axis_ranges.json in $DOB_FIGURE_OUT (written by scripts/make_overview_figures.py from the same files). --frames (needs the
rendered windows and model clips, BENCH_DIR=<bench-v1 root>) also writes the seven Figure 1 frames from the AV2 window and the
model clips. Writes $DOB_FIGURE_OUT/*.csv and their descriptions *.md, and teaser/. Numbers are written at full precision (repr).
--pipeline (needs the rendered windows, model clips and measurement files) writes only $DOB_FIGURE_OUT/pipeline/: the frames,
overlays and readings of the example history of the overview figure (Fig. 2)."""
import argparse, collections, csv, datetime as dt, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]            # repository root
BD = Path(os.environ.get("DOB_ROOT", "."), "outputs/wm-calibration/bench-v1")
OUT = Path(os.environ.get("DOB_FIGURE_OUT", "figure_out"))
TODAY = dt.date.today().isoformat()
SCRIPT = "scripts/export_supply.py"
# key: (v1 analysis dir, v1.2 analysis dir, outcome CSVs, continuation-break dir, group name in that file, name in cross-model files)
M = {
    "orbis2": ("orbis2/analysis-test-v1", "orbis2/analysis-v12-test", ["orbis2/instrument/outcomes/test/outcomes-orbis2-test-k16.csv"],
               "orbis2/continuation-break/orbis2-test", "Orbis2", "Orbis 2"),
    "cosmos3-nano": ("analysis-test-v1", "analysis-v12-test", ["instrument/outcomes/test/outcomes-test-k0-10.csv", "instrument/outcomes/test/outcomes-test-k11-15.csv"],
                     "exploratory/continuation-break-v1", "Cosmos3-Nano", "Cosmos3-Nano"),
    "reworld-video": ("reworld-video/stage2-p84/analysis-test-v1", "reworld-video/stage2-p84/analysis-v12-test",
                      ["reworld-video/stage2-p84/outcomes/test/outcomes-reworld-test-k16.csv"], "reworld-video/continuation-break/reworld-test", "ReWorld-Video", "ReWorld-Video"),
    "drivingworld": ("drivingworld/stage2-p84/analysis-test-v1", "drivingworld/stage2-p84/analysis-v12-test",
                     ["drivingworld/stage2-p84/outcomes/test/outcomes-drivingworld-test-k16.csv"], "drivingworld/continuation-break/drivingworld-test", "DrivingWorld", "DrivingWorld"),
    "cosmos3-edge": ("edge/analysis-test-v1", "edge/analysis-v12-test", ["edge/instrument/outcomes/test/outcomes-edge-test-k0-15.csv"],
                     "exploratory/continuation-break-v1", "Cosmos3-Edge", "Cosmos3-Edge"),
    "epona": ("epona/analysis-test-v1", "epona/analysis-v12-test", ["epona/instrument/outcomes/test/outcomes-epona-test-b16-k16.csv"],
              "epona/continuation-break/epona-test", "Epona", "Epona"),
    "ltx25": ("ltx25d-2s/test/analysis-test-v1", "ltx25d-2s/test/analysis-v12-test", ["ltx25d-2s/test/instrument/outcomes/test/outcomes-ltx25d-test-k16.csv"],
              "ltx25d-2s/continuation-break/ltx25d-2s-test", "LTX2.5-two-stage", "LTX-2.5"),
    "drivelaw-video": ("drivelaw-video/stage2-p84/analysis-test-v1", "drivelaw-video/stage2-p84/analysis-v12-test",
                       ["drivelaw-video/stage2-p84/outcomes/test/outcomes-drivelaw-test-k16.csv"], "drivelaw-video/continuation-break/drivelaw-test", "DriveLaW-Video", "DriveLaW-Video"),
    "vista": ("vista/analysis-test-v1", "vista/analysis-v12-test", ["vista/instrument/outcomes/test/outcomes-vista-test-h0-k16.csv"],
              "vista/continuation-break/vista-test", "Vista", "Vista"),
    "wan21-vace": ("wan21-vace13b/test-v1/analysis-test-v1", "wan21-vace13b/test-v1/analysis-v12-test",
                   ["wan21-vace13b/test-v1/instrument/outcomes/test/outcomes-vace832-test-k16.csv"], "wan21-vace13b/test-v1/continuation-break/t20",
                   "Wan2.1-VACE", "Wan2.1-VACE"),
    "vavim": ("vavim/stage2-p84/analysis-test-v1", "vavim/stage2-p84/analysis-v12-test", ["vavim/stage2-p84/outcomes/test/outcomes-vavim-test-k16.csv"],
              "vavim/continuation-break-2hz/vavim-test", "VaViM", "VaViM"),   # its own 2 Hz step
}
EGO_NAMES = {"Orbis 2": "orbis2", "Epona": "epona", "LTX-2.5": "ltx25", "Vista": "vista", "Wan2.1-VACE": "wan21-vace", "Wan2.2-TI2V-5B": "wan22-ti2v",
             "DriveLaW-Video": "drivelaw-video", "DrivingWorld": "drivingworld", "ReWorld-Video": "reworld-video"}   # SkyReels-V2-DF is not in the paper
POOL = {"lead": "x_4s", "lead_stop": "x_4s", "cutin": "absy_2s", "oncoming": "x_1s", "ego_inter": "ego_dx_0_4s"}
FAM9 = ["lead", "lead_stop", "cutin", "oncoming", "ego_inter", "ped", "startup", "crossing", "rider"]
TRAJ = BD / "cross-model/trajectory-calibration-v1-ext2/run-20261115-v1"
SOURCES = "the frozen TEST analyses, read from $DOB_ROOT/outputs/wm-calibration/bench-v1/"


def js(p):
    return json.load(open(p))


def num(v):
    return "" if v is None else repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)


def write(name, rows, cols, one_row, meanings, notes):
    """$DOB_FIGURE_OUT/<name>.csv plus its description <name>.md (column table generated from `meanings`)."""
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([num(r.get(c)) if not isinstance(r.get(c), str) else r.get(c) for c in cols])
    lines = [f"# {name}.csv", "", f"{one_row} {len(rows):,} rows.", "", "| column | meaning | values or unit |", "|---|---|---|"]
    lines += [f"| {c} | {meanings[c][0]} | {meanings[c][1]} |" for c in cols]
    lines += ["", f"- **Produced by:** `python {SCRIPT}`, "
              f"run on {TODAY}, from {SOURCES}."] + [f"- {n}" for n in notes]
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"data/{name}.csv: {len(rows)} rows")


def case_ranks():
    rows = []
    for key, (a1, a12, csvs, *_r) in M.items():
        rng = np.random.default_rng(2026); vals = collections.defaultdict(dict)
        for p in csvs:
            for r in csv.DictReader(open(BD / p)):
                if r["family"] in POOL and r["outcome"] == POOL[r["family"]] and r["kind"] == "primary" and r["scorable"] == "1" and r["value"]:
                    vals[(r["window"], r["family"])].setdefault(r["sample"], float(r["value"]))
        for (w, f), d in sorted(vals.items()):
            xs = [d[str(k)] for k in range(16) if str(k) in d]
            if "real" not in d or len(xs) < 16:
                continue
            y = d["real"]; below = sum(x < y for x in xs); ties = sum(x == y for x in xs)
            rank = below + int(rng.integers(0, ties + 1)) if ties else below
            rows.append(dict(method=key, window_id=w, family=f, quantity=POOL[f], real_value=y, rank=rank, n_below=below, n_ties=ties,
                             sample_min=min(xs), sample_median=float(np.median(xs)), sample_max=max(xs)))
    write("case_ranks", rows, ["method", "window_id", "family", "quantity", "real_value", "rank", "n_below", "n_ties", "sample_min", "sample_median", "sample_max"],
          "One row is one test case: one model, one history window and one scored family, in which the real reading and all 16 sampled readings of the family's primary quantity are valid (the case pool of the pre-registered test H2).",
          dict(method=("the model", "keys of `methods.csv`"),
               window_id=("the TEST history window (Argoverse 2 log prefix and anchor time)", "e.g. 9947fb6e-t6.5; the same id in every file"),
               family=("the scenario family", "lead, lead_stop, cutin, oncoming, ego_inter"),
               quantity=("the family's primary quantity", "x_4s (lead gap at 4 s, m), absy_2s (cut-in lateral offset at 2 s, m), x_1s (oncoming position at 1 s, m), ego_dx_0_4s (ego advance over 0-4 s, m)"),
               real_value=("the reading of the real future (through the model's own autoencoder)", "m"),
               rank=("rank of the real reading among the 16 samples: the number of samples below it, ties split at random (seed 2026)", "0 (below all) to 16 (above all)"),
               n_below=("samples strictly below the real reading", "0-16"), n_ties=("samples equal to the real reading", "0-16; usually 0"),
               sample_min=("smallest of the 16 sampled readings", "m"), sample_median=("median of the 16 sampled readings", "m"), sample_max=("largest of the 16 sampled readings", "m")),
          ["**Filters:** cases whose real reading or any sampled reading is not valid (lost target, failed t0 match, reliability rule) are left out, as in the frozen analysis. DrivingWorld has no ego_inter rows (its ego family is not scored).",
           "**Paired:** no. Each model has its own retained cases; the windows overlap but differ per model.",
           "**Derived values:** outside = rank is 0 or 16; window-weighted shares (as H2) = mean over windows of the mean over that window's families. Expected outside share under calibration: 2/17 = 11.8 %; central-interval coverage of [x(j), x(17-j)]: (17-2j)/17, observed as j <= rank <= 16-j.",
           "**Empty cells:** none."])


def confirmatory():
    rows = []
    for key, (a1, *_r) in M.items():
        r = js(BD / a1 / "results.json"); c = r["confirmatory"]; two = BD / a1 / "confirmatory-two-sided.json"; d = js(two) if two.exists() else None
        h1, h2 = c["H1"], c["H2"]
        rows.append(dict(method=key, h2_outside_share=h2["T"], h2_ci_low=h2["boot_ci95"][0], h2_ci_high=h2["boot_ci95"][1], h2_n_windows=h2["n_windows"],
                         h2_p_holm=(d["H2"]["p_holm"] if d else h2["p_holm"]), h2_confirmed=int(bool(d["H2"]["confirmed"] if d else h2["confirmed"])),
                         h1_mean_pit_minus_half=h1["mean_pit_minus_half"], h1_ci_low=h1["mean_pit_ci95"][0], h1_ci_high=h1["mean_pit_ci95"][1], h1_n=h1["n"],
                         h1_slope_z=h1["slope_z"], h1_p_holm=(d["H1"]["p_holm"] if d else h1["p_holm"]), h1_confirmed=int(bool(d["H1"]["confirmed"] if d else h1["confirmed"])),
                         h1_sidedness="two-sided" if d else "one-sided"))
    cols = ["method", "h2_outside_share", "h2_ci_low", "h2_ci_high", "h2_n_windows", "h2_p_holm", "h2_confirmed", "h1_mean_pit_minus_half", "h1_ci_low",
            "h1_ci_high", "h1_n", "h1_slope_z", "h1_p_holm", "h1_confirmed", "h1_sidedness"]
    write("confirmatory", rows, cols, "One row is one model: the pre-registered confirmatory statistics as frozen (these are summaries; print them as they are).",
          dict(method=("the model", "keys of `methods.csv`"),
               h2_outside_share=("H2: window-weighted share of cases with the real outcome outside all 16 samples", "share 0-1; calibrated 2/17 = 0.118"),
               h2_ci_low=("95 % window-bootstrap interval of H2, lower end", "share 0-1"), h2_ci_high=("95 % window-bootstrap interval of H2, upper end", "share 0-1"),
               h2_n_windows=("windows in the H2 statistic", "count"), h2_p_holm=("Holm-adjusted p of H2 (one-sided, permutation null, B = 100,000)", "probability"),
               h2_confirmed=("H2 confirmed", "1 = yes, 0 = no"),
               h1_mean_pit_minus_half=("H1: mean PIT - 1/2 of the lead gap at 4 s (negative: generated gaps too large; positive: too small)", "-0.5 to 0.5; calibrated 0"),
               h1_ci_low=("95 % bootstrap interval of H1, lower end", "-0.5 to 0.5"), h1_ci_high=("95 % bootstrap interval of H1, upper end", "-0.5 to 0.5"),
               h1_n=("lead cases in H1", "count"), h1_slope_z=("rank-slope z statistic of H1", "z"),
               h1_p_holm=("Holm-adjusted p of H1 (two-sided, except Cosmos3-Nano: one-sided as registered)", "probability"),
               h1_confirmed=("H1 confirmed (Holm and the bootstrap interval excludes 0 on the side of z)", "1 = yes, 0 = no"),
               h1_sidedness=("sidedness of the registered H1 test", "two-sided or one-sided")),
          ["**Filters:** none; the 11 models with frozen TEST analyses (VaViM scored on lead and lead_stop only).",
           "**Paired:** no.", "**Empty cells:** none.",
           "**Note:** the H2 share recomputed from case_ranks.csv agrees within about 1 point (random tie splits); report these frozen values."])


def family_stats():
    rows = []
    for key, (a1, *_r) in M.items():
        r = js(BD / a1 / "results.json")
        for f in FAM9:
            fr, sc = r["families_results"][f], r["scorability"][f]; s = fr["stats"]; n = s.get("n", 0); cl = fr.get("baselines", {}).get("baselines", {}).get("climatology", {}) if n else {}
            sig = r.get("secondary_bh", {}).get("mean_pit", {}).get(f, {}).get("significant_q_0_10") if fr["role"] == "scored" else None
            rows.append(dict(method=key, family=f, quantity=fr["outcome"], role=fr["role"], scored=int(fr["role"] == "scored" and n > 0), n=n,
                             mean_pit_minus_half=s.get("mean_pit_minus_half") if n else None, mean_pit_ci_low=s["mean_pit_ci95"][0] if n else None,
                             mean_pit_ci_high=s["mean_pit_ci95"][1] if n else None, above_all_share=s["above_all"]["share"] if n else None,
                             below_all_share=s["below_all"]["share"] if n else None, crpss_climatology=cl.get("crpss"),
                             crpss_ci_low=cl.get("crpss_ci95", [None, None])[0], crpss_ci_high=cl.get("crpss_ci95", [None, None])[1],
                             significant_bh=None if sig is None else int(bool(sig)), kept=sc["kept"], windows_with_outcome=sc["windows_with_family_outcome"]))
    cols = ["method", "family", "quantity", "role", "scored", "n", "mean_pit_minus_half", "mean_pit_ci_low", "mean_pit_ci_high", "above_all_share", "below_all_share",
            "crpss_climatology", "crpss_ci_low", "crpss_ci_high", "significant_bh", "kept", "windows_with_outcome"]
    write("family_stats", rows, cols, "One row is one model and one scenario family (primary quantity): the frozen per-family statistics (summaries over the family's test cases).",
          dict(method=("the model", "keys of `methods.csv`"), family=("the scenario family", "lead, lead_stop, cutin, oncoming, ego_inter, ped, startup, crossing, rider"),
               quantity=("its primary quantity", "x_4s, absy_2s, x_1s, ego_dx_0_4s, x_6s"),
               role=("role in the registered analysis", "scored (enters H2 and the scores) or descriptive"),
               scored=("scored and has cases for this model", "1 or 0 (DrivingWorld ego_inter: 0, its odometry gate failed)"),
               n=("cases with the real and all 16 readings valid", "count"),
               mean_pit_minus_half=("mean PIT - 1/2 (positive: real above the samples, generated too small)", "-0.5 to 0.5; empty when n is 0"),
               mean_pit_ci_low=("95 % bootstrap interval, lower end", "empty when n is 0"), mean_pit_ci_high=("95 % bootstrap interval, upper end", "empty when n is 0"),
               above_all_share=("share of cases with the real reading above all 16", "share 0-1; empty when n is 0"),
               below_all_share=("share of cases with the real reading below all 16", "share 0-1; empty when n is 0"),
               crpss_climatology=("CRPS skill against climatology (1 - CRPS_model / CRPS_climatology)", "unitless; 0 = no better than climatology; empty when n is 0"),
               crpss_ci_low=("95 % bootstrap interval of the skill, lower end", "empty when n is 0"), crpss_ci_high=("95 % bootstrap interval of the skill, upper end", "empty when n is 0"),
               significant_bh=("mean PIT - 1/2 significant at BH q <= 0.10 across the scored families", "1 or 0; empty for descriptive families"),
               kept=("windows kept (real and all 16 valid)", "count"), windows_with_outcome=("windows of the family", "count")),
          ["**Filters:** none.", "**Paired:** no.", "**Empty cells:** the statistic columns are empty when n is 0 (no valid case), and significant_bh is empty for descriptive families.",
           "**Derived values:** |PIT - 1/2| average (table column) = mean over the scored families (scored = 1) of abs(mean_pit_minus_half), x 100; informativeness (point) = mean of crpss_climatology over the same families; scorable share = sum(kept) / sum(windows_with_outcome) over the scored families."])


def quantity_stats():
    rows = []
    for key, (a1, a12, *_r) in M.items():
        v = js(BD / a12 / "results-v12.json")
        for lst, tested in ((v["tested"], 1), (v["descriptive"], 0)):
            for x in lst:
                s = x.get("stats") or {}; sc = x.get("scorability") or {}; q = (x.get("bh_q") or {}).get("mean_pit")
                rows.append(dict(method=key, family=x["family"], quantity=x["quantity"], verdict=x["verdict"], tested=tested, binary=int(bool(x.get("binary"))),
                                 n=s.get("n"), mean_pit_minus_half=s.get("mean_pit_minus_half"),
                                 mean_pit_ci_low=(s.get("mean_pit_ci95") or [None, None])[0], mean_pit_ci_high=(s.get("mean_pit_ci95") or [None, None])[1],
                                 q_mean_pit=q if tested else None, significant_bh=(int(q <= 0.10) if (tested and q is not None) else None),
                                 kept=sc.get("kept"), windows_with_quantity=sc.get("windows_with_quantity")))
    cols = ["method", "family", "quantity", "verdict", "tested", "binary", "n", "mean_pit_minus_half", "mean_pit_ci_low", "mean_pit_ci_high", "q_mean_pit",
            "significant_bh", "kept", "windows_with_quantity"]
    write("quantity_stats", rows, cols, "One row is one model and one secondary physical quantity (v1.2 panel): its DEV reliability verdict and, when tested, the frozen TEST calibration statistics.",
          dict(method=("the model", "keys of `methods.csv`"), family=("the family", "lead, lead_stop, cutin, oncoming, ego, startup (ego: the ego family of v1.2)"),
               quantity=("the quantity", "e.g. thw_4s (time headway at 4 s), pet_follow, pet_t0 (post-encroachment time), closing_speed_4s, lat_speed_1_2s, exit_time, ego_dx_0_2s, ego_speed_4s, ego_max_decel_0_4s, ego_stop_4s"),
               verdict=("reliability verdict fixed on DEV before the test", "reliable, not reliable, not informative, insufficient"),
               tested=("tested on TEST (verdict reliable)", "1 or 0"), binary=("binary quantity (analysed as a rate, not ranked)", "1 or 0"),
               n=("test cases", "count; empty for binary quantities and for quantities without statistics"),
               mean_pit_minus_half=("mean PIT - 1/2", "-0.5 to 0.5; empty for binary quantities and when not computed"),
               mean_pit_ci_low=("95 % bootstrap interval, lower end", "empty when not computed"), mean_pit_ci_high=("95 % bootstrap interval, upper end", "empty when not computed"),
               q_mean_pit=("BH q of the mean-PIT test, within each quantity across families", "probability; empty when not tested"),
               significant_bh=("q_mean_pit <= 0.10", "1 or 0; empty when not tested"),
               kept=("windows kept", "count; empty when not computed"), windows_with_quantity=("windows with the quantity", "count; empty when not computed")),
          ["**Filters:** none.", "**Paired:** no.", "**Empty cells:** see the columns; descriptive quantities may carry statistics that are reported but never tested.",
           "**Caveat:** DrivingWorld's ego quantities are tested by their own DEV validation although its reference clips failed the odometry gate; figures mark them with a dagger."])


def trajectory():
    T = js(TRAJ / "results.json")["models"]; C = js(TRAJ / "addendum-1-climatology.json")["models"]; hz, sm, sk = [], [], []
    for key, (*_r, xname) in M.items():
        t = T[xname]; pc = t["positive_control"]["pooled"]; p = t["result"]["pooled"]
        sm.append(dict(method=key, least_central_share=p["bd_least_central"]["T"], least_central_ci_low=p["bd_least_central"]["ci95"][0],
                       least_central_ci_high=p["bd_least_central"]["ci95"][1], n_windows=p["bd_least_central"]["n_windows"], avg_outside_share=p["avg_outside"]["T"],
                       avg_outside_ci_low=p["avg_outside"]["ci95"][0], avg_outside_ci_high=p["avg_outside"]["ci95"][1],
                       control_least_central_share=pc["bd_least_central"], control_avg_outside_share=pc["avg_outside"]))
        for f, o in t["result"]["families"].items():
            for i, ts in enumerate(o["grid"]):
                hz.append(dict(method=key, family=f, t_s=ts, n=o["n"], mean_pit_minus_half=o["mean_pit_t"][i], mean_pit_ci_low=o["mean_pit_t_ci95"][0][i],
                               mean_pit_ci_high=o["mean_pit_t_ci95"][1][i], outside_share=o["outside_t"][i]))
            c = C[xname][f]
            sk.append(dict(method=key, family=f, n=o["n"], least_central_share=o["bd_least_central"], avg_outside_share=o["avg_outside"],
                           es_skill_climatology=c["skill"], es_skill_climatology_ci_low=c["ci95"][0], es_skill_climatology_ci_high=c["ci95"][1],
                           es_skill_cv=o.get("es_skill_vs_cv"), es_skill_cv_ci_low=(o.get("es_skill_ci95") or [None, None])[0], es_skill_cv_ci_high=(o.get("es_skill_ci95") or [None, None])[1]))
    common = ["**Produced from** the trajectory analysis (pre-specified, exploratory; extension 2, run-20261115-v1, covers all eleven models and reproduces the nine of extension 1 exactly).",
              "**Paired:** no.", "**Filters:** the cases of the scalar analysis; DrivingWorld has no ego_inter family."]
    write("trajectory_horizon", hz, ["method", "family", "t_s", "n", "mean_pit_minus_half", "mean_pit_ci_low", "mean_pit_ci_high", "outside_share"],
          "One row is one model, one family and one time after t0: calibration of the reading at that time (the whole-trajectory analysis).",
          dict(method=("the model", "keys of `methods.csv`"), family=("the family", "lead, lead_stop, cutin, oncoming, ego_inter"), t_s=("time after t0", "s"),
               n=("cases", "count"), mean_pit_minus_half=("mean PIT - 1/2 of the real value at this time", "-0.5 to 0.5"),
               mean_pit_ci_low=("95 % window-bootstrap interval, lower end", "-0.5 to 0.5"), mean_pit_ci_high=("95 % window-bootstrap interval, upper end", "-0.5 to 0.5"),
               outside_share=("share of cases with the real value outside all 16 samples at this time", "share 0-1; calibrated 2/17")),
          common + ["**Empty cells:** none."])
    write("trajectory_summary", sm, ["method", "least_central_share", "least_central_ci_low", "least_central_ci_high", "n_windows", "avg_outside_share",
                                      "avg_outside_ci_low", "avg_outside_ci_high", "control_least_central_share", "control_avg_outside_share"],
          "One row is one model: whole-trajectory calibration pooled over its scored families (window-weighted, as H2).",
          dict(method=("the model", "keys of `methods.csv`"),
               least_central_share=("share of cases whose real trajectory is the least central of the 17 by band depth", "share 0-1; calibrated 1/17 = 0.059"),
               least_central_ci_low=("95 % window-bootstrap interval, lower end", "share 0-1"), least_central_ci_high=("95 % window-bootstrap interval, upper end", "share 0-1"),
               n_windows=("windows", "count"), avg_outside_share=("share of cases whose average rank is the lowest or highest", "share 0-1; calibrated 2/17"),
               avg_outside_ci_low=("95 % interval, lower end", "share 0-1"), avg_outside_ci_high=("95 % interval, upper end", "share 0-1"),
               control_least_central_share=("positive control: sample 0 as pseudo-observation against the other 15", "share 0-1; reference 1/16"),
               control_avg_outside_share=("positive control, average-rank outside share", "share 0-1; reference 2/16")),
          common + ["**Empty cells:** none."])
    write("trajectory_skill", sk, ["method", "family", "n", "least_central_share", "avg_outside_share", "es_skill_climatology", "es_skill_climatology_ci_low",
                                    "es_skill_climatology_ci_high", "es_skill_cv", "es_skill_cv_ci_low", "es_skill_cv_ci_high"],
          "One row is one model and one family: whole-trajectory calibration and energy-score skill.",
          dict(method=("the model", "keys of `methods.csv`"), family=("the family", "lead, lead_stop, cutin, oncoming, ego_inter"), n=("cases", "count"),
               least_central_share=("share of cases with the least central real trajectory", "share 0-1"), avg_outside_share=("average-rank outside share", "share 0-1"),
               es_skill_climatology=("energy-score skill against climatology (DEV trajectories of the same family)", "unitless; 0 = no better than climatology"),
               es_skill_climatology_ci_low=("95 % window-bootstrap interval, lower end", "unitless"), es_skill_climatology_ci_high=("95 % interval, upper end", "unitless"),
               es_skill_cv=("energy-score skill against constant-velocity extrapolation of the real history", "unitless; empty for oncoming (no reliable history fit)"),
               es_skill_cv_ci_low=("95 % interval, lower end", "empty for oncoming"), es_skill_cv_ci_high=("95 % interval, upper end", "empty for oncoming")),
          common + ["**Empty cells:** es_skill_cv and its interval are empty for the oncoming family.",
                    "**Derived values:** informativeness (trajectory) = mean of es_skill_climatology over the model's scored families."])


def ego_motion():
    rows = []; iv = {"R_0_0p4": (0.0, 0.4), "R_0p4_0p8": (0.4, 0.8), "R_0p8_2": (0.8, 2.0), "R_2_4": (2.0, 4.0)}
    for r in csv.DictReader(open(BD / "cross-model/ego-motion-prior-v1/run-20261114-new3-v1/primary.csv")):
        if r["metric"] in iv and r["model"] in EGO_NAMES:
            ci = json.loads(r["CI95"]) if r["CI95"].startswith("[") else [None, None]
            rows.append(dict(method=EGO_NAMES[r["model"]], source=r["source"], interval=r["metric"], t_start_s=iv[r["metric"]][0], t_end_s=iv[r["metric"]][1],
                             speed_ratio=float(r["estimate"]), ci_low=ci[0], ci_high=ci[1], n=int(r["n_windows"]), n_clips=int(r["n_clips"])))
    write("ego_motion_dev", rows, ["method", "source", "interval", "t_start_s", "t_end_s", "speed_ratio", "ci_low", "ci_high", "n", "n_clips"],
          "One row is one model, one source (its generated clips or its real reference clips) and one time interval after t0: the generated (or real) ego speed relative to the speed at t0 (DEV, exploratory).",
          dict(method=("the model (for source = reference: the model whose reference clips these are)", "keys of `methods.csv`"),
               source=("generated clips (samples s0-s3) or the model's real reference clips", "generated, reference"),
               interval=("time interval after t0", "R_0_0p4, R_0p4_0p8, R_0p8_2, R_2_4"), t_start_s=("interval start", "s"), t_end_s=("interval end", "s"),
               speed_ratio=("ego displacement per 0.4 s in the interval / the last history group's displacement; median within a window, then over windows", "ratio; 1 = keeps the speed at t0"),
               ci_low=("95 % window-bootstrap interval, lower end", "ratio"), ci_high=("95 % window-bootstrap interval, upper end", "ratio"),
               n=("windows", "count"), n_clips=("clips", "count")),
          ["**Filters:** SkyReels-V2-DF, which is not one of the eleven models of the paper, is left out. DEV windows only; no TEST clip.",
           "**Derived values:** the real curve = median over models of the source = reference rows, per interval (about 1.00, 0.98, 0.90, 0.66).",
           "**Paired:** no (each model's own eligible windows).", "**Empty cells:** none.",
           "**Caveat:** DrivingWorld's reference clips failed the odometry gate; mark its curve."])


def property_scores():
    S = js(OUT / "property_scores.json"); inv = {v[5]: k for k, v in M.items()}; rows = []
    for name, v in S.items():
        rows.append(dict(method=inv[name], scorable_share=v["scorable"], jump_share=v["jump"], validity=v["Validity"], coverage=v["Coverage"], centering=v["Centering"],
                         informativeness=v["Informativeness"], informativeness_point=v["informativeness_point"], informativeness_traj=v["informativeness_traj"], summary=v["Summary"]))
    lim = js(OUT / "radar_axis_ranges.json")
    write("property_scores", rows, ["method", "scorable_share", "jump_share", "validity", "coverage", "centering", "informativeness", "informativeness_point",
                                     "informativeness_traj", "summary"],
          "One row is one model: the four descriptive property scores (higher is better, 1 is ideal) and their mean.",
          dict(method=("the model", "keys of `methods.csv`"), scorable_share=("share of (history, family) cases of the five scored families kept; a family the model's own reconstruction cannot be read for counts as unusable (DrivingWorld ego; VaViM cut-in, oncoming, ego)", "share 0-1"),
               jump_share=("share of generated clips whose first frame jumps (J > 3); VaViM on its own 2 Hz step", "share 0-1"), validity=("scorable_share x (1 - jump_share)", "0-1"),
               coverage=("1/2 [c(H2, 2/17) + c(least central share, 1/17)], c(T, p) = 1 - |T - p| / (1 - p)", "0-1"),
               centering=("1 - 2 x mean |PIT - 1/2| over the scored families", "0-1"), informativeness=("clip((point + trajectory skill) / 2, 0, 1)", "0-1"),
               informativeness_point=("mean CRPS skill against climatology over the scored families", "unitless"),
               informativeness_traj=("mean energy-score skill against climatology over the scored families", "unitless"),
               summary=("mean of validity, coverage, centering and informativeness", "0-1")),
          ["**Produced from** property_scores.json, computed by scripts/make_overview_figures.py from the files above; DrivingWorld uses its four scored families.",
           f"**Radar axis ranges used so far** (each axis scaled from its lowest to its highest model value, rounded outwards to 0.05): {json.dumps(lim)}.",
           "**Paired:** no.", "**Empty cells:** none."])


def continuation_break():
    rows = []
    for key, (*_r, cb, grp, _x) in M.items():
        if cb is None: continue
        g = js(BD / cb / "continuation-break.json")["summary"]["groups"]
        for src, name in (("generated", grp), ("reference", "real reference")):
            x = g[name]
            rows.append(dict(method=key, source=src, n=x["n"], j_median=x["J_median"], boundary_break_share=x["boundary_break"], strong_break_share=x["strong_break"]))
    write("continuation_break", rows, ["method", "source", "n", "j_median", "boundary_break_share", "strong_break_share"],
          "One row is one model and one source (its generated TEST clips, or the real reference clips of the same file): how much the first generated frame jumps.",
          dict(method=("the model", "keys of `methods.csv`"), source=("generated or reference", "generated, reference"), n=("clips", "count"),
               j_median=("median J = mean|f[t0+1] - f[t0]| / mean history step over the last second", "ratio; about 1 for a smooth continuation"),
               boundary_break_share=("share of clips with J > 3", "share 0-1"), strong_break_share=("share of clips with J > 5", "share 0-1")),
          ["**Filters:** none. VaViM is measured on its own 2 Hz grid (continuation_break_2hz.py): the first model frame (t0 + 0.5 s) against t0, relative to its 0.5 s history steps, because its 10 Hz canvas repeats the t0 frame until then. Cosmos3-Nano and Cosmos3-Edge share one file, so their reference rows are the same clips.", "**Paired:** no.", "**Empty cells:** none."])


def teaser(frames):
    T = OUT / "teaser"; T.mkdir(parents=True, exist_ok=True)
    case = js(ROOT / "data/figure1_case.json")
    rows = [dict(method="orbis2", window_id=case["window"], video="real", sample_index="", value_m=case["real"]["gap_m"], shown=1)]
    for k in range(16):
        s = case["samples"][f"sample_{k + 1:02d}"]
        rows.append(dict(method="orbis2", window_id=case["window"], video="imagined", sample_index=str(k + 1), value_m=s["gap_m"], shown=int(f"sample_{k + 1:02d}" in case["shown"])))
    with open(T / "teaser_readings.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "window_id", "video", "sample_index", "value_m", "shown"])
        for r in rows: w.writerow([r["method"], r["window_id"], r["video"], r["sample_index"], repr(float(r["value_m"])), r["shown"]])
    boxes = [("real_4s.png", case["real"]["box_xyxy_px"], "real future")] + [(f"sample{k[7:]}_4s.png", case["samples"][k]["box_xyxy_px"], "imagined future #" + str(int(k[7:]))) for k in case["shown"]]
    with open(T / "teaser_boxes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "x0", "y0", "x1", "y1", "label"])
        for img, bx, lab in boxes: w.writerow([img, *[repr(float(v)) for v in bx], lab])
    prod = f"- **Produced by:** `python {SCRIPT} --frames`, run on {TODAY}, from the frozen measurement files of Orbis 2 (TEST window {case['window']})."
    (T / "teaser_readings.md").write_text("\n".join([
        "# teaser_readings.csv", "", "One row is one video of the Figure 1 window: the real future or one of the 16 imagined futures. 17 rows.", "",
        "| column | meaning | values or unit |", "|---|---|---|",
        "| method | the model | keys of `methods.csv` (orbis2) |",
        f"| window_id | the TEST window | {case['window']} |",
        "| video | the real future (through the model's own autoencoder) or an imagined future | real, imagined |",
        "| sample_index | which imagined future | 1-16; empty for the real future |",
        "| value_m | the lead gap at t0 + 4 s: the frozen instrument reading (median over three trackers) | m |",
        "| shown | whether its image is shown in the figure | 1 or 0 |", "", prod,
        "- **Filters:** none.", "- **Paired:** not applicable.", "- **Empty cells:** sample_index is empty for the real future."]) + "\n")
    (T / "teaser_boxes.md").write_text("\n".join([
        "# teaser_boxes.csv", "", "One row is the instrument's lead-vehicle box in one image of `data/teaser/`: at t0 + 4 s in the real and imagined futures, and at -2 s, -1 s and t0 in the history frames. 7 rows.", "",
        "| column | meaning | values or unit |", "|---|---|---|",
        "| image | the image file in data/teaser/ | real_4s.png, sample15_4s.png, sample08_4s.png, sample11_4s.png; with BENCH_DIR set also history_m2s.png, history_m1s.png, history_t0.png |",
        "| x0 | left edge | pixels from the left of the 512 x 288 image |", "| y0 | top edge | pixels from the top |",
        "| x1 | right edge | pixels |", "| y1 | bottom edge | pixels |", "| label | what the image is | real future, imagined future #N, lead vehicle (history frames) |", "", prod,
        "- **Filters:** none.", "- **Paired:** not applicable.", "- **Empty cells:** none."]) + "\n")
    if os.environ.get("BENCH_DIR"):              # BENCH_DIR set: lead-gap curves and history boxes from the frozen measurement files
        B = os.environ["BENCH_DIR"]; W = case["window"]
        gen = js(f"{B}/orbis2/instrument/measure-test/measure-orbis2-test-k16.json")["windows"][W]
        ref = js(f"{B}/orbis2/instrument/measure-test/measure-orbis2-test-recon.json")["windows"][W]["real"]["families"]["lead"]
        cur = [("real", "", x["t"], x["x"]) for x in sorted(ref["history"]["series"], key=lambda x: x["t"]) if x["t"] < 0 and x.get("x") is not None]
        cur += [("real", "", x["t"], x["x"]) for x in ref["ensemble"]["series"] if x["t"] <= 4.0 + 1e-9 and x.get("x") is not None]
        for k in range(16):
            cur += [("imagined", str(k + 1), x["t"], x["x"]) for x in gen[str(k)]["families"]["lead"]["ensemble"]["series"] if x["t"] <= 4.0 + 1e-9 and x.get("x") is not None]
        with open(T / "teaser_curves.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["method", "window_id", "video", "sample_index", "t_s", "value_m"])
            for v, k, t, x in cur: w.writerow(["orbis2", W, v, k, repr(round(float(t), 3)), repr(float(x))])
        (T / "teaser_curves.md").write_text("\n".join([
            "# teaser_curves.csv", "", f"One row is the lead gap of one video of the Figure 1 window at one time: the real video from t0 - 2 s to t0 + 4 s, and each of the 16 imagined futures from t0 to t0 + 4 s (the validated horizon of the lead gap). {len(cur)} rows.", "",
            "| column | meaning | values or unit |", "|---|---|---|",
            "| method | the model | keys of `methods.csv` (orbis2) |", f"| window_id | the TEST window | {W} |",
            "| video | the real video (history, then the real future through the model's own autoencoder) or an imagined future | real, imagined |",
            "| sample_index | which imagined future | 1-16; empty for the real video |",
            "| t_s | time after t0 (negative: the 2 s history) | s, every 0.1 s |",
            "| value_m | the lead gap: the frozen instrument reading (median over three trackers, each carrying its last reading) | m |", "",
            f"- **Produced by:** `python {SCRIPT}` with BENCH_DIR set, run on {TODAY}, from the frozen measurement files of Orbis 2.",
            "- **Filters:** times after t0 + 4 s are left out (the lead gap is validated up to 4 s). The imagined futures share the real history, so they start at t0.",
            "- **Paired:** not applicable.", "- **Empty cells:** sample_index is empty for the real video.",
            "- **Check:** at t_s = 4.0 the values equal teaser_readings.csv (real 7.881 m; imagined 12.475-18.569 m)."]) + "\n")
        hb = {round(x["t"], 1): x["box"] for x in ref["history"]["series"] if x.get("box")}
        with open(T / "teaser_boxes.csv", "a", newline="") as f:
            w = csv.writer(f)
            for img, t in (("history_m2s.png", -2.0), ("history_m1s.png", -1.0), ("history_t0.png", 0.0)):
                w.writerow([img, *[repr(float(v)) for v in hb[t]], "lead vehicle"])
    if frames:
        B = os.environ["BENCH_DIR"]; W = case["window"]
        def grab(p, i, out):
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", p, "-vf", f"select=eq(n\\,{i})", "-vframes", "1", str(T / out)], check=True)
        for i, out in ((0, "history_m2s.png"), (10, "history_m1s.png"), (20, "history_t0.png")):
            grab(f"{B}/windows-test/{W}/canvas/real.mp4", i, out)
        grab(f"{B}/orbis2/runs/orbis2-test-recon/native/{W}-recon/vision.mp4", 60, "real_4s.png")
        for k in case["shown"]:
            grab(f"{B}/orbis2/runs/orbis2-test-k16/native/{W}-s{int(k[7:]) - 1}/vision.mp4", 60, f"sample{k[7:]}_4s.png")
    print("data/teaser: readings, boxes" + (", 7 frames" if frames else ""))


PIPE_W = "fb1e1e24-t6.0"     # example history of the overview figure (Fig. 2): Orbis 2 scores its four primary quantities; ranks 6, 4, 3, 7
PIPE_ROLES = (("lead", 1), ("cutin", 2), ("oncoming", 3))                      # (family = role, track_id)
PIPE_Q = (("lead", "x_4s"), ("cutin", "absy_2s"), ("oncoming", "x_1s"), ("ego_inter", "ego_dx_0_4s"))
PIPE_SEQ = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)    # times of the future frame stacks: every 0.5 s to 4 s
ttag = lambda t: f"{t:g}".replace(".", "p") + "s"       # 0.5 -> 0p5s, 1.0 -> 1s


def pipeline():
    """$DOB_FIGURE_OUT/pipeline/: frames, overlays and readings of one TEST window for the overview figure (needs the rendered
    windows, model clips and measurement files)."""
    import cv2
    B = str(BD); W = PIPE_W; P = OUT / "pipeline"; P.mkdir(parents=True, exist_ok=True)
    meas = js(f"{B}/orbis2/instrument/measure-test/measure-orbis2-test-k16.json"); cfg = meas["config"]; gen = meas["windows"][W]
    ref = js(f"{B}/orbis2/instrument/measure-test/measure-orbis2-test-recon.json")["windows"][W]["real"]
    win = js(f"{B}/windows-test/{W}/window.json")
    clips = {0: ref, **{k + 1: gen[str(k)] for k in range(16)}}                 # sample_index 0: the real reference clip
    out = lambda k: "real" if k == 0 else "imagined"

    # readings: the frozen outcome file, checked against data/case_ranks.csv
    val = collections.defaultdict(dict)
    with open(f"{B}/orbis2/instrument/outcomes/test/outcomes-orbis2-test-k16.csv") as f:
        for r in csv.DictReader(f):
            if r["window"] == W and (r["family"], r["outcome"]) in PIPE_Q:
                val[(r["family"], r["outcome"])][0 if r["sample"] == "real" else int(r["sample"]) + 1] = float(r["value"])
    with open(OUT / "case_ranks.csv") as f:
        cr = {(r["family"], r["quantity"]): r for r in csv.DictReader(f) if r["method"] == "orbis2" and r["window_id"] == W}
    for q in PIPE_Q:
        v = val[q]; assert len(v) == 17, q
        assert (sum(v[k] < v[0] for k in range(1, 17)), sum(v[k] == v[0] for k in range(1, 17))) == (int(cr[q]["n_below"]), int(cr[q]["n_ties"])), q
        assert abs(v[0] - float(cr[q]["real_value"])) < 1e-6, q
    with open(P / "readings.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "video", "sample_index", "value_m"])
        for fam, q in PIPE_Q:
            for k in range(17): w.writerow([q, out(k), k, repr(val[(fam, q)][k])])

    # the median imagined future (lead gap at 4 s), and the time near 1 s at which the most targets are followed
    lead = val[("lead", "x_4s")]; med = float(np.median([lead[k] for k in range(1, 17)]))
    kmed = min(range(1, 17), key=lambda k: (abs(lead[k] - med), k))
    ens = lambda c, fam: {round(x["t"], 1): x for x in c["families"][fam]["ensemble"]["series"]}
    seen = lambda c, fam, t: bool((ens(c, fam).get(t) or {}).get("followed")) and bool(ens(c, fam)[t].get("box"))
    cand = [round(float(t), 1) for t in np.arange(0.1, 2.01, 0.1)]
    T = max(cand, key=lambda t: (sum(seen(c, fam, t) for c in (ref, clips[kmed]) for fam, _ in PIPE_ROLES), -abs(t - 1.0), -t))
    tag = ttag(T)
    nT = sum(seen(c, fam, T) for c in (ref, clips[kmed]) for fam, _ in PIPE_ROLES)
    seq = sorted(set(PIPE_SEQ) | {T})

    # frames (canvas frame = 20 + 10 t); a time with no frame in a clip is left out and listed
    missing = []
    def grab(p, i, name):
        (P / name).unlink(missing_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(p), "-vf", f"select=eq(n\\,{i})", "-vframes", "1", str(P / name)], check=True)
        if not (P / name).exists(): missing.append(name)
    real = f"{B}/windows-test/{W}/canvas/real.mp4"; recon = f"{B}/orbis2/runs/orbis2-test-recon/native/{W}-recon/vision.mp4"
    smp = lambda k: f"{B}/orbis2/runs/orbis2-test-k16/native/{W}-s{k - 1}/vision.mp4"
    for i, name in ((0, "history_m2s.png"), (10, "history_m1s.png"), (20, "history_t0.png")):
        grab(real, i, name)
    for t in seq:                                   # the three future stacks: raw real, real reference, median imagined future
        f = 20 + int(round(10 * t))
        grab(real, f, f"real_raw_{ttag(t)}.png"); grab(recon, f, f"real_ref_{ttag(t)}.png"); grab(smp(kmed), f, f"sample_med_{ttag(t)}.png")
    for k in range(1, 17):
        grab(smp(k), 60, f"sample{k:02d}_4s.png")
    ltx = js(f"{B}/ltx25d-2s/test/detections/ltx25d-test-k16/{W}-s0.json"); assert ltx["mapping"] is None and ltx["canvas"] == [512, 288]
    grab(ltx["video"], 60, "type_general_4s.png")
    nano = js(f"{B}/detections/test-k0-10/{W}-s0.json"); m = nano["mapping"]
    grab(nano["video"], 60, "_nano_native.png")
    img = cv2.imread(str(P / "_nano_native.png"))[m["crop_rows"][0]:m["crop_rows"][1]]
    cv2.imwrite(str(P / "type_physical_4s.png"), cv2.resize(img, tuple(m["resize"]), interpolation=cv2.INTER_CUBIC)); (P / "_nano_native.png").unlink()
    pngs = sorted(x.name for x in P.glob("*.png"))
    for x in pngs:
        assert cv2.imread(str(P / x)).shape[:2] == (288, 512), x

    # boxes: the instrument's tracked targets (history_t0: the reference clip's t0 frame, shared by every clip)
    bx = []
    for fam, tid in PIPE_ROLES:
        h0 = {round(x["t"], 1): x for x in ref["families"][fam]["history"]["series"]}[0.0]
        bx.append(("history_t0.png", tid, fam, *h0["box"]))
    followed = {}                                   # time -> (targets followed in the real reference, in the median imagined future)
    for t in seq:
        for name, c in ((f"real_ref_{ttag(t)}.png", ref), (f"sample_med_{ttag(t)}.png", clips[kmed])):
            if name not in missing:
                bx += [(name, tid, fam, *ens(c, fam)[t]["box"]) for fam, tid in PIPE_ROLES if seen(c, fam, t)]
        followed[t] = tuple([fam for fam, _ in PIPE_ROLES if seen(c, fam, t)] for c in (ref, clips[kmed]))
    with open(P / "boxes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "track_id", "role", "x0", "y0", "x1", "y1"])
        for r in bx: w.writerow([r[0], r[1], r[2], *[repr(float(v)) for v in r[3:]]])

    # ground rows: the instrument's flat road (window camera; road 0.33 m below the AV2 ego origin), straight ahead (y = 0)
    K = np.asarray(win["intrinsics"]["K_canvas"], float); E = np.asarray(win["intrinsics"]["ego_SE3_cam"], float); Rm, tv = E[:3, :3], E[:3, 3]
    g = float(cfg.get("ground_below_ego_m", 0.33)); dv = float(cfg.get("dv", 0.0)); gr = []
    for d in (5, 10, 15, 20, 30, 40, 60):
        Q = (np.array([d, 0.0, -g]) - tv) @ Rm
        u = K[0, 0] * Q[0] / Q[2] + K[0, 2]; v = K[1, 1] * Q[1] / Q[2] + K[1, 2]
        r = Rm @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]); back = tv + (-(tv[2] + g) / r[2]) * r
        assert abs(back[0] - d) < 1e-6 and abs(back[1]) < 1e-6
        gr.append((d, v + 0.5 - dv, u + 0.5))
    with open(P / "ground_rows.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["distance_m", "row_px", "col_px"])
        for d, rr, cc in gr: w.writerow([d, repr(float(rr)), repr(float(cc))])

    # tracks: positions relative to the ego (instrument) composed with the odometry pose, in the ego frame at t0
    vo = f"{B}/orbis2/ego-vo"
    fut = [round(float(t), 1) for t in np.arange(0, 4.0001, 0.1)]; his = [round(float(t), 1) for t in np.arange(-2.0, -0.0499, 0.1)]
    trk, d4 = [], {}
    for k, c in clips.items():
        if k == 0:
            traj = js(f"{vo}/clips-hist/orbis2-test-recon/{W}__real.json")["history"]["trajectory"] + js(f"{vo}/clips/orbis2-test-recon/{W}__real.json")["planar"]["trajectory"]
        else:
            traj = js(f"{vo}/clips/orbis2-test-k16/{W}__{k - 1}.json")["planar"]["trajectory"]
        traj = sorted({round(p["t"], 1): p for p in traj}.values(), key=lambda p: p["t"])
        tt = np.array([p["t"] for p in traj]); xx = np.array([p["x"] for p in traj]); yy = np.array([p["y"] for p in traj]); pp = np.radians([p["psi_deg"] for p in traj])
        pose = lambda s: (float(np.interp(s, tt, xx)), float(np.interp(s, tt, yy)), float(np.interp(s, tt, pp)))
        d4[k] = pose(4.0)[0]
        times = (his + fut) if k == 0 else fut
        trk += [(out(k), k, "ego", s, *pose(s)[:2], 1) for s in times]
        for fam, _ in PIPE_ROLES:
            e = ens(c, fam); h = {round(x["t"], 1): x for x in c["families"][fam]["history"]["series"]} if k == 0 else {}
            for s in times:
                x = h.get(s) if s < 0 else e.get(s)
                if x is not None and x.get("x") is not None and ((x.get("n") or 0) > 0 if s < 0 else (x.get("followed") or 0) > 0):
                    ex, ey, ps = pose(s)
                    trk.append((out(k), k, fam, s, ex + np.cos(ps) * x["x"] - np.sin(ps) * x["y"], ey + np.sin(ps) * x["x"] + np.cos(ps) * x["y"], 1))
                else:
                    trk.append((out(k), k, fam, s, None, None, 0))
    for k in range(17):
        assert abs(d4[k] - val[("ego_inter", "ego_dx_0_4s")][k]) < 2e-3, (k, d4[k])
    with open(P / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video", "sample_index", "role", "t_s", "x_m", "y_m", "valid"])
        for r in trk: w.writerow([r[0], r[1], r[2], repr(r[3]), "" if r[4] is None else repr(float(r[4])), "" if r[5] is None else repr(float(r[5])), r[6]])

    # DEV pairs: Orbis 2's reference clips against the LiDAR annotations (targets) and the AV2 ego poses (ego)
    dv_rows = []
    for r in js(f"{B}/orbis2/verification-dev/validation/records-orbis2-dev-recon.json"):
        q = {"lead": "x_4s", "lead_stop": "x_4s", "cutin": "absy_2s", "oncoming": "x_1s"}.get(r["family"])
        o = (r.get("outcomes") or {}).get(q) if q else None
        if o and r.get("scorable") and o.get("value") is not None and o.get("err_at_read") is not None:   # truth at the time the value was read
            dv_rows.append(("orbis2", r["name"], r["family"], q, o["value"], o["value"] - o["err_at_read"], "lidar_annotation", int(bool(o.get("right_target"))), None, None, o["read_t"]))
    for name in open(f"{B}/instrument/ego-vo/names-ego-inter-dev-gen.txt").read().split():
        d = js(f"{vo}/clips/orbis2-dev-recon/{name}__real.json")["planar"]["horizons"]["4.0"]["D_m"]
        gt = {e["k"]: e for e in js(f"{B}/windows-dev/{name}/gt.json")["ego"]}[40]["xyz_m"][0]; tol = max(0.15 * abs(gt), 1.5)
        dv_rows.append(("orbis2", name, "ego_inter", "ego_dx_0_4s", d, gt, "av2_ego_pose", None, tol, int(abs(d - gt) <= tol), 4.0))
    with open(P / "dev_lidar.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "window_id", "family", "quantity", "read_t_s", "instrument_m", "truth_m", "truth_source", "right_target", "tolerance_m", "within_tolerance"])
        for r in dv_rows: w.writerow([*r[:4], repr(float(r[10])), repr(float(r[4])), repr(float(r[5])), r[6], "" if r[7] is None else r[7], "" if r[8] is None else repr(float(r[8])), "" if r[9] is None else r[9]])
    ego_err = [abs(r[4] - r[5]) for r in dv_rows if r[2] == "ego_inter"]
    print(f"data/pipeline: window {W}, median sample #{kmed} ({lead[kmed]:.3f} m; median {med:.4f}), early time {T} s ({nT}/6 targets followed), "
          f"{len(pngs)} frames, {len(bx)} boxes, {len(trk)} track rows, {len(dv_rows)} DEV pairs (ego median |err| {np.median(ego_err):.4f} m, n {len(ego_err)}); "
          f"missing frames: {missing or 'none'}; followed per time (real ref | sample #{kmed}): " + "; ".join(f"{t:g}: {','.join(a) or '-'} | {','.join(b) or '-'}" for t, (a, b) in followed.items()))
    pipeline_docs(dict(W=W, kmed=kmed, med=med, lead=lead, T=T, tag=tag, pngs=pngs, nbox=len(bx), ntrk=len(trk), ndev=len(dv_rows), gr=gr,
                       seq=seq, missing=missing, followed=followed))


def pipeline_docs(i):
    P = OUT / "pipeline"; W, tag, T, k = i["W"], i["tag"], i["T"], i["kmed"]
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--", SCRIPT], capture_output=True, text=True).stdout.strip()
    prod = f"- **Produced by:** `BENCH_DIR=<bench-v1> python {SCRIPT} --pipeline` (export tool at repository commit {commit}), run on {TODAY}, from the frozen TEST and DEV files of Orbis 2 listed in README.md."
    std = lambda filt, paired, empty: [prod, f"- **Filters:** {filt}", f"- **Paired:** {paired}", f"- **Empty cells:** {empty}"]
    def doc(name, intro, cols, notes):
        (P / name).write_text("\n".join([f"# {name[:-3]}.csv", "", intro, "", "| column | meaning | values or unit |", "|---|---|---|", *cols, "", *notes]) + "\n")
    doc("readings.md", f"One row is one reading of a primary quantity in the example window {W}: the real reference and the 16 imagined futures of Orbis 2. 68 rows.", [
        "| quantity | the primary quantity | x_4s (lead gap at 4 s), absy_2s (cut-in, absolute lateral offset at 2 s), x_1s (oncoming, forward position at 1 s), ego_dx_0_4s (ego advance over 0-4 s) |",
        "| video | the real reference (the real future through Orbis 2's own autoencoder) or an imagined future | real, imagined |",
        "| sample_index | 0 for the real reference; 1-16 for the imagined futures (sample N is seed index N - 1) | 0-16 |",
        "| value_m | the frozen instrument reading | m |"],
        std("none.", "the 17 readings of one quantity form one case. The rank of the real reading equals data/case_ranks.csv: lead 6, cut-in 4, oncoming 3, ego 7.", "none."))
    seqtags = [ttag(t) for t in i["seq"]]
    fol = ["| time after t0 | followed in `real_ref_<t>.png` | followed in `sample_med_<t>.png` |", "|---|---|---|"] + [
        f"| {t:g} s | {', '.join(a) or 'none'} | {', '.join(b) or 'none'} |" for t, (a, b) in i["followed"].items()]
    doc("boxes.md", f"One row is one target the instrument tracks, in one image of this folder. {i['nbox']} rows.", [
        "| image | the image file | history_t0.png; `real_ref_<t>.png` and `sample_med_<t>.png` for t = " + ", ".join(seqtags) + " |",
        "| track_id | the target, the same in every image | 1 lead, 2 cut-in, 3 oncoming |",
        "| role | the target's family | lead, cutin, oncoming (the instrument tracks only the family targets, so `other` does not occur) |",
        "| x0 | left edge | pixels of the 512 x 288 image, from the left (pixel edges) |",
        "| y0 | top edge | pixels from the top |",
        "| x1 | right edge | pixels |",
        "| y1 | bottom edge | pixels |"],
        std("a target appears in `real_ref_<t>.png` or `sample_med_<t>.png` only if at least one of the three trackers follows it at that time.",
            "not applicable.", "none.") + [
        "- **history_t0.png:** its boxes are those of the reference clip's t0 frame. Frames 0-20 of every Orbis 2 clip are the autoencoder round trip of the history, so the raw frame and these boxes differ only by that round trip, as in data/teaser/.",
        "- **`real_raw_<t>.png`** has no rows: the instrument reads the reference clip, so its boxes are those of `real_ref_<t>.png`.",
        "- **Targets followed per time** (the real reference and imagined future #" + str(k) + "):", "", *fol])
    doc("ground_rows.md", "One row is the image row of the flat road at one distance straight ahead, from the instrument's geometry for this window. 7 rows.", [
        "| distance_m | forward distance from the AV2 ego origin (the rear axle) on the road plane | m: 5, 10, 15, 20, 30, 40, 60 |",
        "| row_px | the image row of that road point, in the pixel-edge convention of the boxes: a box whose bottom edge lies on this row reads this distance | pixels from the top of the 512 x 288 image |",
        "| col_px | the image column of the straight-ahead road point | pixels from the left |"],
        std("none. Rows at 288 or more lie below the image (5 m).", "not applicable.", "none.") + [
        "- **Geometry:** the window's camera (window.json: K_canvas, ego_SE3_cam) and the road 0.33 m below the AV2 ego origin, as in the instrument. Every clip of this window uses the same canvas and geometry, so there is no image column.",
        "- **Check:** back-projecting each (col_px, row_px) gives the distance to within 1e-6 m."])
    doc("tracks.md", f"One row is one road user of one video at one time, in the bird's-eye frame. {i['ntrk']} rows.", [
        "| video | the real reference or an imagined future | real, imagined |",
        "| sample_index | 0 for the real reference; 1-16 imagined | 0-16 |",
        "| role | the road user | ego, lead, cutin, oncoming |",
        "| t_s | time after t0 | s, every 0.1 s: 0-4 s; the real video also -2 to -0.1 s (the history) |",
        "| x_m | forward position in the ego frame at t0; empty where valid = 0 (target not followed) | m |",
        "| y_m | lateral position in the ego frame at t0, positive to the left; empty where valid = 0 | m |",
        "| valid | 1 if the position was observed | 1 or 0 |"],
        std("none.", "not applicable.", "x_m and y_m are empty where valid = 0.") + [
        "- **Frame:** the AV2 ego frame at t0 (origin at the rear axle, x forward, y left), in metres.",
        "- **Ego:** the visual-odometry path of the video. The odometry has a node every 0.4 s; the rows in between are linear interpolations of x, y and heading. The real history (t < 0) comes from the odometry run on the history frames.",
        "- **Targets:** the instrument's reading at time t (x forward, y left, in the ego frame at t) composed with the ego pose at t. valid = 0 where no tracker follows the target (lost or out of view); the instrument then carries its last reading, which is not a position.",
        "- **Check:** the ego x at 4 s equals ego_dx_0_4s in readings.csv to within 2 mm."])
    doc("dev_lidar.md", f"One row is one DEV history of one family: the instrument's reading on Orbis 2's DEV reference clip (the real video through its autoencoder) against the truth. {i['ndev']} rows.", [
        "| method | the model whose reference clips were read | orbis2 |",
        "| window_id | the DEV history | AV2 log prefix and anchor |",
        "| family | the scenario family | lead, lead_stop, cutin, oncoming, ego_inter |",
        "| quantity | the primary quantity | x_4s, absy_2s, x_1s, ego_dx_0_4s |",
        "| read_t_s | when the reading was taken: the horizon, or earlier if the target left the view and the reading was carried | s after t0 |",
        "| instrument_m | the instrument's reading | m |",
        "| truth_m | the truth at read_t_s: the LiDAR annotation of the target, or the AV2 ego pose at 4 s for the ego | m |",
        "| truth_source | where the truth comes from | lidar_annotation, av2_ego_pose |",
        "| right_target | 1 if the instrument read the annotated target; empty for the ego, which has no target box | 1 or 0 |",
        "| tolerance_m | the odometry gate's tolerance, max(0.15 x abs(truth), 1.5 m); empty for the targets, which have no tolerance | m |",
        "| within_tolerance | 1 if abs(instrument - truth) is within the tolerance; empty for the targets | 1 or 0 |"],
        std("targets: the histories in which the target was found at t0 and the quantity was read. Ego: the 32 DEV histories with the ego role and an Orbis 2 reference clip.",
            "not applicable.", "right_target is empty for the ego; tolerance_m and within_tolerance are empty for the targets.") + [
        "- **Name:** the column is `truth_m`, not `lidar_m`, because the ego truth is the AV2 ego pose, not a LiDAR box.",
        "- **Accept rules:** a target quantity is *tested* for a model if the right target is read in at least 90 % of DEV histories on the raw and on the reconstructed video, and the two readings agree (Spearman >= 0.8; the raw readings are not in this file). The ego family is scored if the odometry passes its gate on the raw and on the reconstructed video: median of abs(error) / tolerance <= 1, median heading error <= 3 deg, Spearman >= 0.90.",
        "- **Check:** over the right-target rows, the median absolute errors are 1.092 m (lead), 1.302 m (lead_stop), 0.287 m (cut-in) and 1.730 m (oncoming), as in the frozen validation summary; the ego median absolute error is 0.226 m, with all 32 within tolerance, as in the frozen gate result."])
    have = lambda pre: ", ".join(f"{pre}_{s}.png" for s in seqtags if f"{pre}_{s}.png" not in i["missing"])
    span = f"every 0.5 s from t0 + {i['seq'][0]:g} s to t0 + {i['seq'][-1]:g} s"
    files = [("history_m2s.png, history_m1s.png, history_t0.png", "the real history at -2 s, -1 s and t0 (the raw video)"),
             (have("real_raw"), f"the real future {span}, before the autoencoder (the raw video)"),
             (have("real_ref"), f"the same frames after Orbis 2's autoencoder round trip: the frames the instrument reads"),
             (have("sample_med"), f"imagined future #{k} at the same times: the one with the median lead gap at 4 s (sample_med_4s.png is the same frame as sample{k:02d}_4s.png)"),
             (", ".join(f"sample{j:02d}_4s.png" for j in range(1, 17)), "the 16 imagined futures of Orbis 2 at t0 + 4 s (sample N is seed index N - 1)"),
             ("type_general_4s.png", "imagined future #1 of LTX-2.5 (general world model) at t0 + 4 s, same history"),
             ("type_physical_4s.png", "imagined future #1 of Cosmos3-Nano (physical world model) at t0 + 4 s, same history: its 320 x 192 frame cropped to rows 6-186 and resized to 512 x 288 (bicubic), as the instrument does"),
             ("readings.csv", "the 17 readings of each primary quantity"), ("boxes.csv", "the instrument's target boxes in history_t0.png and in the real_ref and sample_med frames"),
             ("ground_rows.csv", "the image rows of the road at 5-60 m"), ("tracks.csv", "the bird's-eye positions the quantities are read from"),
             ("dev_lidar.csv", "the DEV pairs behind 'validated on DEV against LiDAR'")]
    lk = i["lead"]
    (P / "README.md").write_text("\n".join([
        "# data/pipeline: the example history of the overview figure", "",
        f"One TEST history, {W}, continued by Orbis 2. Times are relative to t0; the canvas frame is 20 + 10 t. Every image is 512 x 288, lossless PNG, with nothing drawn in.", "",
        "## Why this history", "",
        "- Orbis 2 scores it for all four primary quantities, and every rank is typical: lead 6, cut-in 4, oncoming 3, ego 7 (data/case_ranks.csv). The figure explains the method, so a typical case suits it; the teaser (9947fb6e-t6.5) shows a failure.",
        f"- **Boxed early time {T:g} s:** at 1 s not every target is followed in both clips (real reference: {', '.join(i['followed'][1.0][0]) or 'none'}; imagined future #{k}: {', '.join(i['followed'][1.0][1]) or 'none'}). At {T:g} s all three targets are followed in both, so the single boxed frames are the `_{tag}` files.",
        f"- **Future stacks:** the raw real future, the real reference and imagined future #{k} {span}, so that every future can be drawn as a stack of frames like the history. "
        + ("Every time has a frame in all three clips." if not i["missing"] else "Missing (no frame in that clip): " + ", ".join(i["missing"]) + ".")
        + " Which targets are followed at each time: boxes.md.",
        f"- **Median imagined future: #{k}.** Its lead gap at 4 s is {lk[k]:.3f} m. The 16 values have the median {i['med']:.3f} m, between #{sorted(range(1, 17), key=lambda j: lk[j])[7]} ({sorted(lk[j] for j in range(1, 17))[7]:.3f} m) and #{sorted(range(1, 17), key=lambda j: lk[j])[8]} ({sorted(lk[j] for j in range(1, 17))[8]:.3f} m); #{k} is the closer.", "",
        "## Files", "", "| file | what it is |", "|---|---|", *[f"| {a} | {b} |" for a, b in files], "",
        "Each CSV has a description beside it (`<name>.md`).", "",
        "## Licence and privacy", "",
        "The frames come from Argoverse 2 (CC BY-NC-SA 4.0) and from model outputs conditioned on it. No blurring was added, as for data/teaser/; check faces and plates before publication.", "",
        "## Provenance", "",
        prod[2:].replace("**Produced by:** ", "Produced by "), "",
        "Sources, relative to the bench-v1 root:",
        "- `orbis2/instrument/measure-test/measure-orbis2-test-{k16,recon}.json` (tracks, boxes), `orbis2/instrument/outcomes/test/outcomes-orbis2-test-k16.csv` (readings);",
        "- `orbis2/ego-vo/clips{,-hist}/orbis2-test-{k16,recon}/` (odometry), `windows-test/<id>/window.json` (camera), `windows-test/<id>/canvas/real.mp4` and the `vision.mp4` clips of Orbis 2, LTX-2.5 and Cosmos3-Nano (frames);",
        "- `orbis2/verification-dev/validation/records-orbis2-dev-recon.json`, `orbis2/ego-vo/clips/orbis2-dev-recon/`, `windows-dev/<id>/gt.json` and `instrument/ego-vo/names-ego-inter-dev-gen.txt` (DEV pairs)."]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", action="store_true"); ap.add_argument("--pipeline", action="store_true"); a = ap.parse_args()
    if a.pipeline:
        pipeline(); sys.exit(0)
    case_ranks(); confirmatory(); family_stats(); quantity_stats(); trajectory(); ego_motion(); property_scores(); continuation_break(); teaser(a.frames)
