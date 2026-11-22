#!/usr/bin/env python3
"""Shared canvas instrument (canvas-v1): measure and export, the frozen measure_bench_v2.py steps on a model's canvas
geometry and frame rate (ci_core.py; spec models/<model>.json).

measure: per clip, the frozen trackers v3 / v5 / v8, warm-up, t0 check, predicted exit, history run and median
         ensemble (ci_core.measure_clip) with the effective config (ci_core.effective_config of instrument/config-v2.json).
export:  one CSV row per (window, family, sample, outcome), the frozen export (scorability rules, history rows, ego rows
         from the band odometry, ego_vo_band.py) plus, for 'per_clip' prefix models (LTX-2.5), the t0 association check
         of protocol-v1.1-instrument.md ("should a clip still differ, drop the (window, family)"):
           a generated clip differs when fewer than 2 of the 3 trackers matched a detection at t0 in both the clip and
           its real reference with boxes of IoU >= 0.5 (it latched onto another object, or does not find the target
           where the reference does);
           --t0-rule window (default, v1.1 as written): every row of that (window, family) gets scorable = 0, the
             differing clips the flag t0_mismatch, the other samples t0_mismatch_in_window;
           --t0-rule clip (sensitivity): only the differing clips (flag t0_mismatch, scorable = 0).
Detections are looked up as <det-root>/<source>/<window>.json (raw clip, sample 'raw') or <window>-<sample>.json (run
clips: 'recon' -> sample 'real', 's<k>' -> k), in the order of --det-root.

  measure_canvas.py measure --bench B --names-file F --model M --sources SRC... --det-root DIR... --out DIR [--allow-test]
  measure_canvas.py export  --bench B --model M --measure DIR --run RUN --real-run REF --out OUT.csv [--rules R.json]
                            [--raw-source SRC] [--history-out H.csv] [--t0-rule window|clip] [--ego-windows-dir windows-test]
DEV by default; TEST windows only with --allow-test once the model's frozen protocol file exists (ci_core.split_of).
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import ci_core as ci  # noqa: E402
import measure_bench_v2 as mb  # noqa: E402  (frozen, read only)


def det_dir(roots, source):
    for r in roots:
        if (Path(r) / source).is_dir():
            return Path(r) / source
    raise SystemExit(f'no detections for source {source} under {roots}')


def clip_list(roots, names, source):
    d = det_dir(roots, source)
    for name in names:
        f = d / f'{name}.json'
        if f.is_file():
            yield name, 'raw', f
            continue
        for f in sorted(d.glob(f'{name}-*.json')):
            sample = f.stem[len(name) + 1:]
            yield name, ('real' if sample == 'recon' else sample.lstrip('s')), f


def cmd_measure(args):
    spec = ci.load_spec(args.model)
    frozen = json.loads(args.config.read_text())
    cfg = ci.effective_config(frozen, spec)
    names = ci.names_from(args.names_file); split = ci.split_of(args.bench, names, spec, args.allow_test)
    fams = set(args.families.split(',')) if args.families else None
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'effective-config-{spec["model"]}.json').write_text(json.dumps(cfg, indent=1) + '\n')
    for source in args.sources:
        res = dict(config=cfg, config_file=str(args.config), config_sha256=ci.sha256_file(args.config),
                   effective_config_sha256=ci.sha256_file(args.out / f'effective-config-{spec["model"]}.json'),
                   spec=spec, script_sha256=dict(measure_canvas=ci.sha256_file(__file__), ci_core=ci.sha256_file(ci.__file__),
                                                 measure_bench_v2=ci.sha256_file(mb.__file__)),
                   source=source, windows={})
        n = 0
        for name, sample, f in clip_list(args.det_root, names, source):
            window = json.loads((args.bench / ci.windows_dir(split) / name / 'window.json').read_text())
            dets = json.loads(f.read_text())
            res['windows'].setdefault(name, {})[sample] = dict(detections=str(f), families=ci.measure_clip(window, dets, cfg, spec, fams))
            n += 1
        (args.out / f'measure-{source}.json').write_text(json.dumps(res) + '\n')
        print(json.dumps(dict(source=source, model=spec['model'], clips=n, windows=len(res['windows']))), flush=True)


def cmd_export(args):
    """measure_bench_v2.cmd_export with the t0_mismatch rule for per_clip-prefix models and the band odometry."""
    spec = ci.load_spec(args.model)
    gen = json.loads((args.measure / f'measure-{args.run}.json').read_text())
    real = {'windows': {}} if args.real_run == 'none' else json.loads((args.measure / f'measure-{args.real_run}.json').read_text())
    raw = json.loads((args.measure / f'measure-{args.raw_source}.json').read_text()) if args.raw_source else None
    rules = json.loads(args.rules.read_text()) if args.rules else {}
    ego = rules.get('ego')
    per_clip = spec['prefix'] == 'per_clip'
    rows, hrows = [], []
    t0_stats = dict(rule=args.t0_rule if per_clip else 'shared prefix (not applied)', window_families=0, dropped_window_families=0, differing_clips=0)
    names = sorted(set(gen['windows']) | set(real['windows']))
    for name in names:
        samples = dict(gen['windows'].get(name, {}))
        if 'real' in real['windows'].get(name, {}):
            samples['real'] = real['windows'][name]['real']
        if raw is not None and 'raw' in raw['windows'].get(name, {}):
            samples['raw'] = raw['windows'][name]['raw']
        real_fams = real['windows'].get(name, {}).get('real', {}).get('families', {})
        order = sorted(samples, key=lambda x: (x in ('real', 'raw'), x == 'raw', x.zfill(4)))
        differ = {}                                   # family -> generated samples whose t0 association differs
        if per_clip and args.real_run != 'none':
            for sample in order:
                if sample in ('real', 'raw'):
                    continue
                for fam, rec in samples[sample]['families'].items():
                    ref = real_fams.get(fam)
                    if ref is not None and sum(ref['t0_found'].values()) >= 2 and ci.t0_agreement(rec, ref) < 2:
                        differ.setdefault(fam, set()).add(sample)
        t0_stats['window_families'] += len({f for s_ in order if s_ not in ('real', 'raw') for f in samples[s_]['families']})
        t0_stats['dropped_window_families'] += len(differ); t0_stats['differing_clips'] += sum(len(v) for v in differ.values())

        def t0_flag(fam, sample):
            if fam not in differ:
                return None
            if sample in differ[fam]:
                return 't0_mismatch'
            return 't0_mismatch_in_window' if args.t0_rule == 'window' else None
        for sample in order:
            if args.real_run == 'none':
                real_fams = samples[sample]['families']
            for fam, rec in samples[sample]['families'].items():
                tf = t0_flag(fam, sample)
                for oname, kind in mb.OUTCOMES[fam]:
                    o = rec['ensemble']['outcomes'][oname]
                    flags, scorable = mb.scorability(fam, rec, real_fams.get(fam), oname, rules)
                    if tf:
                        flags.append(tf); scorable = False
                    each = dict(zip(sorted(rec['trackers']), o['each']))
                    rows.append(dict(window=name, family=fam, sample=sample, outcome=oname, kind=kind,
                                     value='' if o['value'] is None else round(o['value'], 4), scorable=int(scorable), flags=';'.join(flags),
                                     **{f'value_{k}': ('' if each.get(k) is None else round(each[k], 4)) for k in ('v3', 'v5', 'v8')}))
                if sample in ('real', 'raw'):
                    hr = mb.history_rows(name, fam, sample, rec, real_fams.get(fam), rules)
                    if tf:
                        for r in hr:
                            r['flags'] = ';'.join([x for x in r['flags'].split(';') if x] + [tf]); r['scorable'] = 0
                    rows += hr
                    hrows += mb.history_frame_rows(name, fam, sample, rec)
        if ego:
            wdir = args.bench / (args.ego_windows_dir or ego.get('windows_dir', 'windows-dev')) / name
            if 'ego_inter' in json.loads((wdir / 'window.json').read_text())['roles']:
                for sample in order:
                    src, key = {'real': (args.real_run, 'real'), 'raw': (args.raw_source or args.run, 'raw')}.get(sample, (args.run, sample))
                    rows += mb.ego_rows(args.bench / ego['clips_dir'] / src / f'{name}__{key}.json', name, sample, ego,
                                        args.bench / ego['hist_dir'] / src / f'{name}__{key}.json')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=mb.FIELDS)
        w.writeheader(); w.writerows(rows)
    if args.history_out:
        with args.history_out.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['window', 'family', 'sample', 't', 'x', 'y', 'n_trackers'])
            w.writeheader(); w.writerows(hrows)
    print(json.dumps(dict(rows=len(rows), scorable=sum(r['scorable'] for r in rows), t0_check=t0_stats, out=str(args.out), history_rows=len(hrows))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('measure')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--det-root', nargs='+', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=None, help='frozen tracker config (default BENCH/instrument/config-v2.json)')
    p.add_argument('--families', default='')
    p.add_argument('--allow-test', action='store_true', help="TEST windows; refused unless the spec's test_freeze_file exists in the bench")
    p = sub.add_parser('export')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--model', required=True); p.add_argument('--measure', type=Path, required=True)
    p.add_argument('--run', required=True); p.add_argument('--real-run', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rules', type=Path, default=None); p.add_argument('--raw-source', default=None); p.add_argument('--history-out', type=Path, default=None)
    p.add_argument('--t0-rule', choices=('window', 'clip'), default='window', help="per_clip-prefix models: drop the (window, family) (v1.1) or only the clip")
    p.add_argument('--ego-windows-dir', default=None, help="where window roles are read for the ego rows (TEST: windows-test; default: the rules file's)")
    args = ap.parse_args()
    if args.cmd == 'measure' and args.config is None:
        args.config = args.bench / 'instrument' / 'config-v2.json'
    {'measure': cmd_measure, 'export': cmd_export}[args.cmd](args)


if __name__ == '__main__':
    main()
