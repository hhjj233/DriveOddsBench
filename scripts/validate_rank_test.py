#!/usr/bin/env python3
"""Checks of DriveOddsBench's rank test on simulated numbers with known answers (App. F).

No world model, video or driving data is used.
Writes data/statistical_validation.json, data/statistical_validation.csv and tables/statistical_validation.tex
under the repository root.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0-p) / trials + z*z/(4.0*trials*trials)) / denominator
    return max(0.0, center-half), min(1.0, center+half)


def integer_tv_statistic(ranks: np.ndarray, k: int) -> np.ndarray:
    """Return 2*n*(k+1)*TV as an integer to compare tied statistics exactly."""
    assert ranks.ndim == 2
    batch, n = ranks.shape
    assert np.all((ranks >= 0) & (ranks <= k))
    offsets = (k+1) * np.arange(batch, dtype=np.int64)[:, None]
    counts = np.bincount((ranks+offsets).ravel(), minlength=batch*(k+1)).reshape(batch, k+1)
    assert np.all(counts.sum(axis=1) == n)
    return np.abs((k+1)*counts-n).sum(axis=1, dtype=np.int64)


def mc_pvalues(statistics: np.ndarray, sorted_null: np.ndarray) -> np.ndarray:
    ge = len(sorted_null) - np.searchsorted(sorted_null, statistics, side='left')
    return (ge + 1.0) / (len(sorted_null) + 1.0)


def summarize(case, n, stats, null_bank, k, alpha):
    ps = mc_pvalues(stats, null_bank)
    rejected = int(np.sum(ps <= alpha))
    reps = len(stats)
    rate = rejected/reps
    lo, hi = wilson_interval(rejected,reps)
    return dict(case=case, evaluation_cases=n, replicates=reps, rejections=rejected,
                rejection_rate=rate, conditional_mcse=math.sqrt(rate*(1-rate)/reps),
                wilson95_lower=lo, wilson95_upper=hi,
                mean_empirical_tv=float(np.mean(stats)/(2*n*(k+1))),
                minimum_p=float(np.min(ps)), maximum_p=float(np.max(ps)))


def gaussian_case(rng,n,k,reps,batch_size,mu,sigma):
    out=[]
    for start in range(0,reps,batch_size):
        b=min(batch_size,reps-start)
        observed=rng.normal(0.0,1.0,size=(b,n))
        generated=rng.normal(mu,sigma,size=(b,n,k))
        ranks=np.sum(generated < observed[:,:,None],axis=2,dtype=np.int64)
        out.append(integer_tv_statistic(ranks,k))
    return np.concatenate(out)


def bernoulli_cases(rng,n,k,reps,batch_size):
    proper=[]
    naive=[]
    for start in range(0,reps,batch_size):
        b=min(batch_size,reps-start)
        observed=rng.binomial(1,0.3,size=(b,n))
        generated=rng.binomial(1,0.3,size=(b,n,k))
        smaller=np.sum(generated < observed[:,:,None],axis=2,dtype=np.int64)
        equal=np.sum(generated == observed[:,:,None],axis=2,dtype=np.int64)
        # Conditional on E ties, the observed item has a uniformly random
        # position J in {0,...,E}; this equals independent lexicographic tags
        # in distribution. No comparison with a discrete uniform U is used.
        tie_position=rng.integers(low=0,high=equal+1,size=equal.shape)
        proper.append(integer_tv_statistic(smaller+tie_position,k))
        naive.append(integer_tv_statistic(smaller,k))
    return np.concatenate(proper),np.concatenate(naive)


def write_latex(rows,output):
    cases=[('normal_calibrated','Gaussian: correct'),
           ('normal_mean_shift','Gaussian: mean shift'),
           ('normal_narrow','Gaussian: narrow spread'),
           ('normal_wide','Gaussian: wide spread'),
           ('bernoulli_randomized','Bernoulli: randomized ties'),
           ('bernoulli_naive_strict','Bernoulli: naive strict ranks')]
    lookup={(r['case'],r['evaluation_cases']):r for r in rows}
    lines=[r'\begin{table*}[t]',r'\centering',r'\small',
      r'\caption{\textbf{Scalar statistical validation, not driving-video results.} Rejection frequency (\%) with 95\% Wilson intervals over 2,000 repetitions. Each repetition uses $K=16$ independent generated scalar samples per observed scalar and the total-variation rank statistic at $\alpha=0.05$. Null references use 50,000 multinomial draws for each $N$, a plus-one Monte Carlo $p$-value, and inclusive comparison of tied statistics. Intervals quantify repetition uncertainty conditional on the reused null bank. Gaussian observations follow $\mathcal N(0,1)$; the forecast distributions are $\mathcal N(0,1)$, $\mathcal N(0.5,1)$, $\mathcal N(0,0.5^2)$, and $\mathcal N(0,2^2)$. Both Bernoulli experiments use the same calibrated $\mathrm{Bernoulli}(0.3)$ samples and differ only in tie handling.}',
      r'\label{tab:statistical_validation}',r'\begin{tabular}{lccc}',r'\toprule',
      r'Scalar setting & $N=64$ & $N=256$ & $N=1024$ \\',r'\midrule']
    for key,label in cases:
        cells=[]
        for n in [64,256,1024]:
            r=lookup[key,n]
            cells.append(f"{100*r['rejection_rate']:.1f} [{100*r['wilson95_lower']:.1f}, {100*r['wilson95_upper']:.1f}]")
        lines.append(label+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])
    output.write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261110)
    parser.add_argument('--replicates',type=int,default=2000)
    parser.add_argument('--null-draws',type=int,default=50000)
    parser.add_argument('--batch-size',type=int,default=50)
    args=parser.parse_args()
    if args.replicates!=2000 or args.null_draws!=50000:
        raise ValueError('This paper table is fixed to 2000 repetitions and 50000 null draws; edit caption/settings together for other budgets.')
    k,alpha=16,0.05
    ns=[64,256,1024]
    gaussian_specs=[('normal_calibrated',0.0,1.0),('normal_mean_shift',0.5,1.0),
                    ('normal_narrow',0.0,0.5),('normal_wide',0.0,2.0)]
    rows=[]
    null_meta=[]
    for n,ss in zip(ns,np.random.SeedSequence(args.seed).spawn(len(ns))):
        seeds=ss.spawn(6)
        null_rng=np.random.default_rng(seeds[0])
        null_counts=null_rng.multinomial(n,np.full(k+1,1/(k+1)),size=args.null_draws)
        null_bank=np.sort(np.abs((k+1)*null_counts-n).sum(axis=1,dtype=np.int64))
        null_meta.append(dict(evaluation_cases=n,spawn_key=list(seeds[0].spawn_key),
                             unique_statistic_values=int(len(np.unique(null_bank))),
                             mean_null_tv=float(null_bank.mean()/(2*n*(k+1)))))
        for (case,mu,sigma),seed in zip(gaussian_specs,seeds[1:5]):
            stats=gaussian_case(np.random.default_rng(seed),n,k,args.replicates,args.batch_size,mu,sigma)
            rows.append(summarize(case,n,stats,null_bank,k,alpha))
        proper,naive=bernoulli_cases(np.random.default_rng(seeds[5]),n,k,args.replicates,args.batch_size)
        rows.extend([summarize('bernoulli_randomized',n,proper,null_bank,k,alpha),
                     summarize('bernoulli_naive_strict',n,naive,null_bank,k,alpha)])
        print('Completed n='+str(n),flush=True)
    settings=dict(kind='controlled_scalar_simulation_not_driving_experiment',
      seed=args.seed,K=k,evaluation_cases=ns,replicates=args.replicates,null_draws=args.null_draws,
      batch_size=args.batch_size,alpha=alpha,statistic='TV(empirical rank frequencies, Uniform{0,...,16})',
      statistic_integer_form='sum_r abs(17*count_r - n); TV=integer_statistic/(34*n)',
      pvalue='(1 + number of null statistics >= observed statistic)/(B+1)',
      null_bank_reuse='One independent multinomial-uniform bank per n is reused across cases and repetitions.',
      uncertainty='Monte Carlo standard errors and Wilson intervals are conditional on that fixed null bank; they do not quantify uncertainty from regenerating the bank.',
      ties='L+J with J uniform over integers 0,...,E; naive diagnostic uses L only.',
      bernoulli_pairing='Randomized and naive procedures share exactly the same observed/generated Bernoulli arrays.',
      normal_observed=dict(mean=0.0,sd=1.0),
      normal_generated=[dict(case=c,mean=m,sd=s) for c,m,s in gaussian_specs],
      bernoulli_observed_and_generated_p=0.3,
      numpy_version=np.__version__,python_version=platform.python_version())
    for d in ['data','tables']:(ROOT/d).mkdir(exist_ok=True)
    out=ROOT/'data'/'statistical_validation.json'
    out.write_text(json.dumps(dict(settings=settings,null_banks=null_meta,results=rows),indent=2),encoding='utf-8')
    with (ROOT/'data'/'statistical_validation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write_latex(rows,ROOT/'tables'/'statistical_validation.tex')
    for r in rows:
        print(r['case'],r['evaluation_cases'],f"{r['rejection_rate']:.4f}",f"[{r['wilson95_lower']:.4f},{r['wilson95_upper']:.4f}]")
    print(out)


if __name__=='__main__': main()
