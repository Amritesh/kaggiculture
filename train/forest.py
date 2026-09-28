"""Random-forest surrogate search over the agent's parameters.

CEM moves one cloud of points downhill and is blind to interactions: raising
`herd` alone does nothing because pens cannot be placed, and raising `barn_r`
alone does nothing because the herd target still caps it -- only the pair helps.
CEM looked at both, 330 candidates deep, and saw flat ground in each direction.
The pair was worth +11% when found by hand from a replay.

A forest does not have that blind spot. It is fitted to (config -> score) pairs
from randomly sampled configs, so a split on `barn_r` followed by a split on
`herd` represents exactly that interaction. The fitted model is then used as a
cheap surrogate: score tens of thousands of candidate configs in milliseconds,
run the real simulator only on the few the forest likes, add those results to
the training set, refit. Each round the surrogate gets better where it matters.

    python train/forest.py --rounds 6 --batch 24 --seeds 6
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
sys.path.insert(0, str(ROOT / 'agent'))

import fastsim
import main as agent_mod

SPACE = [
    ('herd', 6.0, 20.0), ('barn_r', 2.5, 6.0), ('animal_last_day', 8.0, 20.0),
    ('animal_cash', 500.0, 3500.0), ('animal_harvest', 2.0, 6.0),
    ('sheep_share', 0.0, 0.6), ('feed_buffer', 1.0, 4.0), ('feed_horizon', 6.0, 20.0),
    ('feed_batch', 4.0, 20.0), ('v_fert', 0.5, 8.0),
    ('workers_hi', 5.0, 12.0), ('active_hi', 2.0, 12.0),
    ('travel_cost', 0.3, 2.0), ('carry_cap', 8.0, 30.0),
    ('plant_radius', 4.0, 10.0), ('seed_cap', 2.0, 8.0),
    ('water_urgent', 100.0, 600.0), ('water_base', 10.0, 120.0),
    ('sell_floor', 0.35, 0.75), ('sell_spread', 1.0, 5.0),
    ('max_land', 2.0, 4.0), ('land_cash', 400.0, 2500.0), ('land_fill', 0.05, 0.8),
]
NAMES = [s[0] for s in SPACE]
LO = np.array([s[1] for s in SPACE])
HI = np.array([s[2] for s in SPACE])

POOL = ['opponents/real_sansm1.py', 'opponents/real_nono00.py',
        'opponents/real_niulai.py', 'opponents/real_ronel_abraham_math.py',
        'opponents/real_igeo_kk.py', 'opponents/real_rohin_kethipally.py']
_OPP = {}


def _load(rel):
    if rel not in _OPP:
        spec = importlib.util.spec_from_file_location('o%d' % abs(hash(rel)), ROOT / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _OPP[rel] = m.agent
    return _OPP[rel]


def evaluate(job):
    """Median ABSOLUTE score. The target is 'score 100k+ reliably', so the
    objective is the agent's own coins, and the median rather than the mean --
    a mean is inflated by a few lucky blowouts."""
    theta, seeds = job
    agent_mod.TUNE.update({NAMES[i]: float(theta[i]) for i in range(len(NAMES))})
    sc = []
    for rel in POOL:
        opp = _load(rel)
        for s in seeds:
            sc.append(fastsim.run_episode(agent_mod.agent, opp, seed=s)['scores'][0])
            sc.append(fastsim.run_episode(opp, agent_mod.agent, seed=s)['scores'][1])
    return statistics.median(sc), 100.0 * sum(1 for v in sc if v >= 100000) / len(sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--output', default='results/forest.json')
    args = ap.parse_args()

    rng = np.random.RandomState(7)
    base = np.array([agent_mod.TUNE[n] for n in NAMES], dtype=float)
    X, y, hit = [base.copy()], [], []
    # First batch is random, to give the forest something to learn from.
    while len(X) < args.batch:
        X.append(LO + rng.rand(len(NAMES)) * (HI - LO))

    workers = args.workers or os.cpu_count()
    best = (-1.0, base.copy(), 0.0)
    for r in range(args.rounds):
        seeds = list(range(50000 + r * args.seeds, 50000 + (r + 1) * args.seeds))
        todo = X[len(y):]
        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(evaluate, [(x, seeds) for x in todo], chunksize=1))
        for med, pct in out:
            y.append(med); hit.append(pct)
        for i, (med, pct) in enumerate(out):
            if med > best[0]:
                best = (med, todo[i].copy(), pct)
        print(f'round {r}: evaluated {len(todo):3d}  best median {best[0]:9,.0f}  '
              f'>=100k {best[2]:3.0f}%  ({time.perf_counter()-t:.0f}s)', flush=True)

        # Fit the forest and let it propose the next batch.
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=r, n_jobs=-1)
        rf.fit(np.array(X), np.array(y))
        cand = LO + rng.rand(20000, len(NAMES)) * (HI - LO)
        pred = rf.predict(cand)
        # Take the forest's favourites, plus a few random ones so the search
        # keeps seeing parts of the space the model is confident about wrongly.
        top = cand[np.argsort(-pred)[:args.batch - 4]]
        X.extend(list(top))
        X.extend(list(LO + rng.rand(4, len(NAMES)) * (HI - LO)))

        imp = sorted(zip(NAMES, rf.feature_importances_), key=lambda z: -z[1])[:6]
        print('   drivers: ' + ', '.join(f'{n} {100*v:.0f}%' for n, v in imp), flush=True)
        out_p = ROOT / args.output
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps({
            'best_median': best[0], 'best_hit100k': best[2],
            'params': {NAMES[i]: round(float(best[1][i]), 4) for i in range(len(NAMES))},
            'importances': {n: float(v) for n, v in zip(NAMES, rf.feature_importances_)}},
            indent=2))

    print(f'\nbest median {best[0]:,.0f}  ({best[2]:.0f}% of games >= 100k)')
    for i, n in enumerate(NAMES):
        if abs(best[1][i] - base[i]) > 0.08 * (HI[i] - LO[i]):
            print(f'  {n:16s} {base[i]:9.2f} -> {best[1][i]:9.2f}')
    print(f'\nwrote {args.output} -- validate paired before shipping')


if __name__ == '__main__':
    main()
