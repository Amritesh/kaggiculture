"""Search the agent's structural parameters for ABSOLUTE score.

`search.py` maximises win rate against the previous version. That is the right
objective when the opponent pool resembles the field, and the wrong one here:
every local opponent descends from this agent, so beating them rewards small
relative edges and never pushes total output. The agent settled at 8 hands and 3
quadrants, which is locally optimal against a weak mirror.

The competition is decided by coins, and the town absorbs roughly 2,319 units a
season while this agent supplies about 28% of them. So this searches the
parameters that set the production ceiling -- crew size, land, herd, seed rate --
and scores candidates on **mean coins earned**, not on beating anything.

Absolute score is far less noisy than win rate (a continuous quantity rather
than a coin flip), so fewer games resolve a real difference.

    python train/scale.py --iters 12 --pop 16 --seeds 24
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
sys.path.insert(0, str(ROOT / 'agent'))

import fastsim
import main as agent_mod

# Parameters that set how much the farm can produce, with generous bounds:
# the point is to find out whether a bigger operation is reachable, so the
# search must be allowed to try one.
SPACE = [
    ('workers_hi', 4.0, 16.0), ('workers_mid', 2.0, 14.0), ('workers_lo', 1.0, 10.0),
    ('active_hi', 1.0, 20.0), ('active_lo', 1.0, 12.0),
    ('max_land', 2.0, 4.0), ('land_cash', 100.0, 3000.0), ('land_last_day', 4.0, 24.0),
    ('land_fill', 0.05, 1.0),
    ('herd', 0.0, 16.0), ('animal_cash', 200.0, 3500.0), ('animal_last_day', 4.0, 18.0),
    ('feed_days', 0.0, 8.0), ('feed_batch', 2.0, 16.0),
    ('seed_cap', 2.0, 24.0), ('sell_floor', 0.25, 1.0), ('sell_spread', 1.0, 8.0),
    ('carry_cap', 5.0, 95.0),
]
NAMES = [s[0] for s in SPACE]
LO = [s[1] for s in SPACE]
HI = [s[2] for s in SPACE]

_OPP = {}


def _load(rel):
    if rel not in _OPP:
        import importlib.util
        spec = importlib.util.spec_from_file_location('o_%d' % abs(hash(rel)), ROOT / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _OPP[rel] = m.agent
    return _OPP[rel]


def evaluate(job):
    """Mean coins this candidate earns, across opponents and both seats."""
    theta, opponents, seeds = job
    agent_mod.TUNE.update({NAMES[i]: theta[i] for i in range(len(NAMES))})
    total = 0.0
    n = 0
    for rel in opponents:
        opp = _load(rel)
        for s in seeds:
            for seat in (0, 1):
                if seat:
                    r = fastsim.run_episode(opp, agent_mod.agent, seed=s)
                    total += r['scores'][1]
                else:
                    r = fastsim.run_episode(agent_mod.agent, opp, seed=s)
                    total += r['scores'][0]
                n += 1
    return total / max(1, n)


def clamp(v):
    return [min(HI[i], max(LO[i], v[i])) for i in range(len(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=12)
    ap.add_argument('--pop', type=int, default=16)
    ap.add_argument('--elite', type=int, default=4)
    ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--opponents', nargs='*',
                    default=['opponents/crops_v5.py', 'opponents/original.py'])
    ap.add_argument('--output', default='results/scale.json')
    args = ap.parse_args()

    rng = random.Random(11)
    base = [agent_mod.TUNE[n] for n in NAMES]
    mu = list(base)
    sigma = [(HI[i] - LO[i]) * 0.30 for i in range(len(NAMES))]
    best = (-1.0, list(base))
    history = []
    workers = args.workers or os.cpu_count()

    for it in range(args.iters):
        seeds = list(range(7000 + it * args.seeds, 7000 + (it + 1) * args.seeds))
        pop = [clamp(mu)] if it == 0 else []
        while len(pop) < args.pop:
            pop.append(clamp([rng.gauss(mu[i], sigma[i]) for i in range(len(NAMES))]))

        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            scored = list(ex.map(evaluate, [(p, args.opponents, seeds) for p in pop],
                                 chunksize=1))

        ranked = sorted(zip(scored, pop), key=lambda z: -z[0])
        elite = [p for _, p in ranked[:args.elite]]
        mu = [sum(e[i] for e in elite) / len(elite) for i in range(len(NAMES))]
        sigma = [max((HI[i] - LO[i]) * 0.04,
                     math.sqrt(sum((e[i] - mu[i]) ** 2 for e in elite) / len(elite)))
                 for i in range(len(NAMES))]

        if ranked[0][0] > best[0]:
            best = (ranked[0][0], list(ranked[0][1]))
        history.append({'iter': it, 'best': round(ranked[0][0]),
                        'median': round(statistics.median(scored)),
                        'seconds': round(time.perf_counter() - t)})
        print(f'iter {it:3d}  best {ranked[0][0]:9,.0f} coins   median '
              f'{statistics.median(scored):9,.0f}   ({time.perf_counter()-t:.0f}s)', flush=True)

        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'best_coins': best[0],
            'params': {NAMES[i]: round(best[1][i], 4) for i in range(len(NAMES))},
            'mean_params': {NAMES[i]: round(mu[i], 4) for i in range(len(NAMES))},
            'history': history}, indent=2))

    print(f'\nbest {best[0]:,.0f} coins')
    for i, n in enumerate(NAMES):
        if abs(best[1][i] - base[i]) > 0.02 * (HI[i] - LO[i]):
            print(f'  {n:16s} {base[i]:8.2f} -> {best[1][i]:8.2f}')
    print(f'\nwrote {args.output} -- validate with eval/league.py before shipping')


if __name__ == '__main__':
    main()
