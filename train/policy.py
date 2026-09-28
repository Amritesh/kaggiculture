"""Learn the agent's decision constants from played games.

Everything the agent "knows" falls in three tiers. Tier 1 is engine arithmetic
(prices, growth times, feed rules) and tier 2 is which tasks are legal on a
tile; both stay, because farming is a chain -- plant, water, water, harvest --
and one missed link kills the crop, so a policy free to break chains loses to
one that cannot. Tier 3 is the hand-set number attached to each task, tier 4 the
crew/land/herd/cash constants and tier 5 the market policy. Those three tiers
are guesses, and this searches them.

The objective is the margin in real games against opponents that are NOT
descended from this agent. Three cheaper objectives were tried first and every
one of them recommended a change that lost:

  * mean coins             -- rated a candidate 88k that the league showed losing
  * win rate vs itself     -- 55-25 against its own parent, 99-61 vs non-clones
  * regression on profit   -- 32.5% variance explained, monotonically worse play

Cross-entropy method: sample a population, keep the best few, refit the mean and
spread. Seeds rotate every generation so nothing can be fitted to one block.

    python train/policy.py --iters 14 --pop 24 --seeds 10
"""
import argparse
import importlib.util
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

# (name, low, high). Tier 3 priorities first, then tier 4/5.
SPACE = [
    ('v_pen', 50.0, 4000.0), ('v_plant', 0.5, 8.0), ('v_aharv', 0.5, 8.0),
    ('v_feed_u', 5.0, 90.0), ('v_feed', 2.0, 60.0), ('v_care', 1.0, 30.0),
    ('v_place', 2000.0, 80000.0), ('v_water_h', 20.0, 900.0),
    ('v_harv', 20.0, 900.0), ('v_harv_m', 0.5, 6.0),
    ('v_pick_w', 300.0, 30000.0), ('v_pick_a', 1000.0, 60000.0),
    ('water_urgent', 40.0, 700.0), ('water_base', 2.0, 200.0),
    ('travel_cost', 0.2, 3.0), ('carry_cap', 5.0, 40.0),
    ('seed_cap', 1.5, 8.0), ('plant_radius', 3.0, 10.0),
    ('workers_hi', 5.0, 12.0), ('active_hi', 2.0, 12.0),
    ('herd', 6.0, 20.0), ('barn_r', 2.5, 6.0), ('animal_harvest', 2.0, 6.0),
    ('animal_last_day', 6.0, 18.0), ('feed_batch', 4.0, 16.0),
    ('feed_horizon', 4.0, 20.0), ('feed_safety', 0.5, 1.6),
    ('sheep_share', 0.0, 0.6), ('sell_floor', 0.3, 0.8),
    ('sell_spread', 1.0, 5.0), ('land_cash', 300.0, 3000.0),
    ('max_land', 2.0, 4.0), ('land_fill', 0.05, 0.9),
]
NAMES = [s[0] for s in SPACE]
LO = [s[1] for s in SPACE]
HI = [s[2] for s in SPACE]

# Opponents fitted to real downloaded episodes (tools/fit_opponent.py). The
# old pool all descended from this agent and shared its blind spots, so it
# saturated at 159-1 and could not rank anything. These reproduce the field's
# actual shape -- herds of 7-19, heavy fertilizer collection, 1-3 quadrants --
# and the agent does NOT dominate them, so fitness discriminates again.
POOL = ['opponents/real_sansm1.py', 'opponents/real_nono00.py',
        'opponents/real_niulai.py', 'opponents/real_rohin_kethipally.py',
        'opponents/real_ronel_abraham_math.py', 'opponents/crops_v5.py']
_OPP = {}


def _load(rel):
    if rel not in _OPP:
        spec = importlib.util.spec_from_file_location('o%d' % abs(hash(rel)), ROOT / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _OPP[rel] = m.agent
    return _OPP[rel]


def evaluate(job):
    """Mean coin margin over the pool, both seats. Margin, not win rate: the
    agent already beats this pool ~99% of the time, so wins cannot rank it."""
    theta, seeds = job
    agent_mod.TUNE.update({NAMES[i]: theta[i] for i in range(len(NAMES))})
    marg = []
    for rel in POOL:
        opp = _load(rel)
        for s in seeds:
            r = fastsim.run_episode(agent_mod.agent, opp, seed=s)
            marg.append(r['scores'][0] - r['scores'][1])
            r = fastsim.run_episode(opp, agent_mod.agent, seed=s)
            marg.append(r['scores'][1] - r['scores'][0])
    return statistics.mean(marg)


def clamp(v):
    return [min(HI[i], max(LO[i], v[i])) for i in range(len(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=14)
    ap.add_argument('--pop', type=int, default=24)
    ap.add_argument('--elite', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--output', default='results/policy_tune.json')
    args = ap.parse_args()

    rng = random.Random(4)
    base = [agent_mod.TUNE[n] for n in NAMES]
    mu = list(base)
    sigma = [(HI[i] - LO[i]) * 0.25 for i in range(len(NAMES))]
    best = (-1e18, list(base))
    history = []

    for it in range(args.iters):
        seeds = list(range(9000 + it * args.seeds, 9000 + (it + 1) * args.seeds))
        # The incumbent is re-scored on this generation's seeds, so promotion
        # always compares like with like rather than across seed blocks.
        pop = [list(base), clamp(mu)]
        while len(pop) < args.pop:
            pop.append(clamp([rng.gauss(mu[i], sigma[i]) for i in range(len(NAMES))]))

        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers or os.cpu_count()) as ex:
            scored = list(ex.map(evaluate, [(p, seeds) for p in pop], chunksize=1))

        ranked = sorted(zip(scored, pop), key=lambda z: -z[0])
        elite = [p for _, p in ranked[:args.elite]]
        mu = [sum(e[i] for e in elite) / len(elite) for i in range(len(NAMES))]
        sigma = [max((HI[i] - LO[i]) * 0.03,
                     math.sqrt(sum((e[i] - mu[i]) ** 2 for e in elite) / len(elite)))
                 for i in range(len(NAMES))]
        if ranked[0][0] > best[0]:
            best = (ranked[0][0], list(ranked[0][1]))
        history.append({'iter': it, 'best': round(ranked[0][0]),
                        'incumbent': round(scored[0]),
                        'median': round(statistics.median(scored))})
        print(f'iter {it:3d}  best {ranked[0][0]:9,.0f}  incumbent {scored[0]:9,.0f}  '
              f'median {statistics.median(scored):9,.0f}  ({time.perf_counter()-t:.0f}s)',
              flush=True)
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'best_margin': best[0],
            'params': {NAMES[i]: round(best[1][i], 4) for i in range(len(NAMES))},
            'history': history}, indent=2))

    print(f'\nbest mean margin {best[0]:,.0f}')
    for i, n in enumerate(NAMES):
        if abs(best[1][i] - base[i]) > 0.05 * (HI[i] - LO[i]):
            print(f'  {n:16s} {base[i]:10.2f} -> {best[1][i]:10.2f}')
    print(f'\nwrote {args.output} -- validate with eval/league.py before shipping')


if __name__ == '__main__':
    main()
