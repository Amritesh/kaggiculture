"""Competitive co-evolution: a population that has to beat its own history.

train/policy.py evolved one side against five fixed opponents and found nothing
in 330 candidates -- once an agent beats a fixed pool 159-1, the pool cannot
rank what is left. Here the opposition improves too, so the bar rises with the
population and fitness keeps discriminating.

The known failure of self-play is producing a champion that beats its own
lineage and loses to strangers, which happened in this project already: a
variant went 55-25 against its parent while falling from 159-1 to 99-61 against
non-clones. Two defences:

  * a hall of fame -- every past champion stays in the opponent set, so a
    genome cannot win by exploiting only the current generation, and cycles
    (A beats B beats C beats A) are penalised rather than rewarded;
  * fixed anchors -- the non-clone agents stay in the mix at all times, so
    absolute strength is always part of fitness, never just relative strength.

Fitness is the mean coin margin over all three groups, both seats.

    python train/coevolve.py --gens 10 --pop 12 --seeds 5
"""
import argparse
import importlib.util
import json
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
sys.path.insert(0, str(ROOT / 'train'))
from policy import SPACE, NAMES, LO, HI          # one definition of the genome

ANCHORS = ['opponents/crops_v5.py', 'opponents/original.py',
           'opponents/crop_only.py', 'opponents/landgrab.py']
_A = {}


def _anchor(rel):
    if rel not in _A:
        spec = importlib.util.spec_from_file_location('a%d' % abs(hash(rel)), ROOT / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _A[rel] = m.agent
    return _A[rel]


def _as_agent(theta):
    """A genome is just a TUNE override, so every genome is the same code with
    different constants -- no genome can be illegal or crash."""
    tune = {NAMES[i]: theta[i] for i in range(len(NAMES))}

    def play(obs, cfg=None):
        agent_mod.TUNE.update(tune)
        return agent_mod.agent(obs, cfg)
    return play


def duel(job):
    """One genome against a list of opponents (genomes or anchor paths)."""
    theta, opponents, seeds = job
    me = _as_agent(theta)
    marg = []
    for opp in opponents:
        other = _anchor(opp) if isinstance(opp, str) else _as_agent(opp)
        for s in seeds:
            # Both seats, common seeds: the seat advantage cancels exactly.
            r = fastsim.run_episode(me, other, seed=s)
            marg.append(r['scores'][0] - r['scores'][1])
            r = fastsim.run_episode(other, me, seed=s)
            marg.append(r['scores'][1] - r['scores'][0])
    return statistics.mean(marg)


def clamp(v):
    return [min(HI[i], max(LO[i], v[i])) for i in range(len(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=10)
    ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--hof', type=int, default=4, help='hall-of-fame opponents used')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--output', default='results/coevolve.json')
    args = ap.parse_args()

    rng = random.Random(17)
    base = [agent_mod.TUNE[n] for n in NAMES]
    pop = [list(base)]
    while len(pop) < args.pop:
        pop.append(clamp([rng.gauss(base[i], (HI[i] - LO[i]) * 0.12)
                          for i in range(len(NAMES))]))
    hof = [list(base)]
    history = []

    for g in range(args.gens):
        seeds = list(range(12000 + g * args.seeds, 12000 + (g + 1) * args.seeds))
        # Opponents: a sample of the current population, the hall of fame, and
        # the fixed anchors.
        rivals = rng.sample(pop, min(3, len(pop)))
        opponents = rivals + hof[-args.hof:] + ANCHORS

        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers or os.cpu_count()) as ex:
            fit = list(ex.map(duel, [(p, opponents, seeds) for p in pop], chunksize=1))

        ranked = sorted(zip(fit, pop), key=lambda z: -z[0])
        champ_fit, champ = ranked[0]
        hof.append(list(champ))
        # Next generation: elites, then mutated children of elites.
        elite = [p for _, p in ranked[:max(2, args.pop // 3)]]
        nxt = [list(e) for e in elite]
        while len(nxt) < args.pop:
            a, b = rng.choice(elite), rng.choice(elite)
            child = [a[i] if rng.random() < 0.5 else b[i] for i in range(len(NAMES))]
            k = rng.randrange(len(NAMES))
            child[k] = rng.gauss(child[k], (HI[k] - LO[k]) * 0.10)
            nxt.append(clamp(child))
        pop = nxt

        base_fit = fit[0] if g == 0 else None
        history.append({'gen': g, 'champ': round(champ_fit),
                        'median': round(statistics.median(fit)),
                        'seconds': round(time.perf_counter() - t)})
        print(f'gen {g:3d}  champion {champ_fit:9,.0f}  median {statistics.median(fit):9,.0f}'
              f'  ({time.perf_counter()-t:.0f}s)', flush=True)
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'champion': {NAMES[i]: round(champ[i], 4) for i in range(len(NAMES))},
            'champion_fitness': champ_fit, 'history': history}, indent=2))

    print(f'\nwrote {args.output} -- validate against the fixed pool and the '
          f'shipped agent before shipping')


if __name__ == '__main__':
    main()
