"""Parallel round-robin evaluation across a pool of agents.

Replaces "10/10 against the starter" with a dominance test over every variant
we have. Each pairing is played on the same seed set in both seats, so seat
advantage cancels.

    python league.py --agents main.py research/crops_v1.py --seeds 16
    python league.py --all --seeds 32 --output results/league.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))

import fastsim

_CACHE = {}


def load(path):
    """Load an agent callable from a file path, memoised per process."""
    if path in _CACHE:
        return _CACHE[path]
    if path == 'starter':
        from kaggle_environments.envs.kaggriculture import kaggriculture as K
        fn = K.starter_agent if hasattr(K, 'starter_agent') else None
        if fn is None:
            import kaggle_environments
            fn = kaggle_environments.agent.build_agent  # pragma: no cover
        _CACHE[path] = fn
        return fn
    spec = importlib.util.spec_from_file_location('ag_%d' % abs(hash(path)), ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[path] = mod.agent
    return mod.agent


def _play(job):
    a_path, b_path, seed, seat = job
    a, b = load(a_path), load(b_path)
    t = time.perf_counter()
    if seat:
        # a plays seat 1, so a's score is scores[1].
        r = fastsim.run_episode(b, a, seed=seed)
        a_score, b_score = r['scores'][1], r['scores'][0]
    else:
        r = fastsim.run_episode(a, b, seed=seed)
        a_score, b_score = r['scores'][0], r['scores'][1]
    return {'a': a_path, 'b': b_path, 'seed': seed, 'seat': seat,
            'a_score': a_score, 'b_score': b_score,
            'a_win': a_score > b_score, 'tie': a_score == b_score,
            'seconds': round(time.perf_counter() - t, 2)}


def run(agents, seeds, workers=None):
    jobs = []
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            for s in seeds:
                for seat in (0, 1):
                    jobs.append((a, b, s, seat))
    workers = workers or os.cpu_count()
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_play, jobs, chunksize=4):
            out.append(r)
    return out


def summarise(agents, games):
    """Per-agent win rate plus the head-to-head matrix."""
    rec = {a: {'w': 0, 'l': 0, 't': 0, 'coins': 0.0, 'n': 0} for a in agents}
    h2h = {a: {b: [0, 0] for b in agents if b != a} for a in agents}
    for g in games:
        a, b = g['a'], g['b']
        rec[a]['coins'] += g['a_score']; rec[a]['n'] += 1
        rec[b]['coins'] += g['b_score']; rec[b]['n'] += 1
        if g['tie']:
            rec[a]['t'] += 1; rec[b]['t'] += 1
        elif g['a_win']:
            rec[a]['w'] += 1; rec[b]['l'] += 1
            h2h[a][b][0] += 1; h2h[b][a][1] += 1
        else:
            rec[b]['w'] += 1; rec[a]['l'] += 1
            h2h[b][a][0] += 1; h2h[a][b][1] += 1
    table = []
    for a in agents:
        r = rec[a]
        played = r['w'] + r['l'] + r['t']
        table.append({
            'agent': a, 'played': played,
            'winrate': round(r['w'] / played, 4) if played else 0.0,
            'w': r['w'], 'l': r['l'], 't': r['t'],
            'mean_coins': round(r['coins'] / r['n'], 1) if r['n'] else 0.0,
        })
    table.sort(key=lambda d: (-d['winrate'], -d['mean_coins']))
    return table, h2h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agents', nargs='*', default=None)
    p.add_argument('--all', action='store_true', help='main.py plus every research/*.py variant')
    p.add_argument('--seeds', type=int, default=16, help='number of seeds (0..n-1)')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--output', default='results/league.json')
    args = p.parse_args()

    if args.all:
        agents = ['agent/main.py'] + sorted(
            str(q.relative_to(ROOT)) for q in (ROOT / 'opponents').glob('*.py'))
    else:
        agents = args.agents or ['agent/main.py']

    seeds = list(range(args.seeds))
    t = time.perf_counter()
    games = run(agents, seeds, args.workers)
    table, h2h = summarise(agents, games)
    elapsed = time.perf_counter() - t

    print(f'{len(games)} games, {len(agents)} agents, {elapsed:.1f}s '
          f'({len(games)/elapsed:.1f} games/s)\n')
    print(f'{"agent":32s} {"winrate":>8s} {"W-L-T":>12s} {"mean coins":>11s}')
    for r in table:
        wlt = '%d-%d-%d' % (r['w'], r['l'], r['t'])
        print(f'{r["agent"]:32s} {r["winrate"]:8.3f} {wlt:>12s} {r["mean_coins"]:11.0f}')

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {'table': table, 'h2h': h2h, 'seeds': seeds, 'games': games}, indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
