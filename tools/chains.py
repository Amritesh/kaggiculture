"""Trace every crop and animal from first action to final sale.

A farm is a chain: plant, water, water, water, harvest, carry, sell. Each link
costs a worker-action and the whole chain pays nothing if any link is missed --
a plant dies after two dry days, so a chain abandoned halfway is pure loss. The
greedy scorer picks the best single task each turn and has no representation of
"this chain cannot be finished", so this measures how often it starts work it
does not complete, and what that costs.

    python tools/chains.py --seed 42 --opponent opponents/original.py
"""
import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
sys.path.insert(0, str(ROOT / 'agent'))


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--opponent', default='opponents/original.py')
    args = ap.parse_args()

    import fastsim
    agent_mod = _load('agent/main.py', 'chain_agent')
    opp = _load(args.opponent, 'chain_opp')

    # actions[(x,y)] counts worker-actions spent on that tile; born/state track
    # the crop that currently occupies it.
    actions = defaultdict(int)
    born = {}
    finished = []          # (crop, actions, yield_units, outcome)
    prev_tiles = {}
    harvested = defaultdict(int)

    def hook(step, state):
        obs = state[0].observation
        farm = obs.farms[0]
        act = state[0].action or {}
        pos = [farm['farmer']] + list(farm.get('hands', []))
        # A unit standing on a tile and doing farm work spends an action there.
        for i, a in enumerate(act.get('hands', []) + [act.get('farmer')]):
            if not a:
                continue
            if a[0] in ('WATER', 'PLANT', 'HARVEST', 'DIG', 'FEED', 'CARE'):
                if i < len(pos):
                    actions[tuple(pos[i])] += 1
                    # A multi-pick crop is back to yield_units 0 right after a
                    # harvest, so the tile's last observed state cannot tell a
                    # picked plant from a rotted one. Record the harvest itself.
                    if a[0] == 'HARVEST':
                        harvested[tuple(pos[i])] += 1
        cur = {}
        for y, row in enumerate(farm['tiles']):
            for x, t in enumerate(row):
                if isinstance(t, dict) and t.get('kind') == 'PLANT':
                    cur[(x, y)] = (t['crop'], t.get('planted_day'), t.get('yield_units', 0))
        for k, v in cur.items():
            if k not in prev_tiles or prev_tiles[k][1] != v[1]:
                born[k] = (v[0], step, actions[k], harvested[k])
        for k, v in prev_tiles.items():
            if k not in cur or cur[k][1] != v[1]:
                if k in born:
                    crop, t0, a0, h0 = born.pop(k)
                    spent = actions[k] - a0
                    picks = harvested[k] - h0
                    finished.append((crop, spent, picks, 'harvested' if picks else 'lost'))
        prev_tiles.clear(); prev_tiles.update(cur)

    r = fastsim.run_episode(agent_mod.agent, opp.agent, seed=args.seed, hooks=[hook])
    for k, (crop, t0, a0, h0) in born.items():
        picks = harvested[k] - h0
        finished.append((crop, actions[k] - a0, picks,
                         'harvested' if picks else 'standing'))

    print(f'final score {r["scores"][0]:,.0f} vs {r["scores"][1]:,.0f}\n')
    by = defaultdict(lambda: [0, 0, 0, 0])   # n, actions, yield, lost
    for crop, spent, yu, outcome in finished:
        b = by[crop]
        b[0] += 1; b[1] += spent; b[2] += yu
        b[3] += (outcome == 'lost')
    print(f'{"crop":12s} {"chains":>7} {"lost":>6} {"loss%":>6} '
          f'{"actions":>8} {"act/chain":>10} {"picks":>7}')
    tot = [0, 0, 0, 0]
    for crop in sorted(by, key=lambda c: -by[c][1]):
        n, a, y, l = by[crop]
        print(f'{crop:12s} {n:>7} {l:>6} {100*l/max(1,n):>5.0f}% '
              f'{a:>8} {a/max(1,n):>10.1f} {y:>7}')
        for i in range(4):
            tot[i] += by[crop][i]
    n, a, y, l = tot
    print(f'{"ALL":12s} {n:>7} {l:>6} {100*l/max(1,n):>5.0f}% '
          f'{a:>8} {a/max(1,n):>10.1f} {y:>7}')
    wasted = sum(s for c, s, yu, o in finished if o == 'lost')
    print(f'\nworker-actions spent on chains that produced nothing: {wasted:,} '
          f'of {a:,}  ({100*wasted/max(1,a):.0f}%)')


if __name__ == '__main__':
    main()
