"""Roll out episodes and label every decision with the profit that followed it.

The agent proposes a set of legal tasks each turn and picks one. What it has
never had is evidence about the tasks it did *not* pick, so exploration is
mixed in: with probability EXPLORE a unit takes a uniformly random legal task.
Without that the dataset only ever describes the heuristic's own habits and a
model fitted to it cannot discover anything the heuristic does not already do.

The label is credit assignment over the money curve. Income here is extremely
lumpy -- roughly 86% of a season's coins arrive on 20 of 720 turns, because
selling is metered and a cow bought on day 4 pays nothing until day 12 -- so a
per-turn reward is almost always zero and tells the model nothing. Instead each
decision is labelled with the discounted sum of all future money changes, which
is what "this action led to profit" actually means when the payoff is delayed.

    python train/attn/collect.py --episodes 60 --out results/attn_data.npz
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim'))
sys.path.insert(0, str(ROOT / 'agent'))

MAX_CAND = 48          # candidate sets are padded/truncated to this width
GAMMA = 0.997          # per-turn discount; ~0.1 weight 720 turns out


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rollout(job):
    seed, opp_rel, explore, stride = job
    import fastsim
    agent_mod = _load('agent/main.py', 'a_%d' % seed)
    opp = _load(opp_rel, 'o_%d' % seed)

    agent_mod.EXPLORE = explore
    agent_mod._RNG.seed(seed * 7919 + 13)
    sample = []
    agent_mod.SAMPLE = sample

    money = []

    def hook(step, state):
        money.append(float(state[0].observation.farms[0]['money']))

    fastsim.run_episode(agent_mod.agent, opp.agent, seed=seed, hooks=[hook])
    agent_mod.SAMPLE = None
    agent_mod.EXPLORE = 0.0

    # Discounted future money change, computed backwards over the money curve.
    delta = np.diff(np.asarray(money, dtype=np.float64), prepend=money[0])
    future = np.zeros_like(delta)
    run = 0.0
    for i in range(len(delta) - 1, -1, -1):
        run = delta[i] + GAMMA * run
        future[i] = run

    feats, masks, chosen, targets, steps = [], [], [], [], []
    for i, (step, day, hour, phis, pick) in enumerate(sample):
        if pick < 0 or i % stride:
            continue
        n = min(len(phis), MAX_CAND)
        if n < 2 or pick >= n:
            continue
        nw = len(phis[0])
        f = np.zeros((MAX_CAND, nw), dtype=np.float32)
        m = np.zeros(MAX_CAND, dtype=np.float32)
        for j in range(n):
            f[j] = phis[j]
            m[j] = 1.0
        feats.append(f)
        masks.append(m)
        chosen.append(pick)
        targets.append(future[min(step, len(future) - 1)])
        steps.append(step)
    if not feats:
        return None
    return (np.stack(feats), np.stack(masks), np.asarray(chosen, np.int64),
            np.asarray(targets, np.float32), np.asarray(steps, np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--explore', type=float, default=0.15)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--out', default='results/attn_data.npz')
    ap.add_argument('--opponents', nargs='*',
                    default=['opponents/crops_v5.py', 'opponents/original.py',
                             'opponents/crop_only.py', 'opponents/landgrab.py'])
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor
    jobs = [(5000 + i, args.opponents[i % len(args.opponents)], args.explore, args.stride)
            for i in range(args.episodes)]
    out = []
    with ProcessPoolExecutor(max_workers=args.workers or os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(rollout, jobs)):
            if r:
                out.append(r)
            print(f'  episode {i+1}/{len(jobs)}', end='\r', flush=True)

    feats = np.concatenate([o[0] for o in out])
    masks = np.concatenate([o[1] for o in out])
    chosen = np.concatenate([o[2] for o in out])
    targets = np.concatenate([o[3] for o in out])
    steps = np.concatenate([o[4] for o in out])
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, feats=feats, masks=masks, chosen=chosen,
                        targets=targets, steps=steps)
    print(f'\n{len(feats):,} decisions  candidates/turn {masks.sum(1).mean():.1f}  '
          f'target mean {targets.mean():,.0f} sd {targets.std():,.0f}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
