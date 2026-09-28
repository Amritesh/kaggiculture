"""Learn the autoregressive action-scoring weights W in main_v7.

The policy picks one unit's action at a time, and each candidate is scored with
the assignments already made this turn as input. W multiplies the hand-built
heuristic by exp(W . phi), so W = 0 is exactly the heuristic and training starts
from a known-good policy rather than noise.

Search is an evolution strategy scored by **paired head-to-head against a frozen
champion**, not by win rate against a mixed pool. Two reasons:

* Paired: every candidate plays the same seeds, in both seats, against the same
  opponent, so seed luck cancels between candidates.
* Head-to-head: "does this beat the thing we would otherwise ship" is the
  question that decides a promotion, so measure it directly.

The champion is re-frozen whenever a candidate beats it at p < 0.05, which makes
this iterated self-improvement rather than a single hill climb.

    python train_policy.py --iters 20 --pop 12 --seeds 80
"""
import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
sys.path.insert(0, str(ROOT / 'agent'))

import fastsim
import main as main_v7

NW = main_v7.NW


def sign_test(w, l):
    """Two-sided probability of a split this lopsided under a fair coin."""
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(w, l) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _play(job):
    """One candidate against the frozen champion over a shared seed set."""
    cand, champ, seeds = job
    wins = losses = 0
    margin = 0.0

    # Both sides run main_v7; each wrapper installs its own weights immediately
    # before the call, so the two policies share a module without interfering.
    def me(obs, cfg=None):
        main_v7.W[:] = cand
        return main_v7.agent(obs, cfg)

    def them(obs, cfg=None):
        main_v7.W[:] = champ
        return main_v7.agent(obs, cfg)

    for s in seeds:
        for seat in (0, 1):
            if seat:
                r = fastsim.run_episode(them, me, seed=s)
                mine, theirs = r['scores'][1], r['scores'][0]
            else:
                r = fastsim.run_episode(me, them, seed=s)
                mine, theirs = r['scores'][0], r['scores'][1]
            margin += mine - theirs
            if mine > theirs:
                wins += 1
            elif mine < theirs:
                losses += 1
    return wins, losses, margin / max(1, len(seeds) * 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--elite', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=80,
                    help='seeds per candidate; 80 seeds x 2 seats = 160 games')
    ap.add_argument('--sigma', type=float, default=0.35)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--output', default='results/policy.json')
    args = ap.parse_args()

    rng = random.Random(7)
    # Start from whatever main_v7 currently ships, not from zero: the shipped
    # weights are already validated, and restarting at the heuristic would throw
    # away every promotion so far.
    champ = list(main_v7.W)
    mu = list(main_v7.W)
    sigma = [args.sigma] * NW
    workers = args.workers or os.cpu_count()
    history = []
    promotions = 0

    for it in range(args.iters):
        lo = 3000 + it * args.seeds
        seeds = list(range(lo, lo + args.seeds))

        pop = [list(mu)]
        while len(pop) < args.pop:
            pop.append([rng.gauss(mu[i], sigma[i]) for i in range(NW)])

        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            scored = list(ex.map(_play, [(c, champ, seeds) for c in pop], chunksize=1))

        ranked = sorted(zip(scored, pop), key=lambda z: (-(z[0][0] - z[0][1]), -z[0][2]))
        (bw, bl, bm), best = ranked[0]
        p = sign_test(bw, bl)

        elite = [c for _, c in ranked[:args.elite]]
        if bw > bl:
            mu = [sum(e[i] for e in elite) / len(elite) for i in range(NW)]
            sigma = [max(0.05, math.sqrt(sum((e[i] - mu[i]) ** 2 for e in elite) / len(elite)))
                     for i in range(NW)]
        else:
            # Nothing beat the champion, so the elite are all losers: averaging
            # them walks the search centre downhill. Hold position and look
            # closer in instead.
            mu = list(champ)
            sigma = [max(0.04, x * 0.7) for x in sigma]

        promoted = False
        if bw > bl and p < 0.05:
            champ = list(best)      # iterate: beat the thing we would ship
            # Re-centre the search on the champion. Leaving mu at the elite mean
            # lets it drift away from the best policy found, and every later
            # candidate is then sampled around something the champion already
            # beats -- generation 5 lost 51-109 for exactly this reason.
            mu = list(best)
            sigma = [args.sigma] * NW      # re-open the search after a move
            promotions += 1
            promoted = True

        history.append({'iter': it, 'w': bw, 'l': bl, 'margin': round(bm, 1),
                        'p': round(p, 5), 'promoted': promoted})
        print(f'iter {it:3d}  best {bw:3d}-{bl:<3d} vs champion  margin {bm:+8.0f}  '
              f'p={p:.4f}{"  PROMOTED" if promoted else ""}  ({time.perf_counter()-t:.0f}s)',
              flush=True)

        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {'W': champ, 'mu': mu, 'promotions': promotions, 'history': history}, indent=2))

    print(f'\n{promotions} promotion(s). Final W:')
    print('  [' + ', '.join(f'{v:.4f}' for v in champ) + ']')
    print(f'\nwrote {out}')
    print('Confirm in league.py over 100+ games before shipping.')


if __name__ == '__main__':
    main()
