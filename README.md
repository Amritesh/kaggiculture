# Kaggriculture agent

Agent and tooling for the 2026 Kaggriculture simulation competition.
Final submission deadline: 30 September 2026. Five submissions a day; the
competition scores your latest two agents.

## Layout

```
agent/main.py        the agent that ships (standard library only)
sim/fastsim.py       fast exact episode driver, 13.6x the official wrapper
eval/league.py       round-robin measurement with a significance test
train/search.py      evolution search over the agent's policy weights
train/promote.py     validate on unseen seeds, then ship
tools/package.py     build submission.tar.gz
opponents/           benchmark agents (see below)
tests/               invariants the agent must not break
vendor/              unmodified official engine, commit 9b6bedea
legacy/              earlier agent versions, kept for reference only
```

## Run

```sh
python -m unittest discover -s tests            # 6 invariants
python eval/league.py --all --seeds 30          # measure everything
python train/search.py --iters 10 --seeds 80    # train
python train/promote.py --ship                  # validate, then ship
python tools/package.py                         # rebuild submission.tar.gz
```

## Submit

Kaggle authentication is not configured here. Sign in and accept the rules at
https://www.kaggle.com/competitions/kaggriculture, then:

```sh
kaggle competitions submit kaggriculture -f submission.tar.gz -m "message"
```

Never put API credentials in this repository.

## Current standing

80 games per matchup, both seats:

| `agent/main.py` vs | record | median margin | p |
|---|---|---|---|
| `opponents/original.py` (was on the leaderboard) | **64-16** | +21,213 | 6e-08 |
| `opponents/crops_v5.py` (best earlier variant) | **73-7** | +23,466 | 6e-15 |

League win rate 0.859 against the original's 0.072. Above 600 on the live
ladder, where the original settled at 514.8.

## How it plays

Full reasoning, with the measurements behind each choice, is in `STRATEGY.md`.
In short:

1. **Livestock.** ~168 milk a season at roughly 315/unit into a market no other
   agent supplies. A cow returns ~160 coins per worker-action against melon's ~118.
2. **A learned scoring policy.** Units are assigned one at a time, each candidate
   scored with the assignments already made this turn as input. Weights multiply
   the rule-based score, so zero weights reproduce the plain heuristic exactly
   and training can only move away from something that already works.
3. **Opponent supply counted in full.** Their standing crops hit the same market
   as ours; discounting them mispriced every decision.
4. **Labour timing.** Full crew engaged early; eight hands is a hard economic ceiling.
5. **Metered selling.** Dumping the shed each turn walked MELON from 272 to 60.

## Opponents

`league.py --all` plays everything in `opponents/`. The mix matters: a pool of
near-clones will confidently recommend the wrong parameter, which happened here
more than once. `original.py` and `crops_v5.py` are genuinely different agents;
`crop_only.py`, `dumper.py` and `landgrab.py` are the shipped agent with
deliberately distorted strategies.

## Two rules worth keeping

**Measure with enough games.** Win rate over N games has a standard error of
`sqrt(0.25/N)`. At 72 games that is 5.9%, so nothing under a ~12-point gap is
real, while genuine differences here are 1-5 points. A four-seed sweep once
showed 0/4 where 120 games showed dead level.

**Suspect the harness before the agent.** Four of the largest findings here were
bugs in the measurement, not the policy: the league swapped scores on seat-1
games and hid a 0.95 agent as 0.50; the search bounded a parameter so its
optimum was unreachable and looked converged on the boundary; the trainer
drifted off its champion and shipped a regression that had to be rolled back.
Every one presented as "the agent has plateaued".

## Sources

- https://www.kaggle.com/competitions/kaggriculture/overview
- https://github.com/Kaggle/kaggle-environments/tree/9b6bedeaeb478067a335ff5776b214fb436218f6/kaggle_environments/envs/kaggriculture
