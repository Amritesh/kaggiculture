# How the agent plays

Reasoning behind `agent/main.py`, with the measurement behind each choice.

## The economics

Price is a closed-form function of market inventory, so the agent computes it
exactly rather than learning it:

```
price(inv) = base ± amp · f(|inv − 10000|),  floored at 1
```

Below the anchor is scarcity and price rises; above it is glut and price falls.
Each product has its own depth `T` and curve shape, and that asymmetry drives
every market decision:

| product | base | T | glut shape | consequence |
|---|---|---|---|---|
| MELON | 250 | 300 | quadratic | richest, collapses fastest |
| MILK | 160 | 122 | linear | thin, high value |
| WOOL | 200 | 105 | quadratic | thinnest |
| STRAWBERRY | 120 | 100 | linear | thin |
| WHEAT | 25 | 400 | log | deep, absorbs volume |
| CARROT | 35 | 450 | sqrt | deepest |

Town shops consume on a fixed tick all season, draining inventory below the
anchor, so goods the town wants sit permanently above base price.

**Labour binds, not land.** A quadrant costs 4,000; the 13th hand costs ~7,000 a
season on the Fibonacci hire curve. Ranked per worker-action:

| activity | coins/action |
|---|---|
| cow, fed and cared | ~160 |
| melon | ~118 |
| sheep | ~100 |
| strawberry / goose / carrot | ~27 / ~27 / ~17 |

Two rules make watering cheaper than it looks: it only adds yield inside
`(max_yield_day+1)//2 ≤ age ≤ max_yield_day`, and a plant dies only after two
dry days, so survival needs watering every other day. Workers never need the
shed — end-of-day auto-drop banks their inventory for free.

## What the agent does

**Livestock.** Eight cows near the shed, fed and cared for daily, harvested at
5+ units: ~168 milk a season at roughly 315/unit into a market no other agent
supplies. `CARE` banks a +1 bonus consumed by the next *fed* production tick, so
feeding is daily — every other day was measured at 112 milk against 168. Cows
only: cow+sheep measured 0.550 against cow-only 0.900.

**Metered selling.** Sells only while the marginal price holds above 0.51 × base,
clearing stock before the season ends. Dumping the shed each turn walked MELON
from 272 down to 60.

**Opponent supply counted in full.** The forecast counts both farms' standing
crops. Discounting the opponent's to 0.569 treated half their harvest as if it
would never reach the market; at 1.00 it is 362-38 against 254-146 on a
non-clone pool. Sharply peaked at exactly 1.0 — a structurally correct value,
not a fitted one.

**Labour timing.** Full crew engaged once six plants stand. Eight hands is a hard
ceiling: at ten the farm runs out of wheat money by day 8 and has lost the whole
herd by day 22, because feed is a standing daily bill and hiring cannibalises it.

**A learned scoring policy.** Units are assigned one at a time and each candidate
is scored with the assignments already made this turn as input (crowding, and
how many units already chose that op). The score is the rule-based value times
`exp(W · φ)` over 26 features, so `W = 0` reproduces the plain heuristic exactly
and training can only move away from something that already works. Nine
promotions, each validated on unseen seeds.

Its largest single find was flipping `fetch` from +0.86 to −2.38, undoing
hand-set constants (`PLACE` 40000, `PICKUP` 20000) that were ~10x too high and
pulling the crew onto logistics ahead of farming.

## Rejected, each measured

- **Fertilizer.** Helps only wheat (4→6) and carrot (3→4); melon, tomato and
  strawberry already cap out from ordinary watering.
- **Fewer plantings.** A third of crops rot, but capping sowing cut score from
  73,570 to 62,502. A seed costs 10–100 and pays at two-thirds survival, so
  over-planting is correct.
- **Group routing, zoning, global assignment.** All cut walking and cut work
  harder. There is no contention to resolve: the crew finishes by mid-afternoon
  and idles 47% of late-day turns, and raising the seed cap to fill that idle
  time loses decisively. The idle afternoon is the farm declining unprofitable
  work.
- **More hands, more land, sheep, adaptive livestock.** All measured worse.
- **End-to-end policy learning.** A network replacing every hand-set value,
  trained by DAgger, scored 0–233 against a working agent's 100,000 — worse than
  untrained, which at least does nothing. A farm is a *chain feasibility*
  problem: plant→water→water→water→harvest dies on one miss, and at the ~10%
  top-1 accuracy a distilled policy reaches, a 20-action chain completes with
  probability 1e-20. You need ~99%. The agent works precisely because rules
  guarantee the chains and the network only reprioritises within them.

## Limits

31% of worker-turns are productive against a ~43% structural ceiling — hands
respawn at the shed each morning and must walk out. It supplies ~27% of what the
town absorbs, and uses under 1% of its per-turn compute budget. Three
independent search methods now agree the weights have converged.

Every number here is measured against agents descended from this one. That pool
has recommended the wrong answer more than once; real opponent replays would
remove the weakness rather than work around it.
