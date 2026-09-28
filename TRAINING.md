# Training on a GPU machine

The agent is a scored-task planner; the RL part learns a **residual** on top of
it. That matters for what the GPU can and cannot do here — see *What to expect*
at the bottom before booking an expensive box.

## 1. Setup

```bash
git clone https://github.com/Amritesh/kaggiculture.git
cd kaggiculture
PY=python3 ./setup.sh                 # venv + torch + numpy + vendored engine
```

For a CUDA box, install the matching torch build first, then re-run setup:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
./setup.sh
```

Check the device is seen:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Train

```bash
# short run, confirms the loop works end to end (~5 min)
.venv/bin/python train/rl/train.py --iters 5 --episodes 24 --temp 0.1 --device cuda

# full run
.venv/bin/python train/rl/train.py \
    --iters 400 --episodes 64 --temp 0.1 --epochs 8 \
    --d-model 64 --layers 2 --lr 3e-4 --device cuda \
    --out results/rl_policy.pt
```

Resume an interrupted run:

```bash
.venv/bin/python train/rl/train.py --resume results/rl_policy.pt --seed0 500000 \
    --iters 400 --episodes 64 --device cuda
```

Rollouts are CPU-bound (the engine is Python), so **more CPU cores matter more
than a bigger GPU**. Pick a box with many cores; episodes scale with
`os.cpu_count()`.

## 3. Other searches

```bash
# random-forest surrogate over the agent's constants (this produced the
# last shipped gain: 1039-881 over 1,920 games, p=3e-04)
.venv/bin/python train/forest.py --rounds 10 --batch 32 --seeds 8

# cross-entropy method over the same space
.venv/bin/python train/policy.py --iters 14 --pop 24 --seeds 10

# competitive co-evolution with a hall of fame
.venv/bin/python train/coevolve.py --gens 12 --pop 16 --seeds 6
```

## 4. Validate before shipping anything

Nothing goes into a submission on training numbers — they are measured on the
seeds the search selected against and overstate every time.

```bash
.venv/bin/python eval/league.py --all --seeds 30
.venv/bin/python train/promote.py            # report only
.venv/bin/python train/promote.py --ship     # validate, bake in, repackage
.venv/bin/python tools/package.py            # build submission.tar.gz
```

## 5. Add real opponents (the highest-value step)

Download an episode replay from the Kaggle competition page, then:

```bash
.venv/bin/python tools/replay.py ~/Downloads/<episode>.json      # what happened
.venv/bin/python tools/fit_opponent.py ~/Downloads/*.json        # -> opponents/real_*.py
```

Every large gain in this project came from a replay, not from a search. The
local pool all descends from this agent, so it shares its blind spots: it went
159-1 against agents that, like it, never collected fertilizer -- an 11% error
that no amount of self-play could see.

## What to expect

RL here is limited by **episodes per second, not FLOPs**. One episode is ~0.8 s
of pure-Python game logic; sampling a 73k-parameter policy adds a forward pass
per decision, ~4,000 per episode. Measured: **0.34 episodes/sec**, about 30k a
day on 8 cores. AlphaZero-scale training is 10^6-10^8 episodes.

A GPU speeds up the part that was never the bottleneck. What actually helps:

* more CPU cores (rollouts are embarrassingly parallel);
* `--epochs 8` (PPO-style reuse of each expensive batch);
* a vectorised re-implementation of the engine, if anyone wants to spend the
  time -- that is the only route to 10^6 episodes.

A 30-iteration run (720 episodes) produced no measurable learning: margins
oscillated between -9,019 and +1,741 with a per-iteration noise of 3,257. That
is the sample-efficiency gap, not a bug.
