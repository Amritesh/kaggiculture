"""PPO on the batched simulator: the policy plays from weights, on GPU.

The agent in agent/main.py decides with hand-written scoring. This trains a
network that decides on its own -- it sees the tile under each worker and its
neighbourhood, and emits an action. Nothing in the loop consults the planner.

What makes it trainable at all is throughput. sim/vecsim.py advances thousands
of farms per tensor op, so a batch of 4,096 costs roughly what one Python
episode costs: ~500x more experience per second, which is the difference
between 30k episodes a day and tens of millions.

Reward shaping: the season's money is the objective, but a policy that only
ever sees "final coins" learns nothing for thousands of episodes, because a
single scalar has to explain ~4,000 decisions. So each turn also pays the
change in FARM VALUE -- standing crops and livestock priced at what they will
produce. Keeping a cow alive is then immediately worth something, and the long
chain (plant, water, water, harvest, sell) is rewarded at every link instead of
only at the end. The terminal term is still the money, so the shaping cannot
change what is ultimately optimal.

    python train/rl/gpu_train.py --device cuda --batch 4096 --iters 2000
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim'))
from vecsim import VecFarms, price, CROP, ANIMAL, N, TPD, DAYS, PROD_NAMES

N_ACT = 15


class Net(nn.Module):
    """Per-worker actor-critic over a small local observation."""

    def __init__(self, n_in, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.pi = nn.Linear(hidden, N_ACT)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def observe(f):
    """(B, U, F) observation: what the worker stands on, what it carries, and
    where it is in the season. Deliberately local -- a global board encoding
    would make the network far larger for little gain at this scale."""
    B, U = f.B, f.U
    g = lambda t: torch.gather(t, 1, f.pos).float()
    kind = g(f.kind)
    crop = g(f.crop)
    animal = g(f.animal)
    feats = [
        kind / 4.0, (crop + 1) / 5.0, (animal + 1) / 3.0,
        g(f.yield_u) / 6.0, g(f.watered), g(f.dry) / 2.0,
        g(f.fed), g(f.cared), g(f.fert),
        (float(f.day) / DAYS) * torch.ones(B, U, device=f.dev),
        ((f.step_i % TPD) / TPD) * torch.ones(B, U, device=f.dev),
        (f.pos // N).float() / N, (f.pos % N).float() / N,
        f.carry[:, :, 0] / 10.0, f.carry.sum(-1) / 20.0,
        (f.money / 20000.0).unsqueeze(1).expand(B, U),
        (f.shed.sum(1) / 100.0).unsqueeze(1).expand(B, U),
    ]
    return torch.stack(feats, dim=-1)


def farm_value(f):
    """Coins the standing farm will still produce, used for reward shaping."""
    dev = f.dev
    p = price(f.market)
    C, A = CROP.to(dev), ANIMAL.to(dev)
    ci = torch.clamp(f.crop, min=0)
    plant = ((f.kind == f.PLANT).float()
             * torch.gather(p, 1, ci.clamp(max=4)) * (f.yield_u + 1.0)).sum(1)
    ai = torch.clamp(f.animal, min=0)
    pidx = A[ai, 4].long()
    animal = ((f.animal >= 0).float()
              * torch.gather(p, 1, pidx) * (f.yield_u + 1.0)).sum(1)
    stock = (f.shed * p).sum(1) + (f.carry.sum(1) * p).sum(1)
    return plant + animal + stock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--horizon', type=int, default=TPD * DAYS)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--gamma', type=float, default=0.999)
    ap.add_argument('--shape', type=float, default=0.01,
                    help='weight on the farm-value shaping term')
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--units', type=int, default=8)
    ap.add_argument('--out', default='results/gpu_policy.pt')
    args = ap.parse_args()

    dev = (('cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu')
           if args.device == 'auto' else args.device)
    probe = VecFarms(2, args.units, device=dev)
    n_in = observe(probe).shape[-1]
    net = Net(n_in).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    print(f'device {dev}  batch {args.batch}  obs {n_in}  '
          f'params {sum(p.numel() for p in net.parameters()):,}', flush=True)

    for it in range(args.iters):
        t0 = time.perf_counter()
        f = VecFarms(args.batch, args.units, device=dev, seed=it)
        obs_l, act_l, logp_l, val_l, rew_l = [], [], [], [], []
        v_prev = farm_value(f)
        with torch.no_grad():
            for t in range(args.horizon):
                o = observe(f)
                logits, v = net(o)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                dm = f.step(a)
                f.sell_all() if (t % TPD == TPD - 1) else None
                v_now = farm_value(f)
                # One reward per farm, shared equally by its workers.
                r = (dm + args.shape * (v_now - v_prev)) / f.U
                v_prev = v_now
                obs_l.append(o); act_l.append(a)
                logp_l.append(dist.log_prob(a)); val_l.append(v)
                rew_l.append(r.unsqueeze(1).expand(f.B, f.U))
        final = f.money
        # Discounted returns with the terminal money folded in.
        R = torch.zeros(args.batch, args.units, device=dev)
        R += (final / 1000.0).unsqueeze(1)
        rets = []
        for t in reversed(range(args.horizon)):
            R = rew_l[t] / 1000.0 + args.gamma * R
            rets.append(R)
        rets.reverse()

        O = torch.stack(obs_l).reshape(-1, n_in)
        Ac = torch.stack(act_l).reshape(-1)
        LP = torch.stack(logp_l).reshape(-1)
        RT = torch.stack(rets).reshape(-1)
        RT = (RT - RT.mean()) / (RT.std() + 1e-6)

        idx = torch.randperm(O.shape[0], device=dev)[:262144]
        for _ in range(args.epochs):
            logits, v = net(O[idx])
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(Ac[idx])
            adv = RT[idx] - v.detach()
            ratio = torch.exp(lp - LP[idx])
            pg = -torch.min(ratio * adv,
                            torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
            loss = pg + 0.5 * F.mse_loss(v, RT[idx]) - args.entropy * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        eps_s = args.batch / (time.perf_counter() - t0)
        print(f'iter {it:4d}  money {final.mean():9,.0f}  best {final.max():9,.0f}  '
              f'loss {loss.item():+.3f}  {eps_s:6.0f} episodes/s', flush=True)
        if it % 20 == 0:
            torch.save({'state': net.state_dict(), 'n_in': n_in, 'iter': it},
                       ROOT / args.out)
    torch.save({'state': net.state_dict(), 'n_in': n_in, 'iter': args.iters},
               ROOT / args.out)
    print(f'\nwrote {args.out} -- validate in the real engine before shipping')


if __name__ == '__main__':
    main()
