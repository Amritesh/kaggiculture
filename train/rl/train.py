"""Policy-gradient RL on the residual, with whole-season credit.

Credit assignment here is the whole problem. A cow bought on day 4 pays nothing
until day 12; a melon sown on day 1 sells on day 11; 86% of a season's coins
arrive on 20 of 720 turns. Any discounting short of the full season hides the
decisions that actually matter, so the return for EVERY decision in an episode
is the final money -- undiscounted. That is the "long chain reward": the signal
is the farm's end state, and the policy has to work out which of its ~4,000
decisions contributed.

Two things make that tractable rather than hopeless:

  * the residual starts at exactly zero, so the policy already farms
    competently and the gradient is about improving a working chain rather
    than discovering one;
  * a learned critic supplies the baseline, and returns are standardised per
    batch, so the advantage is "better or worse than this agent usually does
    from here" instead of a raw six-figure number.

    python train/rl/train.py --iters 20 --episodes 24
"""
import argparse
import importlib.util
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim'))
sys.path.insert(0, str(ROOT / 'agent'))
sys.path.insert(0, str(ROOT / 'train' / 'rl'))

from policy import ResidualPolicy

MAXC = 48
POOL = ['opponents/real_sansm1.py', 'opponents/real_nono00.py',
        'opponents/real_niulai.py', 'opponents/real_ronel_abraham_math.py',
        'opponents/real_igeo_kk.py', 'opponents/real_rohin_kethipally.py']


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rollout(job):
    """Play one episode sampling from the policy; return decisions and score."""
    state_dict, seed, opp_rel, scale, temp, dm, nl = job
    import fastsim
    agent_mod = _load('agent/main.py', 'rl_agent_%d' % seed)
    opp = _load(opp_rel, 'rl_opp_%d' % seed)

    net = ResidualPolicy(d_model=dm, n_layer=nl)
    net.load_state_dict(state_dict)
    net.eval()

    log = []
    torch.manual_seed(seed)

    def policy(phis, base):
        n = min(len(phis), MAXC)
        f = torch.zeros(1, MAXC, len(phis[0]))
        m = torch.zeros(1, MAXC)
        b = torch.full((1, MAXC), -1e9)
        for i in range(n):
            f[0, i] = torch.tensor(phis[i]); m[0, i] = 1.0; b[0, i] = base[i]
        with torch.no_grad():
            logits, _ = net(f, m, b, scale)
        probs = F.softmax(logits[0, :n] / temp, dim=0)
        pick = int(torch.multinomial(probs, 1).item())
        log.append(([list(x) for x in phis[:n]], list(base[:n]), pick))
        return pick

    agent_mod.POLICY = policy
    r = fastsim.run_episode(agent_mod.agent, opp.agent, seed=seed)
    agent_mod.POLICY = None
    # Margin, not raw coins: the opponent's score absorbs the seed's luck
    # (a rich board lifts both farms), which cuts the variance a lot.
    return log, r['scores'][0] - r['scores'][1], r['scores'][0]


def pad(batch, nfeat):
    B = len(batch)
    f = torch.zeros(B, MAXC, nfeat)
    m = torch.zeros(B, MAXC)
    b = torch.full((B, MAXC), -1e9)
    ch = torch.zeros(B, dtype=torch.long)
    for i, (phis, base, pick) in enumerate(batch):
        n = min(len(phis), MAXC)
        for j in range(n):
            f[i, j] = torch.tensor(phis[j]); m[i, j] = 1.0; b[i, j] = base[j]
        ch[i] = min(pick, n - 1)
    return f, m, b, ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--episodes', type=int, default=24)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--sample', type=int, default=3000, help='decisions per update')
    ap.add_argument('--device', default='auto',
                    help="auto | cpu | mps | cuda -- 'auto' picks cuda, then mps, then cpu")
    ap.add_argument('--epochs', type=int, default=8,
                    help='PPO-style passes over each batch; >1 reuses hard-won samples')
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--d-model', type=int, default=32)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--seed0', type=int, default=100000)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--out', default='results/rl_policy.pt')
    args = ap.parse_args()

    if args.device == 'auto':
        dev = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        dev = args.device
    print(f'device: {dev}   net: d_model={args.d_model} layers={args.layers}', flush=True)
    net = ResidualPolicy(d_model=args.d_model, n_layer=args.layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.resume:
        ck = torch.load(ROOT / args.resume, map_location='cpu', weights_only=False)
        net.load_state_dict(ck['state'])
        print(f"resumed from {args.resume} (best margin {ck.get('best_margin',0):+,.0f})")
    best = (-1e18, {k: v.cpu().clone() for k, v in net.state_dict().items()})
    hist = []

    for it in range(args.iters):
        cpu_sd = {k: v.cpu() for k, v in net.state_dict().items()}
        jobs = [(cpu_sd, args.seed0 + it * args.episodes + i,
                 POOL[i % len(POOL)], args.scale, args.temp,
                 args.d_model, args.layers)
                for i in range(args.episodes)]
        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            outs = list(ex.map(rollout, jobs, chunksize=1))

        margins = [o[1] for o in outs]
        scores = [o[2] for o in outs]
        mu, sd = statistics.mean(margins), (statistics.pstdev(margins) or 1.0)
        if mu > best[0]:
            best = (mu, {k: v.cpu().clone() for k, v in net.state_dict().items()})

        # Every decision in an episode carries that episode's whole-season
        # return, standardised across the batch.
        data, adv = [], []
        for log, margin, _ in outs:
            a = (margin - mu) / sd
            step = max(1, len(log) // max(1, args.sample // len(outs)))
            for d in log[::step]:
                data.append(d); adv.append(a)
        if not data:
            continue
        nfeat = len(data[0][0][0])
        f, m, b, ch = pad(data, nfeat)
        A = torch.tensor(adv, dtype=torch.float32)

        net.train()
        f, m, b, ch, A = f.to(dev), m.to(dev), b.to(dev), ch.to(dev), A.to(dev)
        with torch.no_grad():
            old_logits, _ = net(f, m, b, args.scale)
            old_logp = F.log_softmax(old_logits, 1).gather(1, ch[:, None]).squeeze(1)
        # Episodes are expensive here (0.3/s), so each batch is reused for
        # several clipped updates rather than one REINFORCE step. The clip is
        # what makes reuse safe: it stops a batch being optimised into a policy
        # far from the one that generated it.
        for _ in range(args.epochs):
            logits, value = net(f, m, b, args.scale)
            logp = F.log_softmax(logits, 1).gather(1, ch[:, None]).squeeze(1)
            adv_t = A - value.detach()
            ratio = torch.exp(logp - old_logp)
            pg = -torch.min(ratio * adv_t,
                            torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t).mean()
            vloss = F.mse_loss(value, A)
            ent = -(F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(1).mean()
            loss = pg + 0.5 * vloss - 0.01 * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        hist.append({'iter': it, 'margin': round(mu), 'score': round(statistics.mean(scores))})
        print(f'iter {it:3d}  margin {mu:+9,.0f}  score {statistics.mean(scores):9,.0f}  '
              f'decisions {len(data):5d}  loss {loss.item():+.3f}  ({time.perf_counter()-t:.0f}s)',
              flush=True)
        torch.save({'state': best[1], 'best_margin': best[0], 'scale': args.scale,
                    'history': hist}, ROOT / args.out)

    print(f'\nbest mean margin {best[0]:+,.0f} -- validate paired before shipping')


if __name__ == '__main__':
    main()
