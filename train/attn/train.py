"""Fit the set-transformer to the profit that followed each decision.

The raw label -- discounted future money -- is dominated by *when* the decision
happened: everything early is followed by a whole season of income and
everything late is not. Regressing on it directly teaches the clock, not the
farm. So each label is turned into an advantage by subtracting the mean future
money of decisions made at the same point in the season, which leaves only
"did this choice do better than average for its moment".

    python train/attn/train.py --data results/attn_data.npz --epochs 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TaskSetTransformer, PER_UNIT_COLS


def load(path, bins=48):
    z = np.load(ROOT / path)
    feats, masks = z['feats'].copy(), z['masks']
    chosen, targets, steps = z['chosen'], z['targets'], z['steps']
    feats[:, :, PER_UNIT_COLS] = 0.0

    # Advantage against a baseline that depends only on when in the season the
    # decision was made.
    edges = np.linspace(0, steps.max() + 1, bins + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, bins - 1)
    adv = np.zeros_like(targets)
    for b in range(bins):
        sel = idx == b
        if sel.sum() > 1:
            adv[sel] = targets[sel] - targets[sel].mean()
    sd = adv.std() or 1.0
    return feats, masks, chosen, (adv / sd).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='results/attn_data.npz')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d-model', type=int, default=48)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--out', default='results/attn.pt')
    args = ap.parse_args()

    torch.manual_seed(0)
    feats, masks, chosen, adv = load(args.data)
    n = len(feats)
    cut = int(n * 0.85)
    perm = np.random.RandomState(0).permutation(n)
    tr, va = perm[:cut], perm[cut:]
    print(f'{n:,} decisions  train {len(tr):,}  val {len(va):,}')

    dev = 'cpu'
    model = TaskSetTransformer(feats.shape[2], args.d_model, 4, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    F = torch.from_numpy(feats); M = torch.from_numpy(masks)
    C = torch.from_numpy(chosen); A = torch.from_numpy(adv)

    def evaluate(ix):
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(ix), 1024):
                b = ix[i:i + 1024]
                p = model(F[b], M[b]).gather(1, C[b, None]).squeeze(1)
                tot += ((p - A[b]) ** 2).sum().item()
            return tot / len(ix)

    # Baseline: predicting 0 for everything, which is what the shipped agent
    # effectively does. Any model that cannot beat this is worthless.
    base = float((A[va] ** 2).mean())
    best = (1e9, None)
    for ep in range(args.epochs):
        model.train()
        order = np.random.RandomState(ep).permutation(tr)
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            p = model(F[b], M[b]).gather(1, C[b, None]).squeeze(1)
            loss = lossf(p, A[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(va)
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()})
        print(f'  epoch {ep:3d}  val MSE {v:.4f}   (predict-zero {base:.4f})', flush=True)

    print(f'\nbest val MSE {best[0]:.4f} vs predict-zero {base:.4f}  '
          f'-> {100*(1-best[0]/base):+.1f}% variance explained')
    if best[0] >= base:
        print('MODEL IS NO BETTER THAN PREDICTING ZERO -- do not ship it')
    torch.save({'state': best[1], 'd_model': args.d_model, 'layers': args.layers,
                'n_feat': feats.shape[2], 'val_mse': best[0], 'base_mse': base},
               ROOT / args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
