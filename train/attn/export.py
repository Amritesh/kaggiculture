"""Export the trained set-transformer into the agent as plain nested lists.

The submission is a single file and must not depend on torch. numpy ships with
kaggle-environments so the forward pass uses it, but the agent falls back to the
plain heuristic if the import or the maths fails for any reason: a crashed
agent scores zero, and this correction is worth a few percent at most.

    python train/attn/export.py --model results/attn.pt --scale 1.0
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='results/attn.pt')
    ap.add_argument('--out', default='results/attn_weights.json')
    args = ap.parse_args()

    ck = torch.load(ROOT / args.model, map_location='cpu', weights_only=False)
    sd = ck['state']
    w = {k: v.tolist() for k, v in sd.items()}
    meta = {'d_model': ck['d_model'], 'layers': ck['layers'], 'n_feat': ck['n_feat'],
            'val_mse': ck['val_mse'], 'base_mse': ck['base_mse'], 'w': w}
    dest = ROOT / args.out
    dest.write_text(json.dumps(meta))
    print(f"wrote {args.out}  ({dest.stat().st_size/1024:.0f} KB)")
    print(f"val MSE {ck['val_mse']:.4f} vs predict-zero {ck['base_mse']:.4f} "
          f"({100*(1-ck['val_mse']/ck['base_mse']):+.1f}% variance explained)")
    print('keys:', ', '.join(sorted(w)[:6]), '...')


if __name__ == '__main__':
    main()
