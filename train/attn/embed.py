"""Embed the trained network into agent/main.py as a base64 float32 blob.

The submission is one file, so the weights travel inside it. They are stored as
raw float32 rather than JSON numerals: 137 KB instead of 829 KB, and it loads
with one frombuffer call instead of parsing 35,000 numbers on the first turn.

    python train/attn/embed.py --weights results/attn_weights.json --scale 0.5
"""
import argparse
import base64
import json
import re
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / 'agent' / 'main.py'

ORDER = ['inp.weight', 'inp.bias']
for i in (0, 1):
    p = f'enc.layers.{i}'
    ORDER += [f'{p}.self_attn.in_proj_weight', f'{p}.self_attn.in_proj_bias',
              f'{p}.self_attn.out_proj.weight', f'{p}.self_attn.out_proj.bias',
              f'{p}.linear1.weight', f'{p}.linear1.bias',
              f'{p}.linear2.weight', f'{p}.linear2.bias',
              f'{p}.norm1.weight', f'{p}.norm1.bias',
              f'{p}.norm2.weight', f'{p}.norm2.bias']
ORDER += ['out.weight', 'out.bias']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='results/attn_weights.json')
    ap.add_argument('--scale', type=float, default=0.5)
    args = ap.parse_args()

    meta = json.loads((ROOT / args.weights).read_text())
    W = meta['w']
    buf = bytearray()
    shapes = []
    for k in ORDER:
        a = np.asarray(W[k], dtype=np.float32)
        shapes.append(list(a.shape))
        buf += a.tobytes()
    blob = base64.b64encode(bytes(buf)).decode()

    src = AGENT.read_text()
    src = re.sub(r'ATTN_SHAPES = \[.*?\]\n', f'ATTN_SHAPES = {shapes}\n', src, count=1, flags=re.S)
    src = re.sub(r'ATTN_B64 = "[^"]*"\n', f'ATTN_B64 = "{blob}"\n', src, count=1)
    src = re.sub(r'ATTN_SCALE = [-0-9.]+\n', f'ATTN_SCALE = {args.scale}\n', src, count=1)
    AGENT.write_text(src)
    print(f'embedded {sum(np.prod(s) for s in shapes):,.0f} params, '
          f'{len(blob)/1024:.0f} KB base64, scale {args.scale}')


if __name__ == '__main__':
    main()
