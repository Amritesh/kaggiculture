"""Validate freshly trained weights on unseen seeds, then ship them.

`search.py` scores candidates against the previous version, so its own numbers
come from the seeds it selected against and overstate. Nothing ships on that.
This plays the candidate on a different seed block and requires p < 0.05
against **both**:

* the currently shipped agent, and
* a non-clone control.

The control is a gate, not a readout. Training optimises "beat the previous
version", so a candidate can learn to beat its own predecessor while collapsing
against everything else -- one did exactly that (80-0 against the shipped agent,
27-33 against crops_v5, having been 80-0 before) and had to be rolled back.

    python train/promote.py                  # report only
    python train/promote.py --ship           # validate, then bake in and repackage
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / 'agent' / 'main.py'
HISTORY = ROOT / 'results' / 'weight_history.json'


def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)


def write_weights(src, W):
    body = ', '.join(f'{v:+.6f}' for v in W)
    out, n = re.subn(r'W = \[[^\]]*\]', f'W = [{body}]', src, count=1)
    if n != 1:
        raise SystemExit('could not find the W vector in the agent')
    return out


def record(w, l, m, label):
    p = sign_test(w, l)
    print(f'  vs {label:24s} {w:3d}-{l:<3d} median {statistics.median(m):+8.0f} p={p:.2e}')
    return w > l and p < 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='results/policy.json')
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--control', default='opponents/crops_v5.py')
    ap.add_argument('--ship', action='store_true')
    args = ap.parse_args()

    W = json.loads((ROOT / args.weights).read_text())['W']
    src = AGENT.read_text()
    cand = ROOT / 'opponents' / '_candidate.py'
    cand.write_text(write_weights(src, W))

    out = ROOT / 'results' / 'validate.json'
    cmd = [sys.executable, str(ROOT / 'eval' / 'league.py'), '--agents',
           'opponents/_candidate.py', 'agent/main.py', args.control,
           '--seeds', str(args.seeds), '--output', 'results/validate.json']
    if args.workers:
        cmd += ['--workers', str(args.workers)]
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

    games = json.loads(out.read_text())['games']
    ok = True
    for opp, label in (('agent/main.py', 'shipped agent'), (args.control, 'non-clone control')):
        g = [x for x in games if set((x['a'], x['b'])) == {'opponents/_candidate.py', opp}]
        m = [(x['a_score'] - x['b_score']) if x['a'] == 'opponents/_candidate.py'
             else (x['b_score'] - x['a_score']) for x in g]
        w = sum(1 for v in m if v > 0)
        l = sum(1 for v in m if v < 0)
        ok = record(w, l, m, label) and ok
    cand.unlink()

    if not ok:
        print('\nNO PROMOTION - shipped weights stand')
        return
    if not args.ship:
        print('\nwould promote (re-run with --ship)')
        return

    # Keep the outgoing weights: the agent is edited in place, and a promotion
    # was rolled back once with no way to recover the previous vector.
    hist = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
    prev = re.search(r'W = \[([^\]]*)\]', src)
    if prev:
        hist.append([float(x) for x in prev.group(1).replace('\n', ' ').split(',')])
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        HISTORY.write_text(json.dumps(hist))

    AGENT.write_text(write_weights(src, W))
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(ROOT / 'tools' / 'package.py')],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    print(f'PROMOTED; tests pass; submission.tar.gz rebuilt '
          f'(previous weights kept, {len(hist)} in history)')


if __name__ == '__main__':
    main()
