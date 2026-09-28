"""Summarise real Kaggle episodes: what we did, what the winner did.

Local opponents all descend from this agent, so local measurement cannot see
what the field actually does. These are the real games.
"""
import argparse
import json
from collections import Counter


def farm_summary(step, p):
    farm = step[0]['observation']['farms'][p]
    crops = Counter(); animals = pens = 0
    for row in farm['tiles']:
        for t in row:
            if isinstance(t, dict):
                if t.get('kind') == 'PLANT':
                    crops[t['crop']] += 1
                elif 'animal' in t:
                    animals += 1
                elif t.get('kind') in ('PASTURE', 'COOP'):
                    pens += 1
    return crops, animals, pens, farm['money']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()

    for path in args.files:
        d = json.load(open(path))
        st = d['steps']
        teams = d['info'].get('TeamNames', ['?', '?'])
        rew = d['rewards']
        me = 0 if 'Amritesh' in str(teams[0]) else 1
        opp = 1 - me
        print(f"\n=== {path.split('/')[-1]}  {teams} ===")
        print(f"  us {rew[me]:,.0f}   them {rew[opp]:,.0f}   "
              f"{'WIN' if rew[me] > rew[opp] else 'LOSS'}")
        # peak herd and whether it survived
        peak_us = peak_op = 0
        end_us = end_op = 0
        hires_us = hires_op = 0
        for i in range(0, len(st), 24):
            _, a_us, _, _ = farm_summary(st[i], me)
            _, a_op, _, _ = farm_summary(st[i], opp)
            peak_us = max(peak_us, a_us); peak_op = max(peak_op, a_op)
        _, end_us, _, _ = farm_summary(st[-1], me)
        _, end_op, _, _ = farm_summary(st[-1], opp)
        for i in range(len(st)):
            h = st[i][me]['observation']['farms'][me].get('hires_today', 0) if False else 0
        crops_us, _, pens_us, _ = farm_summary(st[len(st)//2], me)
        crops_op, _, pens_op, _ = farm_summary(st[len(st)//2], opp)
        print(f"  herd  us peak {peak_us:2d} end {end_us:2d}   |  them peak {peak_op:2d} end {end_op:2d}")
        print(f"  crops@mid us {dict(crops_us)}")
        print(f"  crops@mid them {dict(crops_op)}")
        pr = st[-1][0]['observation']['market']['prices']
        print(f"  end prices MILK {pr.get('MILK',0):,.0f} MELON {pr.get('MELON',0):,.0f} "
              f"WOOL {pr.get('WOOL',0):,.0f} STRAWBERRY {pr.get('STRAWBERRY',0):,.0f}")


if __name__ == '__main__':
    main()
