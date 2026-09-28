"""Count what dies, and whether it died with work already invested in it.

"Lost" in tools/chains.py lumps two very different things together: a seed
scattered and never watered (a cheap bet that did not come in) and a plant
watered four times and then abandoned one day before harvest (labour thrown
away). Only the second is a defect. Animals are unambiguous -- an animal unfed
two days escapes, taking its purchase price and every future yield with it.
"""
import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
sys.path.insert(0, str(ROOT / 'agent'))


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--opponent', default='opponents/original.py')
    args = ap.parse_args()
    import fastsim
    agent_mod = _load('agent/main.py', 'loss_agent')
    opp = _load(args.opponent, 'loss_opp')

    totals = defaultdict(float)
    for seed in range(args.seeds):
        waters = defaultdict(int)
        prev = {}
        prev_animals = {}
        state_ref = {}

        def hook(step, state):
            farm = state[0].observation.farms[0]
            act = state[0].action or {}
            pos = [farm['farmer']] + list(farm.get('hands', []))
            for i, a in enumerate(list(act.get('hands', [])) + [act.get('farmer')]):
                if a and a[0] == 'WATER' and i < len(pos):
                    waters[tuple(pos[i])] += 1
            cur, animals = {}, {}
            for y, row in enumerate(farm['tiles']):
                for x, t in enumerate(row):
                    if isinstance(t, dict):
                        if t.get('kind') == 'PLANT':
                            cur[(x, y)] = (t['crop'], t.get('planted_day'),
                                           t.get('consecutive_unwatered', 0))
                        elif 'animal' in t:
                            animals[(x, y)] = (t['animal'], t.get('consecutive_unfed', 0))
            for k, v in prev.items():
                if k not in cur or cur[k][1] != v[1]:
                    # Died dry with watering already invested in it.
                    if v[2] >= 1 and waters.get(k, 0) > 0:
                        totals['plants_died_after_work'] += 1
                        totals['wasted_waters'] += waters.get(k, 0)
                    waters[k] = 0
            for k, v in prev_animals.items():
                if k not in animals:
                    totals['animals_lost'] += 1
            totals['at_risk_plant_turns'] += sum(1 for v in cur.values() if v[2] >= 1)
            totals['at_risk_animal_turns'] += sum(1 for v in animals.values() if v[1] >= 1)
            prev.clear(); prev.update(cur)
            prev_animals.clear(); prev_animals.update(animals)

        r = fastsim.run_episode(agent_mod.agent, opp.agent, seed=seed, hooks=[hook])
        totals['score'] += r['scores'][0]

    n = args.seeds
    print(f'over {n} episodes (per episode):')
    print(f"  score                          {totals['score']/n:10,.0f}")
    print(f"  animals lost                   {totals['animals_lost']/n:10.1f}")
    print(f"  plants died dry AFTER watering {totals['plants_died_after_work']/n:10.1f}")
    print(f"  waters wasted on them          {totals['wasted_waters']/n:10.1f}")
    print(f"  turns with a plant 1 day from death  {totals['at_risk_plant_turns']/n:8.0f}")
    print(f"  turns with an animal 1 day from death {totals['at_risk_animal_turns']/n:7.0f}")


if __name__ == '__main__':
    main()
