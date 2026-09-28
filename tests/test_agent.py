"""Invariants for the livestock agent (agent/main.py).

These guard rules the engine enforces silently -- over-requesting seeds drops
every PLANT for that crop, and market orders past the per-turn cap are discarded
without warning -- plus the livestock bugs that cost a whole season each when
they regressed during development.
"""
import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('v3', ROOT / 'agent' / 'main.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_obs(day=0, hour=0, money=3000, quadrants=('NW',)):
    tiles = [[None if x < 5 and y < 5 else 'LOCKED' for x in range(10)]
             for y in range(10)]
    farm = dict(money=money, tiles=tiles, farmer=[4, 4], hands=[],
                unlocked_quadrants=list(quadrants), hires_today=0)
    prices = {'WHEAT': 25, 'CARROT': 35, 'MELON': 250, 'TOMATO': 60,
              'STRAWBERRY': 120, 'EGG': 50, 'MILK': 160, 'WOOL': 200,
              'FERTILIZER': 100}
    return dict(player=0, day=day, hour=hour, step=day * 24 + hour,
                farms=[farm, deepcopy(farm)],
                private=dict(shed={}, seeds={}, inventories=[{}]),
                market=dict(inventory={}, prices=prices),
                town=dict(unlocked_shops=[]))


class MainV3Tests(unittest.TestCase):
    def test_never_plants_more_than_seeds_held(self):
        """Over-requesting a crop makes the engine drop ALL its PLANTs."""
        mod = load()
        mod.TUNE['herd'] = 0.0  # pens would otherwise outbid planting
        obs = make_obs(2, 4)
        obs['private']['seeds'] = {'MELON': 1}
        obs['farms'][0]['hands'] = [[3, 4], [4, 3]]
        obs['private']['inventories'] += [{}, {}]
        result = mod.agent(obs)
        plants = [a for a in [result['farmer']] + result['hands'] if a[0] == 'PLANT']
        for crop, held in obs['private']['seeds'].items():
            self.assertLessEqual(sum(a[1] == crop for a in plants), held)

    def test_market_orders_within_cap(self):
        """Orders past maxMarketOrdersPerTurn are silently discarded."""
        mod = load()
        obs = make_obs(6, 0)
        obs['private']['shed'] = {k: 30 for k in
                                  ('WHEAT', 'CARROT', 'MELON', 'TOMATO',
                                   'STRAWBERRY', 'EGG', 'MILK', 'WOOL')}
        orders = mod.agent(obs, {'maxMarketOrdersPerTurn': 10})['market']
        self.assertLessEqual(len(orders), 10)

    def test_observation_not_mutated(self):
        mod = load()
        obs = make_obs(3, 5)
        before = deepcopy(obs)
        mod.agent(obs)
        self.assertEqual(obs, before)

    def test_animal_is_placed_not_hoarded(self):
        """An animal left in the shed earns nothing, and end-of-day auto-drop
        returns any carried animal -- a unit holding one must place it."""
        mod = load()
        obs = make_obs(5, 3)
        obs['farms'][0]['tiles'][3][3] = dict(kind='PASTURE')
        obs['farms'][0]['farmer'] = [3, 3]
        obs['private']['inventories'] = [{'COW': 1}]
        action = mod.agent(obs)['farmer']
        self.assertEqual(action, ['PLACE', 'COW'])

    def test_hungry_animal_is_fed_before_routine_work(self):
        """Two days unfed and the animal escapes, forfeiting its purchase."""
        mod = load()
        obs = make_obs(12, 3)
        obs['farms'][0]['tiles'][3][3] = dict(
            kind='PASTURE', animal='COW', placed_day=2, yield_units=0,
            consecutive_unfed=1, fed_today=False, cared_today=False,
            fertilizer_available=False)
        obs['farms'][0]['farmer'] = [3, 3]
        obs['private']['inventories'] = [{'WHEAT': 2}]
        action = mod.agent(obs)['farmer']
        self.assertEqual(action, ['FEED'])

    def test_selling_is_metered_once_the_market_cannot_absorb_it(self):
        """Dumping the shed each turn walked MELON from 272 to 60 a season.

        MELON's glut term is quadratic with T=300, so the market swallows a
        modest parcel at near full price but collapses on a large one. Metering
        should only bite in the second case.
        """
        mod = load()
        obs = make_obs(8, 2)
        obs['market']['inventory'] = {'MELON': 10000}

        obs['private']['shed'] = {'MELON': 40}
        small = [o for o in mod.agent(obs)['market']
                 if o[0] == 'SELL' and o[1] == 'MELON']
        self.assertEqual(small[0][2], 40, 'a parcel the market absorbs should sell whole')

        obs['private']['shed'] = {'MELON': 400}
        big = [o for o in mod.agent(obs)['market']
               if o[0] == 'SELL' and o[1] == 'MELON']
        self.assertTrue(big, 'expected some MELON to be sold')
        self.assertLess(big[0][2], 400, 'whole stock dumped into a market that cannot take it')


if __name__ == '__main__':
    unittest.main()
