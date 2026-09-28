"""Benchmark opponent: maximum expansion and hiring.

Not a submission candidate. Wraps the shipped agent with a different TUNE
vector, so the league contains genuinely different strategies rather than
variations on one heuristic -- a pool of near-clones recommends wrong answers.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    '_base_landgrab', Path(__file__).resolve().parents[1] / 'agent' / 'main.py')
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.TUNE.update(dict(max_land=4.0,land_cash=200.0,workers_hi=12.0,herd=0.0))

agent = _base.agent
