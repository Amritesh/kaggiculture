"""Lean episode driver for Kaggriculture training.

Runs the official engine's own transition functions over plain dicts, skipping
the kaggle_environments Struct/observation-copy layer. Semantics are exact by
construction: nothing here reimplements game rules, it only reproduces the
`interpreter` step order from the env module.

Used for CMA-ES self-play and league evaluation, where the wrapper overhead
dominates the actual game logic.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CHECKOUT = ROOT / 'vendor/kaggle-environments'
if _CHECKOUT.exists() and str(_CHECKOUT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT))

from kaggle_environments.envs.kaggriculture import kaggriculture as K

DEFAULTS = {
    'episodeSteps': 720, 'boardSize': 10, 'startingMoney': 3000,
    'maxMarketOrdersPerTurn': 10, 'turnsPerDay': 24, 'shedCapacity': 100,
    'weedSpawnChance': 0.005, 'townShopUnlockInterval': 3,
    'townShopSellInterval': 4, 'townCenterSellInterval': 24,
    'farmHandCostMult': 1, 'marketParams': {},
}


class Obs(dict):
    """Dict with attribute access, matching what the engine functions expect."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class _AgentState:
    __slots__ = ('observation', 'action', 'status', 'reward')

    def __init__(self, observation):
        self.observation = observation
        self.action = None
        self.status = 'ACTIVE'
        self.reward = 0.0


class _Env:
    __slots__ = ('configuration', 'info', 'done')

    def __init__(self, configuration, seed):
        self.configuration = configuration
        self.info = {'seed': seed}
        self.done = False


def _fresh_state(config, seed):
    board = int(config['boardSize'])
    money = config['startingMoney']
    farms = [K._new_farm(board, money) for _ in range(2)]
    privates = [K._new_private() for _ in range(2)]
    overrides = config.get('marketParams') or None
    params = K._resolve_market_params(overrides) if overrides else None
    market = K._new_market(params)
    town = K._new_town()

    state = []
    for i in range(2):
        obs = Obs(player=i, private=privates[i], farms=farms, market=market,
                  town=town, day=0, hour=0, step=0, remainingOverageTime=60)
        state.append(_AgentState(obs))
    return state, _Env(config, seed)


def _step(state, env, step):
    """One interpreter tick. Mirrors kaggriculture.interpreter's ordering."""
    cfg = env.configuration
    tpd = int(cfg['turnsPerDay'])
    board = int(cfg['boardSize'])
    shed_cap = int(cfg['shedCapacity'])
    day = step // tpd
    obs0 = state[0].observation

    for i, s in enumerate(state):
        action = s.action if isinstance(s.action, dict) else {}
        farmer_action = action.get('farmer', ['PASS'])
        hands_actions = action.get('hands', [])
        if not isinstance(hands_actions, list):
            hands_actions = []

        # Atomic PLANT validation: over-requesting a crop drops every PLANT for it.
        plant_demand = {}
        for a in [farmer_action, *hands_actions]:
            if isinstance(a, list) and len(a) >= 2 and a[0] == 'PLANT':
                plant_demand[a[1]] = plant_demand.get(a[1], 0) + 1
        seeds = s.observation.private.get('seeds', {})
        blocked = {c for c, n in plant_demand.items() if n > seeds.get(c, 0)}

        def allowed(a):
            if isinstance(a, list) and len(a) >= 2 and a[0] == 'PLANT' and a[1] in blocked:
                return ['PASS']
            return a

        K._apply_unit_action(obs0.farms[i], s.observation.private, 0,
                             allowed(farmer_action), board, day, tpd, shed_cap)
        for h, ha in enumerate(hands_actions):
            K._apply_unit_action(obs0.farms[i], s.observation.private, h + 1,
                                 allowed(ha), board, day, tpd, shed_cap)

    K._process_market(state, env)
    K._town_consume(env, state, step)
    for farm in obs0.farms:
        K._decay_plants(farm, step)
    if (step + 1) % tpd == 0:
        K._end_of_day(state, env, day)

    nxt = step + 1
    for s in state:
        s.observation.day = nxt // tpd
        s.observation.hour = nxt % tpd
        s.observation.step = nxt


def run_episode(agent0, agent1, seed=0, config=None, hooks=None):
    """Play one full episode. Returns {'scores', 'win', 'steps'}.

    agent0/agent1 are callables (obs, configuration) -> action dict, matching
    the submission signature. `hooks` is an optional list of callables invoked
    as hook(step, state) after each tick, for instrumentation.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)
    state, env = _fresh_state(cfg, seed)
    agents = (agent0, agent1)
    total = int(cfg['episodeSteps']) - 1

    for step in range(total):
        for i, s in enumerate(state):
            try:
                s.action = agents[i](s.observation, cfg)
            except Exception:
                s.action = {'farmer': ['PASS'], 'hands': [], 'market': []}
        _step(state, env, step)
        if hooks:
            for h in hooks:
                h(step, state)

    farms = state[0].observation.farms
    scores = [float(f['money']) for f in farms]
    return {'scores': scores, 'win': scores[0] > scores[1], 'seed': seed,
            'steps': total}
