"""Batched Kaggriculture on tensors: B farms stepped at once, on GPU.

Why this exists: the official engine is pure Python and runs one game at a
time, about 0.8 s an episode. Policy-gradient RL wants 10^6-10^8 episodes, and
on 8 cores that is roughly a month. A GPU cannot fix it, because the work is
dict manipulation, not arithmetic -- the accelerator sits idle waiting for the
interpreter. The only way to make a GPU useful is to express the game itself as
tensor operations, which is what this does: every farm in the batch advances
with the same handful of vectorised ops, so 4,096 games cost about what one
game costs in Python.

It is a TRAINING model, not a replacement for the engine. Faithful on the parts
the policy learns from -- growth, the watering window, the two-dry-day death,
feeding, the care bonus, daily fertilizer, the exact price curve and town
consumption -- and deliberately simplified elsewhere (no weeds, no land
purchase, fixed crew size, opponent supply modelled as a constant drain).
A policy trained here MUST be validated in the real engine before it ships;
sim/fastsim.py is bit-exact and exists for exactly that.

Layout: all state is (B, ...) tensors on one device. Tiles are flattened to
B x 100. Crop and animal types are integer codes, -1 meaning empty.
"""
import torch

# --- static tables, mirroring the engine -----------------------------------
CROP_NAMES = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON']
#            seed  first  maxyield  interval  maxheld  ongoing
CROP = torch.tensor([
    [10, 2, 4, 0, 6, 0],
    [20, 2, 3, 0, 4, 0],
    [50, 8, 8, 1, 4, 1],
    [100, 10, 10, 2, 4, 1],
    [80, 10, 12, 0, 6, 0],
], dtype=torch.float32)

ANIMAL_NAMES = ['GOOSE', 'COW', 'SHEEP']
#              cost  first  interval  maxheld  product_idx
ANIMAL = torch.tensor([
    [300, 4, 1, 4, 5],
    [400, 8, 2, 6, 6],
    [500, 6, 3, 6, 7],
], dtype=torch.float32)

PROD_NAMES = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK',
              'WOOL', 'FERTILIZER']
BASE = torch.tensor([25, 35, 60, 120, 250, 50, 160, 200, 100], dtype=torch.float32)
T_CAP = torch.tensor([400, 450, 200, 100, 300, 332, 122, 105, 200], dtype=torch.float32)
# 0 linear, 1 sq, 2 sqrt, 3 log, 4 hinge
ABOVE_FN = torch.tensor([3, 2, 2, 0, 1, 3, 0, 1, 0], dtype=torch.long)
ABOVE_TG = torch.tensor([0.20, 0.70, 0.60, 1.60, 3.60, 0.20, 1.60, 3.20, 0.40],
                        dtype=torch.float32)
BELOW_FN = torch.tensor([2, 4, 4, 2, 3, 4, 2, 3, 0], dtype=torch.long)
BELOW_TG = torch.tensor([0.80, 1.00, 0.40, 0.70, 0.20, 0.40, 0.60, 0.20, 0.40],
                        dtype=torch.float32)
I0 = 10000.0
HINGE_GAIN = 8.0

N = 10          # board is N x N
TPD = 24        # turns per day
DAYS = 30


def _shape(fn, x, T):
    """The engine's five price curves, evaluated elementwise. f(T) == 1."""
    r = x / T
    lin = r
    sq = r * r
    sqrt = torch.sqrt(torch.clamp(r, min=0.0))
    log = torch.log1p(torch.clamp(x, min=0.0)) / torch.log1p(T)
    # u + gain*max(0,u-1)^2 -- the linear term is NOT clamped; clamping it
    # loses the whole linear component above T and undervalues scarce goods.
    hinge = r + HINGE_GAIN * torch.clamp(r - 1.0, min=0.0) ** 2
    out = torch.where(fn == 0, lin, torch.where(fn == 1, sq, torch.where(
        fn == 2, sqrt, torch.where(fn == 3, log, hinge))))
    return out


def price(inv):
    """inv (B, P) market inventory -> price (B, P), matching the engine."""
    dev = inv.device
    base = BASE.to(dev); T = T_CAP.to(dev)
    d = inv - I0
    x = d.abs()
    above = _shape(ABOVE_FN.to(dev), x, T) * ABOVE_TG.to(dev) * base
    below = _shape(BELOW_FN.to(dev), x, T) * BELOW_TG.to(dev) * base
    p = torch.where(d > 0, base - above, base + below)
    # The engine rounds to whole coins before flooring.
    return torch.clamp(torch.round(p), min=1.0)


class VecFarms:
    """B independent farms, one player each, against a scripted market drain."""

    # tile kinds
    EMPTY, PLANT, PASTURE, COOP, ANIMAL_T = 0, 1, 2, 3, 4

    def __init__(self, B, n_units=8, device='cpu', seed=0):
        self.B, self.U, self.dev = B, n_units, device
        g = torch.Generator(device='cpu').manual_seed(seed)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.kind = z(B, N * N, dt=torch.long)
        self.crop = z(B, N * N, dt=torch.long) - 1
        self.planted = z(B, N * N)
        self.yield_u = z(B, N * N)
        self.watered = z(B, N * N)
        self.dry = z(B, N * N)
        self.animal = z(B, N * N, dt=torch.long) - 1
        self.placed = z(B, N * N)
        self.fed = z(B, N * N)
        self.cared = z(B, N * N)
        self.care_bonus = z(B, N * N)
        self.fert = z(B, N * N)
        self.pos = torch.full((B, self.U), (N // 2) * N + N // 2, dtype=torch.long,
                              device=device)
        self.carry = z(B, self.U, len(PROD_NAMES))
        self.seeds = z(B, len(CROP_NAMES))
        self.shed = z(B, len(PROD_NAMES))
        self.money = z(B) + 3000.0
        self.market = z(B, len(PROD_NAMES)) + I0
        self.step_i = 0
        # Seed the board: a few pens with animals near the shed, so the
        # livestock loop (feed, care, collect, harvest) is present from turn one
        # rather than being unreachable until the policy discovers buying.
        c = N // 2
        k = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                idx = (c + dy) * N + (c + dx)
                self.kind[:, idx] = self.ANIMAL_T
                self.animal[:, idx] = 1 if k % 3 else 2      # cows, some sheep
                self.placed[:, idx] = 0.0
                self.fert[:, idx] = 1.0
                k += 1
        self.seeds = z(B, len(CROP_NAMES)) + 4.0

    # -- helpers ------------------------------------------------------------
    @property
    def day(self):
        return self.step_i // TPD

    def _gather(self, t):
        return torch.gather(t, 1, self.pos)

    def _scatter(self, t, val, mask):
        """Write val into t at each unit's tile where mask is true."""
        idx = self.pos
        src = torch.where(mask, val, torch.gather(t, 1, idx))
        return t.scatter(1, idx, src)

    # -- one turn -----------------------------------------------------------
    def step(self, act):
        """act (B, U) integer actions. Returns money delta for the turn.

        0-3 move N/S/E/W   4 WATER   5 HARVEST   6 FEED   7 CARE
        8 COLLECT_FERT     9-13 PLANT crop 0-4     14 PASS
        """
        B, U, dev = self.B, self.U, self.dev
        money0 = self.money.clone()

        # movement
        row = self.pos // N
        col = self.pos % N
        row = torch.where(act == 0, torch.clamp(row - 1, min=0), row)
        row = torch.where(act == 1, torch.clamp(row + 1, max=N - 1), row)
        col = torch.where(act == 2, torch.clamp(col - 1, min=0), col)
        col = torch.where(act == 3, torch.clamp(col + 1, max=N - 1), col)
        self.pos = row * N + col

        kind = self._gather(self.kind)
        crop = self._gather(self.crop)
        animal = self._gather(self.animal)

        # WATER: marks watered; yield is added at end of day inside the window
        wmask = (act == 4) & (kind == self.PLANT)
        self.watered = self._scatter(self.watered, torch.ones_like(self.watered[:, :1]).expand_as(wmask).float(), wmask)

        # HARVEST: crop or animal stock into the unit's hands
        hmask = (act == 5) & ((kind == self.PLANT) | (animal >= 0))
        yu = self._gather(self.yield_u)
        got = torch.where(hmask, yu, torch.zeros_like(yu))
        pidx = torch.where(animal >= 0,
                           ANIMAL.to(dev)[torch.clamp(animal, min=0), 4].long(),
                           torch.clamp(crop, min=0))
        self.carry.scatter_add_(2, pidx.unsqueeze(-1), got.unsqueeze(-1))
        self.yield_u = self._scatter(self.yield_u, torch.zeros_like(yu), hmask)

        # FEED: costs one wheat from the unit's hands
        wp = price(self.market)[:, 0]
        fmask = ((act == 6) & (animal >= 0) & (self._gather(self.fed) < 1)
                 & (self.money > wp * self.U).unsqueeze(1))
        self.money = self.money - fmask.float().sum(1) * wp
        self.fed = self._scatter(self.fed, torch.ones_like(yu), fmask)

        cmask = (act == 7) & (animal >= 0) & (self._gather(self.cared) < 1)
        self.cared = self._scatter(self.cared, torch.ones_like(yu), cmask)

        # COLLECT_FERTILIZER: one unit per animal per day, free
        gmask = (act == 8) & (animal >= 0) & (self._gather(self.fert) > 0)
        self.carry[:, :, 8] = self.carry[:, :, 8] + gmask.float()
        self.fert = self._scatter(self.fert, torch.zeros_like(yu), gmask)

        # PLANT on an empty tile, paying the seed from cash.
        for ci in range(len(CROP_NAMES)):
            cost = float(CROP[ci, 0])
            pmask = (act == 9 + ci) & (kind == self.EMPTY) & (self.money > cost).unsqueeze(1)
            # only the first planting worker per farm per turn pays, to keep
            # cash accounting exact under simultaneous actions
            first = pmask & (pmask.cumsum(1) == 1)
            self.money = self.money - first.float().sum(1) * cost
            self.kind = self._scatter(self.kind.float(), torch.full_like(kind, float(self.PLANT), dtype=torch.float32), first).long()
            self.crop = self._scatter(self.crop.float(), torch.full_like(kind, float(ci), dtype=torch.float32), first).long()
            self.planted = self._scatter(self.planted, torch.full_like(self.planted[:, :1].expand_as(first), float(self.day)), first)
            self.dry = self._scatter(self.dry, torch.ones_like(self.planted[:, :1].expand_as(first)), first)

        # FEED buys its wheat at the market price rather than modelling the
        # shed-and-carry logistics, which the planner handles and the policy
        # does not need to relearn.
        self.step_i += 1
        if self.step_i % TPD == 0:
            self._end_of_day()
        return self.money - money0

    def _end_of_day(self):
        dev = self.dev
        day = float(self.day)
        C = CROP.to(dev)
        is_plant = self.kind == self.PLANT
        ci = torch.clamp(self.crop, min=0)
        age = day - self.planted
        first, maxy, cap = C[ci, 1], C[ci, 2], C[ci, 4]

        # watering adds yield only inside the window, as in the engine
        lo = torch.where((ci == 0) | (ci == 1), torch.full_like(first, 2.0),
                         torch.ceil((first + 1) / 2))
        in_win = (age >= lo) & (age <= maxy) & is_plant
        self.yield_u = torch.where(in_win & (self.watered > 0),
                                   torch.minimum(self.yield_u + 1, cap), self.yield_u)
        # two dry days kills the plant
        self.dry = torch.where(self.watered > 0, torch.zeros_like(self.dry),
                               self.dry + 1)
        dead = is_plant & (self.dry >= 2)
        # non-ongoing crops also expire past max_yield_day + 1
        expired = is_plant & (C[ci, 5] < 0.5) & (age > maxy + 1)
        clear = dead | expired
        self.kind = torch.where(clear, torch.zeros_like(self.kind), self.kind)
        self.crop = torch.where(clear, torch.full_like(self.crop, -1), self.crop)
        self.yield_u = torch.where(clear, torch.zeros_like(self.yield_u), self.yield_u)
        self.watered = torch.zeros_like(self.watered)

        # animals: production on the interval, care doubles a FED tick
        A = ANIMAL.to(dev)
        has_a = self.animal >= 0
        ai = torch.clamp(self.animal, min=0)
        aage = day - self.placed
        due = (aage >= A[ai, 1]) & (torch.remainder(aage - A[ai, 1], A[ai, 2]) == 0)
        gain = torch.where(self.fed > 0, 1.0 + self.care_bonus, torch.ones_like(self.fed))
        self.yield_u = torch.where(has_a & due,
                                   torch.minimum(self.yield_u + gain, A[ai, 3]),
                                   self.yield_u)
        self.care_bonus = torch.where(has_a & due, torch.zeros_like(self.care_bonus),
                                      self.care_bonus)
        self.care_bonus = torch.where(has_a & (self.cared > 0) & (self.fed > 0),
                                      self.care_bonus + 1, self.care_bonus)
        # unfed twice and the animal is gone
        self.dry = torch.where(has_a & (self.fed > 0), torch.zeros_like(self.dry),
                               torch.where(has_a, self.dry + 1, self.dry))
        lost = has_a & (self.dry >= 2)
        self.animal = torch.where(lost, torch.full_like(self.animal, -1), self.animal)
        self.kind = torch.where(lost, torch.full_like(self.kind, self.PASTURE), self.kind)
        self.fert = torch.where(has_a & ~lost, torch.ones_like(self.fert), self.fert)
        self.fed = torch.zeros_like(self.fed)
        self.cared = torch.zeros_like(self.cared)

        # carried goods are banked, units respawn at the shed
        self.shed = self.shed + self.carry.sum(1)
        self.carry = torch.zeros_like(self.carry)
        self.pos = torch.full_like(self.pos, (N // 2) * N + N // 2)

        # town consumes, which is what keeps scarce goods valuable
        self.market = torch.clamp(self.market - 40.0, min=1.0)

    def sell_all(self):
        """Clear the shed at the current marginal price, one unit at a time
        (approximated by the midpoint of the move, as the agent does)."""
        p0 = price(self.market)
        p1 = price(self.market + self.shed)
        revenue = ((p0 + p1) / 2 * self.shed).sum(1)
        self.money = self.money + revenue
        self.market = self.market + self.shed
        self.shed = torch.zeros_like(self.shed)
        return revenue
