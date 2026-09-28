"""Kaggriculture: coordinated, market-aware farm policy. Standard library only.

v2 of main.py. Changes are confined to market execution: the original sold the
entire shed every turn at whatever the spot price was, which walked MELON from
272 down to 60 over a season (its glut term is quadratic, and T is only 300).
Selling is now metered against the marginal price curve, while still clearing
stock before the season ends.
"""
import math

# seed cost, first harvest, peak harvest, peak units, last harvest
CROPS = {'WHEAT':(10,2,4,4,4), 'CARROT':(20,2,3,3,3),
         'MELON':(80,10,10,6,12), 'TOMATO':(50,8,11,4,11),
         'STRAWBERRY':(100,10,16,4,16)}
PARAMS = {
 'WHEAT':(25,400,'sqrt',.8,'log',.2), 'CARROT':(35,450,'hinge',1.,'sqrt',.7),
 'TOMATO':(60,200,'hinge',.4,'sqrt',.6), 'STRAWBERRY':(120,100,'sqrt',.7,'linear',1.6),
 'MELON':(250,300,'log',.2,'sq',3.6), 'EGG':(50,332,'hinge',.4,'log',.2),
 'MILK':(160,122,'sqrt',.6,'linear',1.6), 'WOOL':(200,105,'log',.2,'sq',3.2),
 'FERTILIZER':(100,200,'linear',.4,'linear',.4)}
# Livestock. GOOSE is omitted: ~27 coins per worker-action against MELON's ~118
# and COW's ~160. CARE banks a +1 bonus that the next fed production tick
# consumes, doubling output; an animal unfed two days escapes and is lost.
ANIMALS = {
    'COW':   {'cost':400,'pen':'PASTURE','first':8,'iv':2,'held':6,'prod':'MILK'},
    'SHEEP': {'cost':500,'pen':'PASTURE','first':6,'iv':3,'held':6,'prod':'WOOL'},
    # GOOSE was excluded on coins-per-worker-action, which ignored how often it
    # pays: it yields EVERY day from day 4, where a cow yields every second day
    # from day 8. That makes it the only livestock that returns cash during the
    # opening, when the farm sits on ~150 coins and cannot afford anything.
    'GOOSE': {'cost':300,'pen':'COOP','first':4,'iv':1,'held':4,'prod':'EGG'},
}
SHOPS = {'BAKERY':['EGG','WHEAT'], 'PIZZA_SHOP':['MILK','TOMATO','WHEAT'],
 'BRUNCH_SPOT':['EGG','WHEAT','STRAWBERRY'], 'YARN_STORE':['WOOL','WOOL'],
 'ICE_CREAM_SHOP':['STRAWBERRY','MILK','WHEAT'], 'PET_CAFE':['CARROT','CARROT'],
 'SMOOTHIE_SHOP':['STRAWBERRY','MILK'], 'FARMERS_MARKET':['WHEAT','CARROT','TOMATO','STRAWBERRY']}
# Tunable vector (see tune.py). Named TUNE, not PARAMS: PARAMS above is this
# agent's market price table and must not be shadowed.
TUNE = {
    'workers_hi': 10.61, 'workers_mid': 4.676, 'workers_lo': 4.707,
    'active_hi': 5.578, 'active_lo': 4.0,
    'max_land': 2.487, 'land_cash': 1405, 'land_last_day': 20.31, 'land_fill': 0.3983,
    'sell_floor': 0.6263, 'sell_spread': 2.297, 'sell_clear': 0.2318,
    'carry_cap': 12.52, 'travel_cost': 1.204, 'seed_cap': 4.261,
    'water_urgent': 251, 'water_base': 69.73, 'opp_weight': 1.0,
    'herd': 7.245, 'barn_r': 2.882, 'animal_last_day': 12.57, 'animal_cash': 2754,
    'feed_batch': 6.401, 'animal_harvest': 5.948, 'feed_days': 3.0, 'sheep_share': 0.1251,
    'labor_weight': 0.0,   # 0 = per-tile-day scoring, 1 = per-action (1.0 measured far worse)
    'feed_horizon': 19.73, 'feed_safety': 1.0,
    # Tier-3 task priorities. These were hand-set guesses; the learned linear
    # model had already found PLACE/PICKUP were ~10x too high, so they are
    # parameters now and searched like everything else.
    'v_pen': 500.0, 'v_plant': 2.0, 'v_aharv': 2.0, 'v_feed_u': 30.0,
    'v_feed': 14.0, 'v_care': 5.0, 'v_place': 40000.0, 'v_water_h': 250.0,
    'v_harv': 300.0, 'v_harv_m': 2.0, 'v_pick_w': 6000.0, 'v_pick_a': 20000.0,
    'goose_share': 0.0,   # measured worse: a daily feed+care+collect for a 50-coin egg
    'feed_buffer': 2.001, 'feed_max': 24.0,
    'optimal_assign': 1,   # 0 = greedy crew assignment, 1 = Hungarian
    'v_fert': 4.646,   # weight on collecting fertilizer from owned animals
    'plant_radius': 8.043,   # only sow within this distance of the shed
}
SPACE = [
    ('workers_hi', 4.0, 13.0), ('workers_mid', 3.0, 11.0), ('workers_lo', 2.0, 9.0),
    ('active_hi', 2.0, 40.0), ('active_lo', 1.0, 20.0),
    ('max_land', 2.0, 4.0), ('land_cash', 100.0, 4000.0),
    ('land_last_day', 6.0, 24.0), ('land_fill', 0.2, 1.2),
    ('sell_floor', 0.30, 1.10), ('sell_spread', 1.0, 10.0), ('sell_clear', 0.0, 4.0),
    ('carry_cap', 10.0, 95.0), ('travel_cost', 0.5, 5.0), ('seed_cap', 3.0, 20.0),
    ('water_urgent', 100.0, 900.0), ('water_base', 20.0, 400.0),
    ('opp_weight', 0.85, 1.15),
    ('herd', 0.0, 16.0), ('barn_r', 2.0, 5.0), ('animal_last_day', 4.0, 16.0),
    ('animal_cash', 200.0, 3000.0), ('feed_batch', 2.0, 14.0),
    ('animal_harvest', 1.0, 6.0), ('labor_weight', 0.0, 0.5),
    ('feed_days', 0.0, 8.0), ('sheep_share', 0.0, 0.6),
    ('plant_radius', 3.0, 10.0),
]
# Worker-actions one tile of each crop consumes over its life: plant, the waters
# that actually add yield, the survival waters between them, and harvest(s).
# Labour is the binding constraint -- land is 4000 a quadrant while the 13th
# hand costs ~7000 a season -- so crops are ranked per action, not per tile-day.
LABOR = {'WHEAT':6.0, 'CARROT':5.0, 'MELON':12.0, 'TOMATO':11.0, 'STRAWBERRY':14.0}
# ---------------------------------------------------------------- learned policy
# Actions are chosen autoregressively: units are assigned one at a time and each
# candidate is scored with the assignments already made this turn as input, so
# the policy can spread the crew out or pile onto urgent work.
#
# The score is the hand-built heuristic multiplied by exp(W . phi). W is all
# zeros by default, which reproduces the heuristic exactly -- training can only
# move away from a known-good baseline, never start from noise.
#
# phi, in order (both plant and animal work share one feature space, so the
# model trades a cow's feeding against a melon's watering on the same scale):
#   0 water plant      1 harvest plant    2 plant seed     3 feed animal
#   4 care animal      5 harvest animal   6 fetch/place    7 build pen
#   8 distance/10      9 urgent          10 day progress  11 hour progress
#  12 crowding        13 same-op already 14 yield/6       15 bias
NW = 26
# Learned by train_policy.py: 151-9 against the zero-weight heuristic
# in training, 54-6 (p=1e-10) on held-out seeds. The model prefers
# high-yield tiles and prompt logistics, spreads the crew out
# (negative crowding), and works harder later in the day.
W = [+0.289229, +0.242076, +0.662843, +0.545308, -0.366402, -0.148995, -2.291418, +0.664476, -0.418600, -0.171905, +0.808082, -0.007582, -0.611780, +0.103980, +0.695016, -0.502503, +0.546291, +0.243043, -0.261623, +0.078200, -0.648019, -0.224436, -0.017673, +0.156826, -0.218265, -0.037267]






def shape(kind, x, t):
    x=max(0,x)
    if kind=='sqrt': return math.sqrt(x)
    if kind=='sq': return x*x
    if kind=='log': return math.log1p(x)
    if kind=='log10': return math.log10(1+x)
    if kind=='hinge': return x/t+8*max(0,x/t-1)**2
    return x


def price(crop, inventory, overrides=None):
    base,t,lo,lt,hi,ht=PARAMS[crop]
    p=(overrides or {}).get(crop,{})
    base=p.get('base',base); t=p.get('T',t); anchor=p.get('I0',10000)
    below=inventory<anchor
    f=p.get('below_func' if below else 'above_func',lo if below else hi)
    target=p.get('below_target' if below else 'above_target',lt if below else ht)
    return max(1,round(base+(1 if below else -1)*base*target*shape(f,abs(inventory-anchor),t)/shape(f,t,t)))


def task_features(op, tile, day, hour, tpd, final_day):
    """The part of phi that depends only on the task, so it is computed once per
    task per turn instead of once per (unit, task) pair -- ~9x fewer calls."""
    return features(op, 0, tile, day, hour, tpd, final_day, 0, 0)




# Training hooks. Both inert at their defaults, so the shipped decision path is
# exactly the one that was measured.
POLICY=None    # callable(feats, base_logits) -> chosen index, for RL rollouts
RLLOG=None     # list to append (feats, base_logits, chosen) per decision

def hungarian(cost):
    """Minimum-cost perfect assignment of rows to columns (Jonker-Volgenant
    shortest augmenting path). cost is a rectangular list of lists with
    len(rows) <= len(cols); returns a list giving each row's column.

    The crew is assigned greedily today: worker 1 takes the best task, worker 2
    the best of what is left, and so on. That is a 1/2-approximation -- an early
    worker can take a job a later worker was far better placed for, and nothing
    can undo it. With at most ~10 workers and a few dozen candidate tasks the
    exact solve costs microseconds, so there is no reason to approximate.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    INF = float('inf')
    u = [0.0]*(n+1)
    v = [0.0]*(m+1)
    p = [0]*(m+1)
    way = [0]*(m+1)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        minv = [INF]*(m+1)
        used = [False]*(m+1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m+1):
                if not used[j]:
                    cur = cost[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            if delta == INF:
                break
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [-1]*n
    for j in range(1, m+1):
        if 1 <= p[j] <= n:
            out[p[j]-1] = j-1
    return out


def features(op, dist, tile, day, hour, tpd, final_day, crowd, same_op):
    """phi for one candidate action, given what this turn has already assigned."""
    p = [0.0]*NW
    is_animal = isinstance(tile, dict) and 'animal' in tile
    if op == 'WATER': p[0] = 1.0
    elif op == 'HARVEST': p[5 if is_animal else 1] = 1.0
    elif op == 'PLANT': p[2] = 1.0
    elif op == 'FEED': p[3] = 1.0
    elif op == 'CARE': p[4] = 1.0
    elif op in ('PLACE', 'PICKUP'): p[6] = 1.0
    elif op == 'BUILD_PASTURE': p[7] = 1.0
    p[8] = dist / 10.0
    if isinstance(tile, dict):
        if tile.get('consecutive_unwatered', 0) >= 1 or tile.get('consecutive_unfed', 0) >= 1:
            p[9] = 1.0
        p[14] = min(1.0, tile.get('yield_units', 0) / 6.0)
    p[10] = min(1.0, day / max(1.0, float(final_day)))
    p[11] = hour / float(tpd)
    p[12] = min(1.0, crowd / 4.0)
    p[13] = min(1.0, same_op / 4.0)
    p[15] = 1.0
    # Rule-derived facts about the tile's situation. These are consequences of
    # the engine's rules, like "is this piece attacked" -- they describe the
    # position, they do not rank it. The correction network decides what they
    # are worth. Appended after index 15 so the weights learned over the
    # original sixteen features keep their meaning.
    crop = tile.get('crop') if isinstance(tile, dict) else None
    if crop in CROPS:
        cost, first, peak, units, last = CROPS[crop]
        age = day - tile.get('planted_day', day)
        yu = tile.get('yield_units', 0)
        p[16] = min(1.0, max(0, units - yu) / 6.0)          # yield headroom
        p[17] = min(1.0, max(0, first - age) / 12.0)        # turns to maturity
        lo = 2 if crop in ('WHEAT', 'CARROT') else 6
        p[18] = 1.0 if lo <= age <= last else 0.0           # watering pays today
        p[19] = 1.0 if crop in ('TOMATO', 'STRAWBERRY') else 0.0
        mls = tile.get('max_lifespan_step', -1)
        if mls >= 0:
            p[20] = min(1.0, max(0, mls - (day * tpd + hour)) / (4.0 * tpd))
        p[21] = min(1.0, cost / 100.0)
    if is_animal:
        p[22] = min(1.0, tile.get('consecutive_unfed', 0) / 2.0)
        p[23] = 1.0 if tile.get('cared_today') else 0.0
    p[24] = min(1.0, day / max(1.0, float(final_day)))
    p[25] = 1.0 if day >= final_day else 0.0
    return p


def distance(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])


def move(a,b):
    if a[0]<b[0]: return ['EAST']
    if a[0]>b[0]: return ['WEST']
    if a[1]<b[1]: return ['SOUTH']
    if a[1]>b[1]: return ['NORTH']
    return ['PASS']


def agent(obs, configuration=None):
    cfg=configuration or {}
    if not obs.get('farms'): return {'farmer':['PASS'],'hands':[],'market':[]}
    farm=obs['farms'][obs['player']]; private=obs['private']
    tiles=farm['tiles']; n=len(tiles); half=n//2
    shed_positions=[(half-1,half-1),(half,half-1),(half-1,half),(half,half)]
    positions=[farm['farmer']]+farm.get('hands',[])
    inventories=private['inventories']; seeds=dict(private.get('seeds',{}))
    shed=private.get('shed',{}); day=obs['day']; hour=obs['hour']
    tpd=cfg.get('turnsPerDay',24); step=obs.get('step',day*tpd+hour)
    final_step=cfg.get('episodeSteps',720)-2
    days_left=max(0,(final_step-step)//tpd)
    final_day=final_step//tpd; final_hour=final_step%tpd
    remaining=(final_hour-hour+1) if day==final_day else tpd-hour
    market_inv=obs['market'].get('inventory',{}); prices=obs['market']['prices']
    overrides=cfg.get('marketParams',{})
    demand={c:1.0 for c in PARAMS}; demand['FERTILIZER']=0
    for shop in obs.get('town',{}).get('unlocked_shops',[]):
        for c in SHOPS.get(shop,[]): demand[c]+=tpd/cfg.get('townShopSellInterval',4)
    n_animals=n_pens=0
    empty_pens={'PASTURE':0,'COOP':0}
    for _row in tiles:
        for _t in _row:
            if isinstance(_t,dict):
                if 'animal' in _t: n_animals+=1
                elif _t.get('kind') in ('COOP','PASTURE'):
                    n_pens+=1; empty_pens[_t['kind']]+=1
    owned={'COW':0,'SHEEP':0,'GOOSE':0}
    for _r in tiles:
        for _t in _r:
            if isinstance(_t,dict) and _t.get('animal') in owned: owned[_t['animal']]+=1
    _tot=owned['COW']+owned['SHEEP']+owned['GOOSE']
    if TUNE['goose_share']*(_tot+1)>owned['GOOSE']: want_animal='GOOSE'
    elif TUNE['sheep_share']*(_tot+1)>owned['SHEEP']: want_animal='SHEEP'
    else: want_animal='COW'
    want_kind=ANIMALS[want_animal]['pen']
    want_pens=0
    if day<=TUNE['animal_last_day'] and farm['money']>TUNE['animal_cash']:
        want_pens=max(0,int(TUNE['herd'])-n_animals-n_pens)
    commitments={c:0.0 for c in CROPS}
    own_count={c:0 for c in CROPS}
    for fi,f in enumerate(obs['farms']):
        for row in f['tiles']:
            for tile in row:
                if isinstance(tile,dict) and tile.get('kind')=='PLANT':
                    c=tile['crop']; age=day-tile['planted_day']; data=CROPS[c]
                    if age<=data[4]:
                        commitments[c]+=max(tile.get('yield_units',0),data[3]*(TUNE['opp_weight'] if fi!=obs['player'] else 1))
                        if fi==obs['player']: own_count[c]+=1
    def crop_score(c):
        cost,first,peak,units,last=CROPS[c]
        horizon=min(peak,days_left)
        if days_left<first+1: return -1000
        if c in ('TOMATO','STRAWBERRY'):
            units=min(4,1+(horizon-first)//(1 if c=='TOMATO' else 2))
        elif c=='WHEAT': units=min(4,max(1,horizon-1))
        elif c=='CARROT': units=min(3,max(1,horizon))
        forecast=market_inv.get(c,10000)+commitments[c]-demand[c]*horizon*.8
        estimated=(price(c,forecast,overrides)+price(c,forecast+units,overrides))/2
        profit=estimated*units-cost
        # Blend tile-days and worker-actions. At labor_weight=1 this is pure
        # coins-per-action, which ranks MELON (~118) far above STRAWBERRY (~27).
        w=TUNE['labor_weight']
        return profit/((horizon+1)**(1-w)*LABOR[c]**w)
    crop_values={c:crop_score(c) for c in CROPS}
    best_crop=max(CROPS,key=lambda c:crop_values[c])
    best_value=crop_values[best_crop]
    active=sum(own_count.values())
    unlocked=len(farm.get('unlocked_quadrants',['NW']))
    money=farm['money']; orders=[]; keep=[]; expected_drop={}
    feed_hold=n_animals*max(1,prices.get('WHEAT',25))*TUNE['feed_days']
    # Cash and stock accounting includes same-turn deliveries: actions precede market.
    reserved=set(); tasks=[]; op_count={}
    for y,row in enumerate(tiles):
        for x,tile in enumerate(row):
            target=(x,y)
            if tile=='LOCKED': continue
            if tile is None or (isinstance(tile,dict) and tile.get('kind')=='WEED'):
                if (tile is None and want_pens>0 and remaining>=3
                        and distance(target,shed_positions[0])<=TUNE['barn_r']):
                    want_pens-=1
                    tasks.append((target,'BUILD_'+want_kind,TUNE['v_pen'],None))
                elif (best_value>2 and remaining>=3
                        and distance(target,shed_positions[0])<=TUNE['plant_radius']):
                    tasks.append((target,'PLANT',max(4,best_value*TUNE['v_plant']),best_crop))
                continue
            if 'animal' in tile:
                ad=ANIMALS.get(tile['animal'])
                if ad is None: continue
                pprice=prices.get(ad['prod'],ad['cost'])
                ayu=tile.get('yield_units',0)
                # Every animal produces one fertilizer a day, unconditionally --
                # the engine sets fertilizer_available on every animal at every
                # end of day, fed or not. It is free output from livestock the
                # farm already owns and was being left on the ground.
                if tile.get('fertilizer_available') and day!=final_day:
                    tasks.append((target,'COLLECT_FERTILIZER',
                                  prices.get('FERTILIZER',100)*TUNE['v_fert'],None))
                if ayu>=TUNE['animal_harvest'] or ayu>=ad['held'] or (day==final_day and ayu>0):
                    tasks.append((target,'HARVEST',ayu*pprice*TUNE['v_aharv'],None))
                elif not tile.get('fed_today') and day!=final_day:
                    # Fed every day on purpose. Feeding only every other day
                    # (enough to survive) was measured at 112 milk against 168,
                    # because the CARE bonus is consumed only on a fed
                    # production tick -- the lost milk far outweighs the wheat.
                    urgent=tile.get('consecutive_unfed',0)>=1
                    tasks.append((target,'FEED',pprice*(TUNE['v_feed_u'] if urgent else TUNE['v_feed']),'WHEAT'))
                elif not tile.get('cared_today') and day!=final_day:
                    tasks.append((target,'CARE',pprice*TUNE['v_care'],None))
                continue
            if tile.get('kind') in ('COOP','PASTURE'):
                # One task per species that fits this pen. COW and SHEEP share
                # PASTURE, so offering only the first leaves a unit carrying the
                # other unable to place it: it loops on PICKUP and end-of-day
                # auto-drop returns the animal to the shed, forever.
                for a in ANIMALS:
                    if ANIMALS[a]['pen']==tile.get('kind'):
                        tasks.append((target,'PLACE',TUNE['v_place'],a))
                continue
            if tile.get('kind')!='PLANT': continue
            c=tile['crop']; cost,first,peak,units,last=CROPS[c]; age=day-tile['planted_day']
            yield_units=tile.get('yield_units',0)
            mature=age>=first and yield_units>0
            end=(day==final_day)
            harvest=mature and (age>=peak or end or tile.get('max_lifespan_step',-1)>=0 and step>=tile['max_lifespan_step']-tpd)
            if harvest:
                # Water before harvest if it adds an immediate one-time yield.
                window=2 if c in ('WHEAT','CARROT') else 6
                cap=6 if c in ('WHEAT','MELON') else 4
                if c not in ('TOMATO','STRAWBERRY') and not tile['watered_today'] and window<=age<=last and yield_units<cap:
                    tasks.append((target,'WATER',max(TUNE['v_water_h'],prices.get(c,25)*yield_units),None))
                else: tasks.append((target,'HARVEST',max(TUNE['v_harv'],prices.get(c,25)*yield_units*TUNE['v_harv_m']),None))
            elif not tile.get('watered_today') and not end:
                urgency=TUNE['water_urgent'] if tile.get('consecutive_unwatered',0)>=1 else TUNE['water_base']
                tasks.append((target,'WATER',urgency+max(0,prices.get(c,25)*.2),None))
    hungry=sum(1 for _r in tiles for _t in _r
               if isinstance(_t,dict) and 'animal' in _t and not _t.get('fed_today'))
    carried_wheat=sum(i.get('WHEAT',0) for i in inventories)
    # Only a unit with wheat IN HAND may feed, so what matters is how many
    # units carry some, not how much wheat exists. Comparing hungry animals to
    # the total let one unit hold ten wheat and satisfy the test, leaving the
    # other eight units permanently unable to feed: in a real game 8 animals
    # stayed hungry all day beside 28 wheat in the shed, and four cows starved.
    carriers=sum(1 for i in inventories if i.get('WHEAT',0)>0)
    # An animal that missed yesterday dies tonight if it misses again, and only
    # a unit with wheat in hand may feed. Spreading wheat over more units costs
    # extra shed trips, so it is done only for animals actually at risk, not
    # for every animal that simply has not eaten yet today.
    at_risk=sum(1 for _r in tiles for _t in _r
                if isinstance(_t,dict) and 'animal' in _t
                and not _t.get('fed_today') and _t.get('consecutive_unfed',0)>=1)
    if ((hungry>carried_wheat or at_risk>carriers)
            and shed.get('WHEAT',0)>0 and day!=final_day):
        take=int(min(shed['WHEAT'],max(1,TUNE['feed_batch'])))
        for st in shed_positions:
            tasks.append((st,'PICKUP',TUNE['v_pick_w'],('WHEAT',take)))
    carried_animals=sum(i.get(a,0) for i in inventories for a in ANIMALS)
    if carried_animals==0 and n_pens>0:
        for a in ANIMALS:
            if shed.get(a,0)>0:
                for st in shed_positions:
                    tasks.append((st,'PICKUP',TUNE['v_pick_a'],(a,shed[a])))
    # W . phi, precomputed per task. phi differs per unit only at indices 8
    # (distance), 12 (crowding) and 13 (same op already chosen), so the dot
    # product is reused and the inner loop does three multiplies instead of 26.
    base_dot=[]
    for (t,op,_v,_c) in tasks:
        ph=task_features(op,tiles[t[1]][t[0]],day,hour,tpd,final_day)
        base_dot.append(sum(W[k]*ph[k] for k in range(NW)))

    # Optimal assignment of the whole crew at once. Scores are built for every
    # (worker, task) pair that is legal for that worker -- a worker with no
    # wheat cannot FEED, one with no seed cannot PLANT -- then Hungarian picks
    # the best set of pairs jointly instead of first-come-first-served.
    assign={}
    if TUNE['optimal_assign']:
        free=[i for i,pos in enumerate(positions)
              if not (sum((inventories[i] if i<len(inventories) else {}).values())
                      and (day==final_day
                           and remaining<=min(distance(pos,q) for q in shed_positions)+2
                           or sum((inventories[i] if i<len(inventories) else {}).values())
                           >=TUNE['carry_cap']))]
        BIG=1e9
        rows=[]
        for i in free:
            pos=positions[i]
            inv=inventories[i] if i<len(inventories) else {}
            row=[]
            for ti,(target,op,value,c) in enumerate(tasks):
                dist=distance(pos,target)
                needed=2+(1 if tiles[target[1]][target[0]] is not None else 0) if op=='PLANT' else 1
                ok=True
                if op=='PLANT' and seeds.get(c,0)<=0: ok=False
                if op=='FEED' and inv.get('WHEAT',0)<=0: ok=False
                if op=='PLACE' and inv.get(c,0)<=0: ok=False
                if dist+needed>remaining: ok=False
                if not ok: row.append(BIG); continue
                z=base_dot[ti]+W[8]*(dist/10.0)
                if z>3.0: z=3.0
                elif z<-3.0: z=-3.0
                row.append(-value/(dist+TUNE['travel_cost'])*math.exp(z))
            rows.append(row)
        if rows and tasks:
            # Pad so every worker has a column to take (doing nothing).
            width=len(tasks)+len(rows)
            for k,row in enumerate(rows):
                row.extend([0.0 if j==k else BIG for j in range(len(rows))])
            sol=hungarian([r[:width] for r in rows])
            # The joint solve has no notion of a shared stock, so it will
            # happily send three workers to plant one seed -- and the engine
            # drops ALL of a crop's PLANTs when more are ordered than held.
            # Keep the best-scoring ones and let the rest fall back to greedy.
            picked=sorted(((rows[k][sol[k]],k) for k in range(len(free))
                           if 0<=(sol[k] if k<len(sol) else -1)<len(tasks)
                           and rows[k][sol[k]]<BIG/2))
            budget=dict(seeds)
            for _sc,k in picked:
                j=sol[k]; target,op,value,c=tasks[j]
                if op=='PLANT':
                    if budget.get(c,0)<=0: continue
                    budget[c]-=1
                assign[free[k]]=j
    actions=[]
    for idx,pos in enumerate(positions):
        inv=inventories[idx] if idx<len(inventories) else {}
        closest=min(shed_positions,key=lambda p:distance(pos,p)); home_dist=distance(pos,closest)
        carrying=sum(inv.values())
        if carrying and (day==final_day and remaining<=home_dist+2 or carrying>=TUNE['carry_cap']):
            action=['DROP'] if home_dist==0 else move(pos,closest)
            if home_dist==0:
                for c,q in inv.items(): expected_drop[c]=expected_drop.get(c,0)+q
            actions.append(action); continue
        selected=None; selected_score=-1
        if (POLICY is not None or RLLOG is not None) and tasks:
            # Score every legal task for this unit, then let the policy choose.
            cand=[]
            for ti,(target,op,value,c) in enumerate(tasks):
                if target in reserved: continue
                dist=distance(pos,target)
                needed=2+(1 if tiles[target[1]][target[0]] is not None else 0) if op=='PLANT' else 1
                if op=='PLANT' and seeds.get(c,0)<=0: continue
                if op=='FEED' and inv.get('WHEAT',0)<=0: continue
                if op=='PLACE' and inv.get(c,0)<=0: continue
                if dist+needed>remaining: continue
                crowd=sum(1 for t in reserved if distance(t,target)<=2)
                z=(base_dot[ti]+W[8]*(dist/10.0)+W[12]*min(1.0,crowd/4.0)
                   +W[13]*min(1.0,op_count.get(op,0)/4.0))
                if z>3.0: z=3.0
                elif z<-3.0: z=-3.0
                sc=value/(dist+TUNE['travel_cost'])*math.exp(z)
                ph=features(op,dist,tiles[target[1]][target[0]],day,hour,tpd,final_day,
                            crowd,op_count.get(op,0))
                cand.append((ti,(target,op,c),sc,ph))
            if cand:
                pick=0
                if POLICY is not None:
                    pick=POLICY([c[3] for c in cand],
                                [math.log(max(1e-6,c[2])) for c in cand])
                else:
                    pick=max(range(len(cand)),key=lambda i:cand[i][2])
                if RLLOG is not None:
                    RLLOG.append(([c[3] for c in cand],
                                  [math.log(max(1e-6,c[2])) for c in cand],pick))
                selected=cand[pick][1]
        if selected is None and idx in assign:
            target,op,value,c=tasks[assign[idx]]
            if target not in reserved:
                selected=(target,op,c)
        if selected is None:
            for ti,(target,op,value,c) in enumerate(tasks):
                if target in reserved: continue
                dist=distance(pos,target)
                needed=1
                if op=='PLANT':
                    needed=2+(1 if tiles[target[1]][target[0]] is not None else 0)
                    if seeds.get(c,0)<=0: continue
                if op=='FEED' and inv.get('WHEAT',0)<=0: continue
                if op=='PLACE' and inv.get(c,0)<=0: continue
                if dist+needed>remaining: continue
                if day==final_day and op=='WATER':
                    delivery=min(distance(target,p) for p in shed_positions)
                    if dist+3+delivery>remaining:
                        op='HARVEST'
                if day==final_day and op=='HARVEST':
                    delivery=min(distance(target,p) for p in shed_positions)
                    if dist+1+delivery+1>remaining: continue
                # Autoregressive: crowd and op_count describe the assignments
                # already made this turn, so later units see earlier decisions.
                crowd=sum(1 for t in reserved if distance(t,target)<=2)
                z=(base_dot[ti]+W[8]*(dist/10.0)+W[12]*min(1.0,crowd/4.0)
                   +W[13]*min(1.0,op_count.get(op,0)/4.0))
                if z>3.0: z=3.0
                elif z<-3.0: z=-3.0
                score=value/(dist+TUNE['travel_cost'])*math.exp(z)
                if score>selected_score: selected_score=score; selected=(target,op,c)
        if selected:
            target,op,c=selected; reserved.add(target)
            op_count[op]=op_count.get(op,0)+1
            if tuple(pos)!=target: action=move(pos,target)
            elif op=='PLANT':
                if tiles[target[1]][target[0]] is not None: action=['DIG']
                else:
                    action=['PLANT',c]; seeds[c]-=1
            elif op=='PICKUP': action=['PICKUP',c[0],int(c[1])]
            elif op=='PLACE': action=['PLACE',c]
            else: action=[op]
        elif carrying:
            action=['DROP'] if home_dist==0 else move(pos,closest)
            if home_dist==0:
                for c,q in inv.items(): expected_drop[c]=expected_drop.get(c,0)+q
        else: action=['PASS']
        actions.append(action)
    # Count animals already in transit, not just those sitting in the shed:
    # a pen a unit is walking toward is not a pen that needs restocking.
    pending=(sum(shed.get(a,0) for a in ANIMALS)
             + sum(i.get(a,0) for i in inventories for a in ANIMALS))
    if day<=TUNE['animal_last_day'] and n_pens>pending:
        # Milk and wool are separate thin markets, so a herd of one species
        # saturates its own price while the other sits unsupplied. sheep_share
        # sets how much of the herd is sheep; 0 keeps the cow-only behaviour.
        total=_tot
        # Buying a goose when only pastures stand leaves it in the shed for
        # ever: the crew picks it up, cannot place it, and end-of-day auto-drop
        # returns it. Only buy what there is somewhere to put.
        a=want_animal
        if empty_pens.get(ANIMALS[a]['pen'],0)<=0:
            a=None
            for _cand in ANIMALS:
                if empty_pens.get(ANIMALS[_cand]['pen'],0)>0: a=_cand; break
        # Reserve the feed this herd will actually consume rather than a flat
        # constant. Each animal eats 2 wheat a day, so an animal bought on day d
        # commits ~2*price*(days left) on top of its purchase price. A fixed
        # threshold ignores that and is a knife edge: 2303 measured 87k while
        # 2400 measured 21k, because affording the Nth cow on one exact turn
        # cascades through the whole season.
        if a is None: a='COW'; n_pens=pending
        wprice=max(1,prices.get('WHEAT',25))
        horizon=max(0,min(final_day-day,TUNE['feed_horizon']))
        reserve=(total+1)*2*wprice*horizon*TUNE['feed_safety']
        afford=int(max(0,(money-reserve)//ANIMALS[a]['cost']))
        qty=min(n_pens-pending,afford,3)
        if qty>0:
            orders.append(['BUY_ANIMAL',a,qty]); money-=qty*ANIMALS[a]['cost']
    if n_animals>0 and day!=final_day:
        wp=max(1,prices.get('WHEAT',25))
        # Buy a buffer, not one day's ration. An animal unfed twice escapes,
        # taking its cost and every future yield; in real games the herd was
        # lost outright in 3 of 5 episodes while the winners bought 235 wheat
        # to our 31. One thin turn -- a dip in cash, or a market order pushed
        # off the end of the cap -- is enough to start that.
        need=int(n_animals*2*TUNE['feed_buffer']-shed.get('WHEAT',0)-carried_wheat)
        qty=max(0,min(need,int(max(0,(money-200)//wp)),int(TUNE['feed_max'])))
        if qty>0:
            keep.append(['BUY_PRODUCT','WHEAT',qty]); money-=qty*wp

    for c in PARAMS:
        quantity=shed.get(c,0)+expected_drop.get(c,0)
        if c=='WHEAT' and n_animals>0 and day!=final_day:
            # Wheat held back as feed; selling it just to re-buy it next turn
            # churns market-order slots for nothing.
            # Hold back the SAME buffer the feed purchase targets. Reserving
            # one day while buying two made the farm buy 1,718 wheat and sell
            # 1,625 of it straight back in a real game: a loss on every unit
            # and a market-order slot burned every turn.
            quantity-=int(n_animals*2*TUNE['feed_buffer'])
        if quantity<=0: continue
        inv_now=market_inv.get(c,10000); base=PARAMS[c][0]
        if days_left<=TUNE['sell_clear']:
            qty=quantity
        else:
            # Units sellable before the marginal price falls through the floor.
            cap=0
            while cap<quantity and price(c,inv_now+cap,overrides)>=TUNE['sell_floor']*base:
                cap+=1
            # Never stall: stock still has to clear before the season ends.
            must=-(-quantity//max(1,int(min(days_left,TUNE['sell_spread']))))
            qty=min(quantity,max(cap,must))
        if qty>0: orders.append(['SELL',c,int(qty)])
    # Reserve cash for seeds, workers and ongoing operation before expansion.
    if hour<2 and unlocked<TUNE['max_land'] and day<TUNE['land_last_day'] and active>=unlocked*(n//2)**2*TUNE['land_fill']:
        cost=[1000,2000,4000][unlocked-1]
        if money>cost+TUNE['land_cash']:
            keep.append(['BUY_LAND']); money-=cost
    workers=TUNE['workers_hi'] if active>TUNE['active_hi'] else (TUNE['workers_mid'] if active>TUNE['active_lo'] else TUNE['workers_lo'])
    if day==0: workers=TUNE['workers_mid']
    if best_value<0 and active==0: workers=0
    if hour<=2 and remaining>10:
        fib=[1,1,2,3,5,8,13,21,34,55,89,144]
        for hire in range(farm.get('hires_today',0),int(workers)):
            cost=fib[hire]*cfg.get('farmHandCostMult',1)
            # Feed is a standing daily bill and an animal unfed for two days
            # escapes, taking 400 coins and every future tick of milk with it.
            # A marginal hand is worth a few coins a day, so payroll yields to
            # the herd: at 10 hands the farm ran out of wheat money by day 8 and
            # had lost the whole herd by day 22.
            if money-cost < feed_hold+50: break
            keep.append(['HIRE']); money-=cost
    # Buffer only a few seeds, avoiding capital tied up in obsolete crop choices.
    if best_value>2 and hour<tpd-3:
        free_tiles=sum(t is None or isinstance(t,dict) and t.get('kind')=='WEED' for row in tiles for t in row)
        free_tiles-=sum(a[0]=='PLANT' for a in actions)
        desired=min(max(0,free_tiles),int(TUNE['seed_cap']),max(2,len(positions)))
        needed=max(0,desired-seeds.get(best_crop,0))
        affordable=max(0,int((money-60)//CROPS[best_crop][0]))
        quantity=min(needed,affordable)
        if quantity: keep.append(['BUY_SEED',best_crop,quantity])
    # Orders past maxMarketOrdersPerTurn are silently discarded. Sells go first
    # so their cash is available to the buys behind them in the same turn, but
    # they must never crowd out payroll or investment: a morning with a full
    # shed produces many sell orders, and at nine hands that pushed BUY_SEED off
    # the end entirely -- the farm stopped sowing and scores halved.
    cap=int(cfg.get('maxMarketOrdersPerTurn',10))
    # Within the protected list, hires yield last: a dropped HIRE costs one
    # hand for one day, a dropped BUY_SEED costs a tile for the rest of the
    # season. Stable sort, so the rest keep their order.
    keep.sort(key=lambda o: 1 if o[0]=='HIRE' else 0)
    room=max(0,cap-len(keep))
    orders.sort(key=lambda o:-(prices.get(o[1],0)*o[2]) if o[0]=='SELL' else 0)
    market=(orders[:room]+keep)[:cap]
    return {'farmer':actions[0],'hands':actions[1:],'market':market}
