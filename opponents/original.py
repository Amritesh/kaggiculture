"""Kaggriculture: coordinated, market-aware farm policy. Standard library only."""
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
SHOPS = {'BAKERY':['EGG','WHEAT'], 'PIZZA_SHOP':['MILK','TOMATO','WHEAT'],
 'BRUNCH_SPOT':['EGG','WHEAT','STRAWBERRY'], 'YARN_STORE':['WOOL','WOOL'],
 'ICE_CREAM_SHOP':['STRAWBERRY','MILK','WHEAT'], 'PET_CAFE':['CARROT','CARROT'],
 'SMOOTHIE_SHOP':['STRAWBERRY','MILK'], 'FARMERS_MARKET':['WHEAT','CARROT','TOMATO','STRAWBERRY']}
WORKERS = 8
MAX_LAND = 3


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
    commitments={c:0.0 for c in CROPS}
    own_count={c:0 for c in CROPS}
    for fi,f in enumerate(obs['farms']):
        for row in f['tiles']:
            for tile in row:
                if isinstance(tile,dict) and tile.get('kind')=='PLANT':
                    c=tile['crop']; age=day-tile['planted_day']; data=CROPS[c]
                    if age<=data[4]:
                        commitments[c]+=max(tile.get('yield_units',0),data[3]*(.8 if fi!=obs['player'] else 1))
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
        return (estimated*units-cost)/(horizon+1)
    crop_values={c:crop_score(c) for c in CROPS}
    best_crop=max(CROPS,key=lambda c:crop_values[c])
    best_value=crop_values[best_crop]
    active=sum(own_count.values())
    unlocked=len(farm.get('unlocked_quadrants',['NW']))
    money=farm['money']; orders=[]; expected_drop={}
    # Cash and stock accounting includes same-turn deliveries: actions precede market.
    reserved=set(); tasks=[]
    for y,row in enumerate(tiles):
        for x,tile in enumerate(row):
            target=(x,y)
            if tile=='LOCKED': continue
            if tile is None or (isinstance(tile,dict) and tile.get('kind')=='WEED'):
                if best_value>2 and remaining>=3:
                    tasks.append((target,'PLANT',max(4,best_value*2),best_crop))
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
                    tasks.append((target,'WATER',max(250,prices.get(c,25)*yield_units),None))
                else: tasks.append((target,'HARVEST',max(300,prices.get(c,25)*yield_units*2),None))
            elif not tile.get('watered_today') and not end:
                urgency=400 if tile.get('consecutive_unwatered',0)>=1 else 110
                tasks.append((target,'WATER',urgency+max(0,prices.get(c,25)*.2),None))
    actions=[]
    for idx,pos in enumerate(positions):
        inv=inventories[idx] if idx<len(inventories) else {}
        closest=min(shed_positions,key=lambda p:distance(pos,p)); home_dist=distance(pos,closest)
        carrying=sum(inv.values())
        if carrying and (day==final_day and remaining<=home_dist+2 or carrying>=35):
            action=['DROP'] if home_dist==0 else move(pos,closest)
            if home_dist==0:
                for c,q in inv.items(): expected_drop[c]=expected_drop.get(c,0)+q
            actions.append(action); continue
        selected=None; selected_score=-1
        for target,op,value,c in tasks:
            if target in reserved: continue
            dist=distance(pos,target)
            needed=1
            if op=='PLANT':
                needed=2+(1 if tiles[target[1]][target[0]] is not None else 0)
                if seeds.get(c,0)<=0: continue
            if dist+needed>remaining: continue
            if day==final_day and op=='WATER':
                delivery=min(distance(target,p) for p in shed_positions)
                if dist+3+delivery>remaining:
                    op='HARVEST'
            if day==final_day and op=='HARVEST':
                delivery=min(distance(target,p) for p in shed_positions)
                if dist+1+delivery+1>remaining: continue
            score=value/(dist+1.5)
            if score>selected_score: selected_score=score; selected=(target,op,c)
        if selected:
            target,op,c=selected; reserved.add(target)
            if tuple(pos)!=target: action=move(pos,target)
            elif op=='PLANT':
                if tiles[target[1]][target[0]] is not None: action=['DIG']
                else:
                    action=['PLANT',c]; seeds[c]-=1
            else: action=[op]
        elif carrying:
            action=['DROP'] if home_dist==0 else move(pos,closest)
            if home_dist==0:
                for c,q in inv.items(): expected_drop[c]=expected_drop.get(c,0)+q
        else: action=['PASS']
        actions.append(action)
    for c in PARAMS:
        quantity=shed.get(c,0)+expected_drop.get(c,0)
        if quantity>0: orders.append(['SELL',c,quantity])
    # Reserve cash for seeds, workers and ongoing operation before expansion.
    if hour<2 and unlocked<MAX_LAND and day<17 and active>=unlocked*(n//2)**2*.65:
        cost=[1000,2000,4000][unlocked-1]
        if money>cost+900:
            orders.append(['BUY_LAND']); money-=cost
    workers=WORKERS if active>35 else (6 if active>15 else 4)
    if day==0: workers=6
    if best_value<0 and active==0: workers=0
    if hour<=2 and remaining>10:
        fib=[1,1,2,3,5,8,13,21,34,55,89,144]
        for hire in range(farm.get('hires_today',0),workers):
            cost=fib[hire]*cfg.get('farmHandCostMult',1)
            if money<cost+50: break
            orders.append(['HIRE']); money-=cost
    # Buffer only a few seeds, avoiding capital tied up in obsolete crop choices.
    if best_value>2 and hour<tpd-3:
        free_tiles=sum(t is None or isinstance(t,dict) and t.get('kind')=='WEED' for row in tiles for t in row)
        free_tiles-=sum(a[0]=='PLANT' for a in actions)
        desired=min(max(0,free_tiles),8,max(2,len(positions)))
        needed=max(0,desired-seeds.get(best_crop,0))
        affordable=max(0,int((money-60)//CROPS[best_crop][0]))
        quantity=min(needed,affordable)
        if quantity: orders.append(['BUY_SEED',best_crop,quantity])
    return {'farmer':actions[0],'hands':actions[1:],'market':orders[:cfg.get('maxMarketOrdersPerTurn',10)]}
