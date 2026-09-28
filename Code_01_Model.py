"""
Code_01_Model.py
Core model for the study of retrofit and fuel-switching decisions for a mid-life bulk carrier
under the IMO Net-Zero Framework (see README.md).

Contents
  1. Technology data, read from 'EETs 12.csv' (Table 2), with the worst-case bounding rule
     of Section 3.3 (maximum CAPEX/OPEX, minimum efficiency gain).
  2. Vessel, fuel and regulatory parameters (Tables 1, 3, 4; Section 3.5).
  3. The two-tier GFI deficit charge, Eqs. (4)-(5).
  4. The MILP of Section 3.7, Eqs. (1)-(12), built with explicit variables
     x, z, v, w, y, u, q, A, e and Phi, and solved with HiGHS (scipy.optimize.milp).
  5. The two-step solution procedure of Section 3.7.
Options used by the robustness analyses: multiplicative efficiency (Section 4.4),
residual value, Eq. (13) (Section 4.4), relaxed dry-dock timing (Appendix D),
and fuel-price states (Section 3.5 / 4.6).

All money values are in thousand USD (kUSD). Energy in GJ.
"""
import os, re, time
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Technology data (Table 2)
# ---------------------------------------------------------------------------
CODES = {  # technology name in 'EETs 12.csv' -> code used in the manuscript
    'Wind-assisted propulsion systems': 'T1', 'Air lubrication system': 'T2',
    'High performance coatings': 'T3', 'ESDs AFT of the propeller': 'T4',
    'Propeller measures': 'T5', 'Shaft generator (PTO/PTI)': 'T6',
    'Hull and propeller cleaning': 'T7', 'Electronic auto-tuning': 'T8',
    'Engine performance testing and tuning': 'T9', 'Variable frequency drives': 'T10',
    'Trim optimization': 'T11', 'Weather routing': 'T12'}
# Recurring service contracts (Table 2, note **): CAPEX moved to annual OPEX (kUSD/year)
SERVICE_OPEX = {'Weather routing': 50.0, 'Hull and propeller cleaning': 50.0,
                'Engine performance testing and tuning': 150.0}

def _nums(s):
    return [float(n) for n in re.findall(r'\d*\.\d+|\d+', str(s).replace(',', ''))]

def load_technologies(csv_file=os.path.join(HERE, 'EETs 12.csv')):
    df = pd.read_csv(csv_file)
    tech = {}
    for _, row in df.iterrows():
        name = row['Technology'].strip()
        capex_n = _nums(row['CAPEX (kUSD)'])
        capex = max(capex_n) if capex_n else 0.0                     # maximum CAPEX
        eff = min(_nums(row['Efficiency Gain (%)'])) / 100.0          # minimum efficiency gain
        op_txt = str(row['OPEX (% or kUSD)'])
        op_n = _nums(op_txt)
        opex = (max(op_n) / 100.0 * capex if '%' in op_txt else max(op_n)) if op_n else 0.0
        if name in SERVICE_OPEX:
            capex, opex = 0.0, SERVICE_OPEX[name]
        level = str(row['Retrofit level']).lower()
        structural = ('docking' in level) and ('in operation' not in level) and ('in operating' not in level)
        tech[CODES[name]] = dict(name=name, cls='st' if structural else 'op', eff=eff, capex=capex, opex=opex)
    return dict(sorted(tech.items(), key=lambda kv: int(kv[0][1:])))

TECH = load_technologies()
OP = [i for i in TECH if TECH[i]['cls'] == 'op']
ST = [i for i in TECH if TECH[i]['cls'] == 'st']

# ---------------------------------------------------------------------------
# 2. Parameters (Table 1, Table 4, Section 3.5)
# ---------------------------------------------------------------------------
T = list(range(1, 11))          # model years 1..10 = 2028..2037
DOCKS = [3, 8]                  # statutory dry-dock years, set D
R = 0.08                        # discount rate
F0 = 12000.0                    # t HFO per year
LHV = 40.5                      # GJ/t
E0 = F0 * LHV                   # 486,000 GJ per year
HFO_PRICE_T = 600.0             # USD/t
HFO_GJ = HFO_PRICE_T / LHV      # 14.81 USD/GJ
BUDGET = 8500.0                 # EET capital ceiling B (kUSD), Eq. (8a)
BUDGET_F = 14000.0              # fuel conversion envelope B_F (kUSD), Eq. (8b)
TCE = 20.0                      # kUSD per day
OFF_DAYS_ST, OFF_DAYS_F = 30, 45
ABAR = 0.25                     # efficiency ceiling, Eq. (10b)

# Fuel pathways (Table 4): WtW intensity (gCO2eq/MJ), price multiple of HFO, convertible fuel
PATH = {'HFO':    (94.0, 1.00, None),
        'LNG':    (85.0, 1.40, 'LNG'),
        'MeOH_f': (95.0, 1.30, 'MeOH'),
        'MeOH_r': (8.0,  6.50, 'MeOH'),
        'NH3_f':  (103.0, 1.10, 'NH3'),
        'NH3_r':  (5.0,  2.33, 'NH3')}
PATH_LABEL = {'HFO': 'Heavy fuel oil', 'LNG': 'LNG', 'MeOH_f': 'Methanol (fossil)', 'MeOH_r': 'Methanol (renewable)',
              'NH3_f': 'Ammonia (fossil)', 'NH3_r': 'Ammonia (renewable)'}
FUELCAP = {'LNG': 12000.0, 'MeOH': 10000.0, 'NH3': 14000.0}   # conversion CAPEX (kUSD)

# IMO Net-Zero Framework (Table 3): Base Target reduction (%) by model year; Direct Compliance = Base + 13
G2008 = 93.3
BASE_RED = {1: 4.0, 2: 6.0, 3: 8.0, 4: 12.4, 5: 16.8, 6: 21.2, 7: 25.6, 8: 30.0}
BASE_2040 = 65.0                # 2040 Base Target reduction (%); Years 9-10 interpolated (Section 3.5)
DC_GAP = 13.0
TAU1, TAU2 = 100.0, 380.0       # USD per t CO2eq
ETS_PRICE, EU_SHARE = 85.0, 0.20   # Scenario S1: EU ETS price (USD/t) on 20% of WtW emissions

def base_reduction(t, reg, pace=1.0):
    """Base Target reduction (fraction) in model year t. S2 moves linearly to the 2040 target after 2035;
    S3 tightens at twice the S2 pace; `pace` scales the post-2035 pace (Section 4.4)."""
    if t <= 8:
        return BASE_RED[t] / 100.0
    step = (BASE_2040 - BASE_RED[8]) / 5.0 * pace * (2.0 if reg == 'S3' else 1.0)
    return (BASE_RED[8] + step * (t - 8)) / 100.0

def targets(t, reg, pace=1.0):
    bB = base_reduction(t, reg, pace)
    return G2008 * (1 - bB), G2008 * (1 - bB - DC_GAP / 100.0)          # Eq. (4)

def deficit_charge(g, t, reg, pace=1.0):
    """Two-tier deficit charge in USD per GJ, Eq. (5); S1 applies the EU ETS to 20% of emissions."""
    if reg == 'S1':
        return g * ETS_PRICE * EU_SHARE / 1000.0
    GB, GD = targets(t, reg, pace)
    return (TAU1 * min(max(g - GD, 0.0), GB - GD) + TAU2 * max(g - GB, 0.0)) / 1000.0

def disc(t):
    return 1.0 / (1.0 + R) ** t

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario(reg, name=None, price_mult=None, fossil_mult=1.0, renew_mult=1.0, pace=1.0, obs=None):
    """price_mult overrides the multiple of HFO for given pathways; fossil_mult / renew_mult implement
    the fuel-price states F1-F3 (Section 3.5)."""
    mult = {p: PATH[p][1] for p in PATH}
    if price_mult: mult.update(price_mult)
    price = {}
    for p in PATH:
        f = renew_mult if p.endswith('_r') else fossil_mult
        price[p] = mult[p] * HFO_GJ * f
    nm = name or reg
    return dict(name=nm, reg=reg, price=price, pace=pace, obs=obs or nm)

BASE_SCENARIOS = [scenario('S1'), scenario('S2'), scenario('S3')]
FUEL_STATES = {'F1': (0.75, 1.00), 'F2': (1.00, 1.00), 'F3': (1.25, 0.75)}   # (fossil, renewable) multipliers

def nine_scenarios(observe='full'):
    out = []
    for reg in ['S1', 'S2', 'S3']:
        for f, (fm, rm) in FUEL_STATES.items():
            nm = f'{reg}-{f}'
            out.append(scenario(reg, nm, fossil_mult=fm, renew_mult=rm, obs=nm if observe == 'full' else reg))
    return out

# ---------------------------------------------------------------------------
# 4. MILP builder
# ---------------------------------------------------------------------------
class _LP:
    def __init__(s): s.lb, s.ub, s.integ, s.rows, s.n = [], [], [], [], 0
    def var(s, lb=0.0, ub=np.inf, binary=False):
        s.lb.append(0.0 if binary else lb); s.ub.append(1.0 if binary else ub); s.integ.append(1 if binary else 0)
        s.n += 1; return s.n - 1
    def add(s, coef, lo, hi): s.rows.append((coef, lo, hi))
    def solve(s, obj):
        c = np.zeros(s.n)
        for k, v in obj.items(): c[k] += v
        A = lil_matrix((len(s.rows), s.n)); lo = np.empty(len(s.rows)); hi = np.empty(len(s.rows))
        for r, (co, l, h) in enumerate(s.rows):
            for k, v in co.items(): A[r, k] += v
            lo[r], hi[r] = l, h
        res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.array(s.integ),
                   bounds=Bounds(np.array(s.lb), np.array(s.ub)), options={'mip_rel_gap': 1e-9})
        if res.status != 0: raise RuntimeError(res.message)
        return res

def solve(scens, adaptive=True, criterion='regret', Cstar=None, fuelcap=None, residual=None,
          multiplicative=False, relaxed=None, fix_op=None, fix_st=None, allow_eet=True,
          allow_fuel=True, gwtw=None, two_step=True):
    """
    scens      : list of scenario dicts
    adaptive   : True -> structural and conversion decisions indexed by scenario (Section 3.1)
    criterion  : 'regret' (needs Cstar) or 'cost' (C*(s) = 0), Eq. (2)
    residual   : None, or X (extra trading years) for the residual-value credit of Eq. (13)
    relaxed    : None, or (st_in, st_out, conv_in, conv_out) off-hire days: docking allowed in every year (Appendix D)
    two_step   : apply the two-step procedure of Section 3.7
    """
    fc = dict(FUELCAP); fc.update(fuelcap or {})
    g = {p: PATH[p][0] for p in PATH}; g.update(gwtw or {})
    D = list(T) if relaxed else DOCKS
    def off_st(k): return ((relaxed[0] if k in DOCKS else relaxed[1]) if relaxed else OFF_DAYS_ST) * TCE
    def off_f(k):  return ((relaxed[2] if k in DOCKS else relaxed[3]) if relaxed else OFF_DAYS_F) * TCE
    S = [s['name'] for s in scens]
    key = {s['name']: (s['obs'] if adaptive else 'all') for s in scens}
    K = sorted(set(key.values()))
    m = _LP()
    x = {(i, t): m.var(binary=True) for i in OP for t in T}
    z = {(i, t): m.var(binary=True) for i in OP for t in T}
    v = {(i, k, h): m.var(binary=True) for i in ST for k in D for h in K}
    y = {(f, k, h): m.var(binary=True) for f in fc for k in D for h in K}
    w = {(i, t, s): m.var(binary=True) for i in ST for t in T for s in S}
    u = {(f, t, s): m.var(binary=True) for f in fc for t in T for s in S}
    q = {(p, t, s): m.var(binary=True) for p in PATH for t in T for s in S}
    A = {(t, s): m.var(0.0, ABAR) for t in T for s in S}                       # Eq. (10b) as a bound
    e = {(p, t, s): m.var(0.0, ABAR) for p in PATH for t in T for s in S}
    Phi = m.var(-np.inf, np.inf)
    # Constraint 1 (6a-6c)
    for i in OP: m.add({x[i, t]: 1 for t in T}, -np.inf, 1)
    for h in K:
        for i in ST: m.add({v[i, k, h]: 1 for k in D}, -np.inf, 1)
        m.add({y[f, k, h]: 1 for f in fc for k in D}, -np.inf, 1)
    # Constraint 2 (7a)
    for i in OP:
        for t in T:
            co = {z[i, t]: 1}
            for tau in range(1, t + 1): co[x[i, tau]] = -1
            m.add(co, 0, 0)
    for s in S:
        h = key[s]
        for t in T:
            for i in ST:   # (7b)
                co = {w[i, t, s]: 1}
                for k in D:
                    if k <= t: co[v[i, k, h]] = -1
                m.add(co, 0, 0)
            for f in fc:   # (7c)
                co = {u[f, t, s]: 1}
                for k in D:
                    if k <= t: co[y[f, k, h]] = -1
                m.add(co, 0, 0)
    # Constraints 3-4 (8a, 8b)
    for h in K:
        co = {x[i, t]: TECH[i]['capex'] for i in OP for t in T}
        co.update({v[i, k, h]: TECH[i]['capex'] for i in ST for k in D})
        m.add(co, -np.inf, BUDGET)
        m.add({y[f, k, h]: fc[f] for f in fc for k in D}, -np.inf, BUDGET_F)
    for s in S:
        for t in T:
            m.add({q[p, t, s]: 1 for p in PATH}, 1, 1)                                   # (9a)
            for p in PATH:
                f = PATH[p][2]
                if f: m.add({q[p, t, s]: 1, u[f, t, s]: -1}, -np.inf, 0)                  # (9b)
            if not multiplicative:                                                        # (10a)
                co = {A[t, s]: 1}
                for i in OP: co[z[i, t]] = -TECH[i]['eff']
                for i in ST: co[w[i, t, s]] = -TECH[i]['eff']
                m.add(co, -np.inf, 0)
            else:  # multiplicative variant (Section 4.4): 1 - A >= prod(1 - eta), tangent cuts of exp()
                for a in np.linspace(np.log(0.75), 0.0, 60):
                    ea = np.exp(a); co = {A[t, s]: 1}
                    for i in OP:
                        if TECH[i]['eff'] > 0: co[z[i, t]] = ea * np.log(1 - TECH[i]['eff'])
                    for i in ST:
                        if TECH[i]['eff'] > 0: co[w[i, t, s]] = ea * np.log(1 - TECH[i]['eff'])
                    m.add(co, -np.inf, 1 - ea * (1 - a))
            for p in PATH:                                                                # (11a-11c)
                m.add({e[p, t, s]: 1, A[t, s]: -1}, -np.inf, 0)
                m.add({e[p, t, s]: 1, q[p, t, s]: -ABAR}, -np.inf, 0)
                m.add({e[p, t, s]: 1, A[t, s]: -1, q[p, t, s]: -ABAR}, -ABAR, np.inf)
    if not allow_eet:
        for i in OP:
            for t in T: m.add({x[i, t]: 1}, 0, 0)
        for key_ in v: m.add({v[key_]: 1}, 0, 0)
    if not allow_fuel:
        for key_ in y: m.add({y[key_]: 1}, 0, 0)
    if fix_op is not None:
        for (i, t), val in fix_op.items(): m.add({x[i, t]: 1}, val, val)
    if fix_st is not None:
        for (i, k, h), val in fix_st.items(): m.add({v[i, k, h]: 1}, val, val)
    # Cost of ownership C(s), Eq. (3)
    cost = {}
    sc_by = {sc['name']: sc for sc in scens}
    for s in S:
        sc = sc_by[s]; h = key[s]; C = {}
        def add(k_, c_): C[k_] = C.get(k_, 0.0) + c_
        for t in T:
            d = disc(t)
            for i in OP:
                add(x[i, t], d * TECH[i]['capex'])                        # (3a)
                add(z[i, t], d * TECH[i]['opex'])                         # (3c)
            for i in ST:
                add(w[i, t, s], d * TECH[i]['opex'])                      # (3c)
                if t in D: add(v[i, t, h], d * (TECH[i]['capex'] + off_st(t)))   # (3a)+(3b)
            for f in fc:
                if t in D: add(y[f, t, h], d * (fc[f] + off_f(t)))                # (3a)+(3b)
            for p in PATH:                                                # (3d)
                cp = E0 / 1000.0 * (sc['price'][p] + deficit_charge(g[p], t, sc['reg'], sc['pace']))
                add(q[p, t, s], d * cp); add(e[p, t, s], -d * cp)
        if residual:   # Eq. (13): rho(k) = X / (T - d(k) + 1 + X)
            X = residual
            for k in D:
                rho = X / (len(T) - k + 1 + X)
                for i in ST: add(v[i, k, h], -disc(len(T)) * rho * TECH[i]['capex'])
                for f in fc: add(y[f, k, h], -disc(len(T)) * rho * fc[f])
        cost[s] = C
        cs = (Cstar[s] if criterion == 'regret' else 0.0)
        co = {Phi: 1.0}
        for k_, c_ in C.items(): co[k_] = co.get(k_, 0.0) - c_
        m.add(co, -cs, np.inf)                                            # Eq. (2)
    size = (m.n, len(m.rows))
    t0 = time.time()
    res = m.solve({Phi: 1.0})                                             # Eq. (1)
    phistar = res.x[Phi]
    if two_step:   # Section 3.7, second step
        m.add({Phi: 1.0}, -np.inf, phistar + 1e-6 * max(1.0, abs(phistar)))
        obj = {}
        for s in S:
            for k_, c_ in cost[s].items(): obj[k_] = obj.get(k_, 0.0) + c_
        res = m.solve(obj)
    secs = time.time() - t0
    X_ = res.x
    out = dict(phi=phistar, size=size, seconds=secs, cost={}, op=sorted((i, t) for (i, t), j in x.items() if X_[j] > 0.5),
               st={}, fuel={}, A={}, path={}, x=X_, index=dict(x=x, v=v, y=y, e=e, A=A, q=q))
    for s in S:
        h = key[s]
        out['cost'][s] = sum(c_ * X_[k_] for k_, c_ in cost[s].items())
        out['st'][s] = [(i, k) for i in ST for k in D if X_[v[i, k, h]] > 0.5]
        out['fuel'][s] = [(f, k) for f in fc for k in D if X_[y[f, k, h]] > 0.5]
        out['A'][s] = [X_[A[t, s]] for t in T]
        out['path'][s] = [[p for p in PATH if X_[q[p, t, s]] > 0.5][0] for t in T]
    return out

# ---------------------------------------------------------------------------
# 5. Convenience functions used by Code_02 - Code_05
# ---------------------------------------------------------------------------
def deterministic(scens, **kw):
    """Perfect-foresight optimum C*(s) for each scenario (Table 5)."""
    return {s['name']: solve([s], criterion='cost', **kw) for s in scens}

def bau(scens, **kw):
    return {s['name']: solve([s], criterion='cost', allow_eet=False, allow_fuel=False, **kw)['cost'][s['name']] for s in scens}

def four_formulations(scens, **kw):
    """Table 7 / Table 9: static and adaptive, minimax cost and minimax regret."""
    det = deterministic(scens, **kw); cstar = {k: v['cost'][k] for k, v in det.items()}
    rows = {}
    for adaptive in (False, True):
        for crit in ('cost', 'regret'):
            o = solve(scens, adaptive=adaptive, criterion=crit, Cstar=cstar, **kw)
            rows[('Adaptive' if adaptive else 'Static', crit)] = dict(
                obj=o['phi'], regret={k: o['cost'][k] - cstar[k] for k in cstar}, sol=o)
    return cstar, det, rows

def adopts(reg, path, mult, fuelcap=None, **kw):
    sc = scenario(reg, price_mult={path: mult}, pace=kw.pop('pace', 1.0))
    return bool(solve([sc], criterion='cost', fuelcap=fuelcap, **kw)['fuel'][reg])

def threshold(reg, path, lo=0.1, hi=8.0, tol=1e-4, **kw):
    """Price (multiple of HFO) below which conversion to `path` is adopted; bisection to `tol` (Table 8A note)."""
    if not adopts(reg, path, lo, **kw): return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if adopts(reg, path, mid, **kw): lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

if __name__ == '__main__':
    print('Technology data (Table 2):')
    for i, d in TECH.items():
        print(f"  {i:4s} {d['name']:40s} {d['cls']}  eff={d['eff']*100:4.1f}%  CAPEX={d['capex']:7.1f}  OPEX={d['opex']:6.1f}")
    print(f'E0 = {E0:,.0f} GJ, HFO = {HFO_GJ:.2f} USD/GJ')
