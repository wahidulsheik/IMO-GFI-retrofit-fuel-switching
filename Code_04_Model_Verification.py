"""
Code_04_Model_Verification.py
Reproduces the verification checks of Appendix C.
  C.1  Exact linearisation: e(p,t,s) equals A(t,s) * q(p,t,s) in every reported solution.
  C.2  Check 1: total cost of ownership recomputed from the decisions by an independent routine
                that does not use the objective-function coefficients.
       Check 2: deterministic optima cross-checked against the branches of the multi-scenario model.
       Check 3: static minimax-cost objective equals the S3 deterministic optimum (degeneracy, Section 2.2).
       Check 4: two-step regrets equal regrets from re-solving each scenario alone with the shared
                decisions fixed.
"""
from Code_01_Model import *

def independent_tco(sol, sc):
    """Recompute C(s) from the decisions only (Eq. 3 written out directly)."""
    s = sc['name']; tot = 0.0
    op_year = {i: t for i, t in sol['op']}
    st_year = {i: k for i, k in sol['st'][s]}
    conv = sol['fuel'][s]
    for t in T:
        capex = sum(TECH[i]['capex'] for i, ty in op_year.items() if ty == t)
        capex += sum(TECH[i]['capex'] + OFF_DAYS_ST * TCE for i, k in st_year.items() if k == t)
        capex += sum(FUELCAP[f] + OFF_DAYS_F * TCE for f, k in conv if k == t)
        opex = sum(TECH[i]['opex'] for i, ty in op_year.items() if ty <= t) + sum(TECH[i]['opex'] for i, k in st_year.items() if k <= t)
        eff = min(ABAR, sum(TECH[i]['eff'] for i, ty in op_year.items() if ty <= t) + sum(TECH[i]['eff'] for i, k in st_year.items() if k <= t))
        p = sol['path'][s][t - 1]
        energy = E0 * (1 - eff) * (sc['price'][p] + deficit_charge(PATH[p][0], t, sc['reg'])) / 1000.0
        tot += (capex + opex + energy) / (1 + R) ** t
    return tot

sc = BASE_SCENARIOS
cstar, det, t7 = four_formulations(sc)
sols = [det[s] for s in det] + [r['sol'] for r in t7.values()]

# C.1 linearisation
worst = 0.0
for o in sols:
    X, ix = o['x'], o['index']
    for (p, t, s), j in ix['e'].items():
        worst = max(worst, abs(X[j] - X[ix['A'][t, s]] * X[ix['q'][p, t, s]]))
print(f'C.1  largest |e - A*q| over all reported solutions: {worst:.2e}')

# Check 1
dev = 0.0
for o in sols:
    for s_ in sc:
        if s_['name'] in o['cost']:
            dev = max(dev, abs(independent_tco(o, s_) - o['cost'][s_['name']]))
print(f'Check 1  largest difference between model TCO and independent recomputation: {dev:.2e} kUSD')

# Check 2
dev = 0.0
for s_ in sc:
    s = s_['name']
    fix = {(i, t): (1 if (i, t) in det[s]['op'] else 0) for i in OP for t in T}
    o = solve(sc, adaptive=True, criterion='regret', Cstar=cstar, fix_op=fix)
    # with the scenario's own operational plan, branch s of the multi-scenario model must reach C*(s)
    dev = max(dev, abs(o['cost'][s] - cstar[s]))
print(f'Check 2  largest difference between standalone optimum and the corresponding multi-scenario branch: {dev:.2e} kUSD')

# Check 3
print('Check 3  static minimax-cost objective = %.1f; S3 deterministic optimum = %.1f' % (t7[('Static', 'cost')]['obj'], cstar['S3']))

# Check 4
dev = 0.0
for (struct, crit), r in t7.items():
    o = r['sol']
    fix = {(i, t): (1 if (i, t) in o['op'] else 0) for i in OP for t in T}
    for s_ in sc:
        s = s_['name']
        fs = None
        if struct == 'Static':
            fs = {(i, k, s): (1 if (i, k) in o['st'][s] else 0) for i in ST for k in DOCKS}
        b = solve([s_], criterion='cost', fix_op=fix, fix_st=fs)
        dev = max(dev, abs((b['cost'][s] - cstar[s]) - r['regret'][s]))
print(f'Check 4  largest difference between two-step regret and re-solved regret: {dev:.2e} kUSD')
