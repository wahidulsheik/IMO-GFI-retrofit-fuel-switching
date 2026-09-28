"""
Code_03_Reproduce_All_Results.py
Reproduces every table in Appendix B (Full sensitivity results), Tables B.1-B.8,
which support Sections 4.3, 4.4 and 4.6 of the manuscript. Outputs go to ./results/.
"""
import os
import pandas as pd
from Code_01_Model import *

OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
def save(df, name): df.to_csv(os.path.join(OUT, name), index=False); print(df.to_string(index=False)); print()
def r1(x): return round(x + 0.0, 1)
def r2(x): return None if x is None else round(x + 0.0, 2)

def adaptive_regret(price_mult=None, fuelcap=None):
    sc = [scenario(r, price_mult=price_mult) for r in ['S1', 'S2', 'S3']]
    det = deterministic(sc, fuelcap=fuelcap); cstar = {k: v['cost'][k] for k, v in det.items()}
    o = solve(sc, adaptive=True, criterion='regret', Cstar=cstar, fuelcap=fuelcap)
    return o['phi'], o['fuel']

def label(fuel, name):
    if not fuel: return 'none'
    f, k = fuel[0]
    return name if k == 3 else f'{name} (Year {k})'

# B.1 Renewable ammonia price sweep
print('TABLE B.1'); rows = []
for mlt in [2.33, 2.00, 1.75, 1.50, 1.40, 1.30, 1.26, 1.20, 1.10, 1.00]:
    phi, fu = adaptive_regret({'NH3_r': mlt})
    rows.append(dict(multiple=mlt, price=round(mlt * HFO_GJ, 2), worst_case_regret=max(0.0, r1(phi)),
                     **{s: label(fu[s], 'Ammonia') for s in ['S1', 'S2', 'S3']}))
save(pd.DataFrame(rows), 'TableB1_ammonia_sweep.csv')

# B.2 Renewable methanol price sweep (multiples of HFO)
print('TABLE B.2'); rows = []
for mlt in [6.50, 5.00, 4.00, 3.00, 2.00, 1.60, 1.50, 1.40, 1.37, 1.30]:
    phi, fu = adaptive_regret({'MeOH_r': mlt})
    rows.append(dict(multiple=mlt, price=round(mlt * HFO_GJ, 2), worst_case_regret=max(0.0, r1(phi)),
                     **{s: label(fu[s], 'Methanol') for s in ['S1', 'S2', 'S3']}))
save(pd.DataFrame(rows), 'TableB2_methanol_sweep.csv')
print('Renewable methanol threshold S2 / S3:', r2(threshold('S2', 'MeOH_r')), r2(threshold('S3', 'MeOH_r')), '\n')

# B.3 Conversion capital sweep
print('TABLE B.3'); rows = []
for cap in [14000, 10000, 6000, 3000, 1000]:
    a = [r for r in ['S2', 'S3'] if adopts(r, 'NH3_r', 2.33, fuelcap={'NH3': cap})]
    b = [r for r in ['S2', 'S3'] if adopts(r, 'NH3_r', 1.26, fuelcap={'NH3': cap})]
    rows.append(dict(capex=cap, at_2_33=', '.join(a) or 'none', at_1_26=('Ammonia, Year 3 (' + ', '.join(b) + ')') if b else 'none'))
save(pd.DataFrame(rows), 'TableB3_capital_sweep.csv')

# B.4 Post-2035 tightening pace (Scenario S2)
print('TABLE B.4'); rows = []
for pace in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    o = solve([scenario('S2', pace=pace)], criterion='cost')
    rows.append(dict(pace=pace, TCO=r1(o['cost']['S2']), conversion='none' if not o['fuel']['S2'] else str(o['fuel']['S2'])))
save(pd.DataFrame(rows), 'TableB4_tightening.csv')

# B.5 Threshold by conversion capital
print('TABLE B.5'); rows = []
for cap in [14000, 10000, 6000, 3000, 1000, 0.001]:
    rows.append(dict(capex=0 if cap < 1 else cap, th_S2=r2(threshold('S2', 'NH3_r', lo=1.0, hi=3.0, fuelcap={'NH3': cap})),
                     th_S3=r2(threshold('S3', 'NH3_r', lo=1.0, hi=3.0, fuelcap={'NH3': cap}))))
save(pd.DataFrame(rows), 'TableB5_threshold_by_capital.csv')

# B.6 Multiplicative efficiency
print('TABLE B.6'); rows = []
add_ = deterministic(BASE_SCENARIOS); mul_ = deterministic(BASE_SCENARIOS, multiplicative=True)
for s in ['S1', 'S2', 'S3']:
    a, m_ = add_[s], mul_[s]
    rows.append(dict(scenario=s, TCO_add=r1(a['cost'][s]), TCO_mult=r1(m_['cost'][s]),
                     change_pct=round(100 * (m_['cost'][s] / a['cost'][s] - 1), 2),
                     maxA_add=r1(100 * max(a['A'][s])), maxA_mult=r1(100 * max(m_['A'][s])),
                     plan_add=' '.join(f'{i}@Y{t}' for i, t in a['op']) + ' | ' + ' '.join(f'{i}@Y{k}' for i, k in a['st'][s]),
                     plan_mult=' '.join(f'{i}@Y{t}' for i, t in m_['op']) + ' | ' + ' '.join(f'{i}@Y{k}' for i, k in m_['st'][s])))
save(pd.DataFrame(rows), 'TableB6_multiplicative.csv')
eta = [TECH[i]['eff'] for i, t in add_['S2']['op']] + [TECH[i]['eff'] for i, k in add_['S2']['st']['S2']]
print('S2 additive portfolio evaluated multiplicatively: %.1f%%' % (100 * (1 - np.prod([1 - x for x in eta]))))
cm = {k: v['cost'][k] for k, v in mul_.items()}
om = solve(BASE_SCENARIOS, adaptive=True, criterion='regret', Cstar=cm, multiplicative=True)
print('Adaptive minimax regret, multiplicative: %.1f; operational plan %s\n' % (om['phi'], om['op']))

# B.7 Residual value (Eq. 13, X = 5)
print('TABLE B.7'); X = 5
dr = deterministic(BASE_SCENARIOS, residual=X); crv = {k: v['cost'][k] for k, v in dr.items()}
orv = solve(BASE_SCENARIOS, adaptive=True, criterion='regret', Cstar=crv, residual=X)
rows = [dict(quantity='share credited Year 3 / Year 8', base='0 / 0', with_rv='%.1f%% / %.1f%%' % (100 * X / (10 - 3 + 1 + X), 100 * X / (10 - 8 + 1 + X)))]
for s in ['S1', 'S2', 'S3']:
    rows.append(dict(quantity=f'{s} deterministic TCO', base=r1(add_[s]['cost'][s]), with_rv=r1(crv[s])))
rows.append(dict(quantity='EET portfolio', base=str(add_['S2']['st']['S2']), with_rv=str({s: dr[s]['st'][s] for s in dr})))
rows.append(dict(quantity='adaptive minimax regret', base=50.4, with_rv=r1(orv['phi'])))
rows.append(dict(quantity='conversion at 2.33x, capital 1,000', base='declined',
                 with_rv='declined' if not any(adopts(r, 'NH3_r', 2.33, fuelcap={'NH3': 1000}, residual=X) for r in ['S2', 'S3']) else 'adopted'))
for p in ['NH3_r', 'MeOH_r', 'LNG']:
    rows.append(dict(quantity=f'{p} threshold S2 / S3',
                     base='%.2f / %.2f' % (threshold('S2', p), threshold('S3', p)),
                     with_rv='%.2f / %.2f' % (threshold('S2', p, residual=X), threshold('S3', p, residual=X))))
save(pd.DataFrame(rows), 'TableB7_residual_value.csv')

# B.8 Nine-scenario set
print('TABLE B.8'); sc9 = nine_scenarios('full')
c9, d9, t9 = four_formulations(sc9)
rows = []
for s in c9:
    rows.append(dict(scenario=s, optimal_TCO=r1(c9[s]), structural=' '.join(f'{i}@Y{k}' for i, k in d9[s]['st'][s]) or 'none',
                     static_cost=r1(t9[('Static', 'cost')]['regret'][s]), static_regret=r1(t9[('Static', 'regret')]['regret'][s]),
                     adaptive_cost=r1(t9[('Adaptive', 'cost')]['regret'][s]), adaptive_regret=r1(t9[('Adaptive', 'regret')]['regret'][s])))
save(pd.DataFrame(rows), 'TableB8_nine_scenarios.csv')
for s in ['S2', 'S3']:
    th = threshold(s, 'NH3_r', lo=0.5, hi=4.0) if False else None
F3 = {s: None for s in ['S2', 'S3']}
for s in ['S2', 'S3']:
    lo, hi = 0.5, 4.0
    def ad(mlt): return bool(solve([scenario(s, price_mult={'NH3_r': mlt}, fossil_mult=1.25, renew_mult=1.0)], criterion='cost')['fuel'][s])
    while hi - lo > 0.001:
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if ad(mid) else (lo, mid)
    F3[s] = 0.5 * (lo + hi) / 1.25
print('F3 break-even for renewable ammonia (multiple of F3 HFO price): S2 %.2f, S3 %.2f; F3 actual %.2f' % (F3['S2'], F3['S3'], 2.33 * 0.75 / 1.25))
