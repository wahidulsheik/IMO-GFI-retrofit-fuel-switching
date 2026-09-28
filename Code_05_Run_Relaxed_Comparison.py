"""
Code_05_Run_Relaxed_Comparison.py
Reproduces Appendix D (robustness to the dry-docking timing restriction) and the numbers in Section 4.5.
Relaxed variant: structural installation and fuel conversion allowed in every year. Off-hire (Eq. D.1):
  structural EET: 30 days inside a survey year (3, 8), 55 days outside;
  fuel conversion: 45 days inside, 70 days outside  (outside = 25-day survey-equivalent baseline + extension).
In the penalty sweep (Table D.2) the conversion off-hire outside a window is kept 15 days above the
structural value, as in Eq. (D.1).
"""
import os
import pandas as pd
from Code_01_Model import *

OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
def save(df, name): df.to_csv(os.path.join(OUT, name), index=False); print(df.to_string(index=False)); print()
def r1(x): return round(x + 0.0, 1)
REL = (30, 55, 45, 70)

# Table D.1
print('TABLE D.1. Locked window vs relaxed timing')
cL, dL, tL = four_formulations(BASE_SCENARIOS)
cR, dR, tR = four_formulations(BASE_SCENARIOS, relaxed=REL)
rows = [dict(quantity=f'{s} deterministic TCO', locked=r1(cL[s]), relaxed=r1(cR[s]), change=r1(cR[s] - cL[s])) for s in cL]
for k in tL:
    a, b = max(tL[k]['regret'].values()), max(tR[k]['regret'].values())
    rows.append(dict(quantity=f'{k[0]}, minimax {k[1]} - worst-case regret', locked=r1(a), relaxed=r1(b), change=r1(b - a)))
tim = lambda d: ' / '.join(','.join(f'Year {k}' for i, k in d[s]['st'][s]) or '-' for s in ['S1', 'S2', 'S3'])
rows.append(dict(quantity='Structural retrofit timing (S1 / S2 / S3)', locked=tim(dL), relaxed=tim(dR), change='none' if tim(dL) == tim(dR) else 'changed'))
rows.append(dict(quantity='Fuel conversion at prevailing prices', locked='declined' if not any(dL[s]['fuel'][s] for s in dL) else 'adopted',
                 relaxed='declined' if not any(dR[s]['fuel'][s] for s in dR) else 'adopted', change=''))
szL = tL[('Adaptive', 'regret')]['sol']['size']; szR = tR[('Adaptive', 'regret')]['sol']['size']
rows.append(dict(quantity='Model size (variables, constraints), adaptive minimax regret', locked=f'{szL[0]}, {szL[1]}', relaxed=f'{szR[0]}, {szR[1]}', change=''))
save(pd.DataFrame(rows), 'TableD1_locked_vs_relaxed.csv')

# Table D.2
print('TABLE D.2. Sensitivity to the off-window off-hire penalty')
rows = []
for days in [55, 50, 45, 42, 40, 38, 36, 35, 30, 25, 20, 10, 0]:
    rel = (30, days, 45, days + 15)
    c, d, t = None, deterministic(BASE_SCENARIOS, relaxed=rel), None
    cs = {k: v['cost'][k] for k, v in d.items()}
    o = solve(BASE_SCENARIOS, adaptive=True, criterion='regret', Cstar=cs, relaxed=rel)
    rows.append(dict(off_window_days=days, S2_TCO=r1(cs['S2']), worst_case_regret=r1(o['phi']),
                     timing=' / '.join(','.join(f'Year {k}' for i, k in d[s]['st'][s]) or '-' for s in ['S1', 'S2', 'S3'])))
save(pd.DataFrame(rows), 'TableD2_penalty_sweep.csv')

# Break-even off-window penalty for the Year 3 -> Year 1 move of coatings (S2)
lo, hi = 38.0, 40.0
while hi - lo > 0.01:
    mid = 0.5 * (lo + hi)
    y = deterministic([scenario('S2')], relaxed=(30, mid, 45, mid + 15))['S2']['st']['S2'][0][1]
    lo, hi = (mid, hi) if y == 1 else (lo, mid)
print('Break-even off-window penalty (S2): %.1f days' % (0.5 * (lo + hi)))
print('TCO range over the sweep (S2): %.2f%%' % (100 * (rows[0]['S2_TCO'] - rows[-1]['S2_TCO']) / rows[0]['S2_TCO']))

# Section 4.5: value of free timing when renewable ammonia is at its threshold price (1.26x)
for s in ['S2', 'S3']:
    sc = scenario(s, price_mult={'NH3_r': 1.26})
    a = solve([sc], criterion='cost'); b = solve([sc], criterion='cost', relaxed=REL)
    print(f'{s} at 1.26x: locked {a["cost"][s]:.1f} {a["fuel"][s]}, relaxed {b["cost"][s]:.1f} {b["fuel"][s]}, '
          f'saving {a["cost"][s]-b["cost"][s]:.1f} kUSD ({100*(a["cost"][s]-b["cost"][s])/a["cost"][s]:.2f}%)')
