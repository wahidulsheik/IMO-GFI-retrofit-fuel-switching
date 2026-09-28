"""
Code_02_Main_Results.py
Reproduces the main-text results: Tables 3, 5, 6, 7, 8A, 8B and 9, the model size and
solve time (Section 4), and the numbers quoted in the text of Sections 4.1-4.3 and 4.6.
Outputs are printed and written to ./results/.
"""
import os
import pandas as pd
from Code_01_Model import *

OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
def save(df, name): df.to_csv(os.path.join(OUT, name), index=False); print(df.to_string(index=False)); print()
def r2(x): return round(x + 0.0, 2)
def r1(x): return round(x + 0.0, 1)

# ---------------- Table 3 ----------------
print('TABLE 3. GFI reduction factors and targets')
save(pd.DataFrame([dict(model_year=t, calendar_year=2027 + t, base_red=BASE_RED[t], dc_red=BASE_RED[t] + DC_GAP,
                        base_target=r2(targets(t, 'S2')[0]), dc_target=r2(targets(t, 'S2')[1])) for t in range(1, 9)]),
     'Table3_GFI_targets.csv')

# ---------------- Table 5 ----------------
print('TABLE 5. Deterministic optimum vs business-as-usual')
det = deterministic(BASE_SCENARIOS); b = bau(BASE_SCENARIOS)
rows = []
for s in ['S1', 'S2', 'S3']:
    o = det[s]
    st_capex = sum(TECH[i]['capex'] for i, k in o['st'][s])
    op_capex = sum(TECH[i]['capex'] for i, t in o['op'])
    rows.append(dict(scenario=s, optimal_TCO=r1(o['cost'][s]), BAU_TCO=r1(b[s]), saving=r1(b[s] - o['cost'][s]),
                     structural_EET_capex=st_capex, operational_capex=op_capex,
                     structural_retrofit=', '.join(f'{i} at Year {k}' for i, k in o['st'][s]) or 'none',
                     operational=', '.join(f'{i}@Y{t}' for i, t in o['op']),
                     max_efficiency_pct=r1(100 * max(o['A'][s]))))
save(pd.DataFrame(rows), 'Table5_deterministic.csv')

# ---------------- Table 6 ----------------
print('TABLE 6. Two-tier deficit charge under S2 (USD/GJ)')
save(pd.DataFrame([dict(pathway=PATH_LABEL[p], **{f'Year {t}': r2(deficit_charge(PATH[p][0], t, 'S2')) for t in [1, 3, 5, 8, 10]})
                   for p in PATH]), 'Table6_deficit_charge.csv')

# ---------------- Table 7 ----------------
print('TABLE 7. Worst-case regret by criterion and recourse structure (kUSD)')
cstar, _, t7 = four_formulations(BASE_SCENARIOS)
rows = []
for (struct, crit), r in t7.items():
    rows.append(dict(formulation=f'{struct}, minimax {crit}', objective=r1(r['obj']),
                     **{f'regret_{s}': r1(r['regret'][s]) for s in cstar}, worst_case=r1(max(r['regret'].values())),
                     operational=' '.join(f'{i}@Y{t}' for i, t in r['sol']['op']),
                     structural='; '.join(f"{s}:{','.join(f'{i}@Y{k}' for i, k in r['sol']['st'][s]) or '-'}" for s in cstar)))
save(pd.DataFrame(rows), 'Table7_regret.csv')
w = {k: max(r['regret'].values()) for k, r in t7.items()}
print('Reductions (Section 4.2): criterion static %.0f%%, criterion adaptive %.0f%%, recourse cost %.0f%%, recourse regret %.0f%%' % (
    100 * (1 - w[('Static', 'regret')] / w[('Static', 'cost')]), 100 * (1 - w[('Adaptive', 'regret')] / w[('Adaptive', 'cost')]),
    100 * (1 - w[('Adaptive', 'cost')] / w[('Static', 'cost')]), 100 * (1 - w[('Adaptive', 'regret')] / w[('Static', 'regret')])))
print('Static minimax-cost regret as share of S1 optimum: %.2f%%' % (100 * w[('Static', 'cost')] / cstar['S1']))
sz = t7[('Adaptive', 'regret')]['sol']
print(f"Model size, adaptive minimax regret: {sz['size'][0]} variables, {sz['size'][1]} constraints; solve time {sz['seconds']:.2f} s\n")

# ---------------- Table 8A ----------------
print('TABLE 8A. Adoption thresholds (multiple of HFO energy-equivalent price)')
rows = []
for p in ['LNG', 'MeOH_f', 'MeOH_r', 'NH3_f', 'NH3_r']:
    th = {reg: threshold(reg, p) for reg in ['S1', 'S2', 'S3']}
    fossil = p in ('MeOH_f', 'NH3_f')
    ref = th['S1'] if fossil else th['S2']
    rows.append(dict(pathway=PATH_LABEL[p], wtw=PATH[p][0], price_at_present=PATH[p][1],
                     threshold=r2(ref) , basis='S1 (dagger)' if fossil else 'S2',
                     required_change_pct=round(100 * (round(ref, 2) / PATH[p][1] - 1)),
                     th_S1=r2(th['S1']), th_S2=r2(th['S2']), th_S3=r2(th['S3'])))
save(pd.DataFrame(rows), 'Table8A_thresholds.csv')

# ---------------- Table 8B ----------------
print('TABLE 8B. Conversion capital sweep, renewable ammonia (deterministic S2 and S3)')
rows = []
for cap in [14000, 10000, 6000, 3000, 1000]:
    a = [reg for reg in ['S2', 'S3'] if adopts(reg, 'NH3_r', 2.33, fuelcap={'NH3': cap})]
    bb = [reg for reg in ['S2', 'S3'] if adopts(reg, 'NH3_r', 1.26, fuelcap={'NH3': cap})]
    rows.append(dict(capex=cap, at_2_33=('Ammonia ' + '/'.join(a)) if a else 'declined', at_1_26=('Ammonia, Year 3 (' + ', '.join(bb) + ')') if bb else 'declined'))
save(pd.DataFrame(rows), 'Table8B_capital.csv')

# ---------------- Section 4.3 text: LNG tests and capital vs threshold ----------------
print('Section 4.3: LNG at 1.40x with lower intensity (S2, S3):')
for gg in [85, 70, 50]:
    print('  intensity', gg, [bool(solve([s], criterion='cost', gwtw={'LNG': gg})['fuel'][s['name']]) for s in BASE_SCENARIOS[1:]])
print('  LNG at 0.90x, intensity 85:', {s: solve([scenario(s, price_mult={'LNG': 0.90})], criterion='cost')['fuel'][s] for s in ['S2', 'S3']})
print('Section 4.3: renewable ammonia threshold (S2) by conversion capital:',
      {cap: r2(threshold('S2', 'NH3_r', lo=1.0, hi=2.6, fuelcap={'NH3': cap})) for cap in [14000, 10000, 6000, 1000, 1]})
print()

# ---------------- Table 9 ----------------
print('TABLE 9. Extended nine-scenario set (kUSD)')
rows = []
for observe in ['full', 'reg']:
    sc9 = nine_scenarios(observe)
    c9, d9, t9 = four_formulations(sc9)
    for (struct, crit), r in t9.items():
        if struct == 'Static' and observe == 'reg': continue
        worst = max(r['regret'], key=r['regret'].get)
        rows.append(dict(formulation=f'{struct}, minimax {crit}', responds_to='-' if struct == 'Static' else ('full scenario' if observe == 'full' else 'regulatory state only'),
                         objective=r1(r['obj']), worst_case=r1(r['regret'][worst]), scenario=worst,
                         conversion='none' if not any(r['sol']['fuel'].values()) else 'yes'))
    if observe == 'full':
        c9_full, d9_full, t9_full = c9, d9, t9
save(pd.DataFrame(rows), 'Table9_nine_scenarios.csv')
print('Nine-scenario optima range: %.1f - %.1f' % (min(c9_full.values()), max(c9_full.values())))
