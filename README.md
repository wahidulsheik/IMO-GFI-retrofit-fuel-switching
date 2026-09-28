# Retrofit and fuel-switching decisions for mid-life bulk carriers under the IMO Net-Zero Framework

Code and data accompanying a research article on how the IMO Net-Zero Framework's two-tier
greenhouse gas fuel intensity (GFI) charge shapes energy-efficiency retrofit and fuel-switching
decisions for a mid-life Capesize bulk carrier.

The repository contains a two-stage mixed-integer linear programming (MILP) model that:

* represents the two-tier GFI charge agreed at MEPC 83 in its piecewise form;
* jointly optimises the adoption of 12 energy-efficiency technologies and the choice among 6 fuel pathways
  (heavy fuel oil, LNG, fossil and renewable methanol, fossil and renewable ammonia) over a ten-year horizon;
* evaluates decisions under minimax-cost and minimax-regret criteria, with static and adaptive
  (dry-dock recourse) structures, across regulatory and fuel-price scenarios.

Running the scripts reproduces every table, figure and appendix result reported in the article.

## Requirements
Python 3.9 or later with `numpy`, `pandas`, `scipy` (1.9 or later; the MILP is solved with HiGHS through
`scipy.optimize.milp`) and `matplotlib`. No commercial solver is needed.

    pip install -r requirements.txt

## Repository contents
| File | Purpose |
|---|---|
| `EETs 12.csv` | Technology dataset: 12 energy-efficiency technologies with ranges for efficiency gain, CAPEX and OPEX (source: DNV, *Energy-efficiency measures and technologies*, 2025). |
| `Code_01_Model.py` | Core model: data and parameters, the two-tier GFI charge, the MILP formulation, the two-step solution procedure, and options used in the robustness analyses (residual value, multiplicative efficiency, relaxed dry-dock timing, fuel-price states). |
| `Code_02_Main_Results.py` | Main-text tables (GFI targets, deterministic optima, deficit charges, regret by formulation, fuel-conversion thresholds, conversion-capital sweep, nine-scenario results), model size and solve time. |
| `Code_03_Reproduce_All_Results.py` | Appendix B: full sensitivity results. |
| `Code_04_Model_Verification.py` | Appendix C: model verification checks. |
| `Code_05_Run_Relaxed_Comparison.py` | Appendix D: robustness to the dry-docking timing restriction. |
| `Code_06_Figures.py` | All figures (reads the CSV files written by Code_02 and Code_03). |
| `results/` | Output tables (CSV) produced by the scripts, included for reference. |
| `figures/` | Output figures (PNG) produced by Code_06. |

## How to run
From the repository folder, run the scripts in order:

    python Code_02_Main_Results.py
    python Code_03_Reproduce_All_Results.py
    python Code_04_Model_Verification.py
    python Code_05_Run_Relaxed_Comparison.py
    python Code_06_Figures.py

Tables are printed and written to `results/`; figures are written to `figures/`.
Typical run times on a standard desktop: Code_02 about 1 minute, Code_03 about 2 minutes, the others under 30 seconds.

## Notes
* Money values are in thousand USD (kUSD), discounted at 8 per cent.
* Regret figures use a two-step procedure: the minimax value is found first, and the sum of scenario costs
  is then minimised with that value held fixed, so that every scenario uses its best response to the shared decisions.
* Fuel-conversion thresholds are found by bisection to a tolerance of 0.0001 times the heavy-fuel-oil price.
* Figures use Times New Roman if installed; otherwise TeX Gyre Termes, a Times-compatible typeface.

## Citation
If you use this code, please cite the associated article (details will be added on publication).

## Licence
Code: MIT Licence (see `LICENSE`). The technology data are derived from publicly available DNV figures;
please cite the original source when reusing them.
