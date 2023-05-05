# Fairer-Together-Mitigating-Disparate-Exposure-in-Kemeny-Aggregation
Corresponding source code and experimental implementation for "Fairer Together: Mitigating Disparate Exposure in Kemeny Aggregation" paper in FAcct'23.




## EPIK and EPIRA code:

Please see src folder for python scripts for EPIK and EPIRA implementations.
Epik is in `epik.py` and EPIRA is in `epira.py`,

The baseline method PRE-FE (preprocessing) is in the `preprocess_kem.py` script. 

## Comparative baselines:

All baselines used are in the baselines folder. 
 - **KEM (Kemeny)**. Please see `kemeny.py`.
 - **RAPF**. Please see `baseline_weietal.py`.
 - **PFAIR-KEM**. Please see `baseline_cacheletal.py`.
 

## Experiments:
### Mallows datasets.
See the Mallows_Datasets directory.

All experiments utilizing the Mallows datasets are performed in the `run_mallows.py` script, with results written to the `mallows_results.csv` file. This includes running all proposed algorithms, comparative baselines, EPIRA with the Copeland rule without the WiG property, and running EPIRA with different voting rules. Each Mallows preference profile is in its own csv file of the format `R_disp_<disperasion param>_fairp_<reference ranking>_.csv`.
#### Plotting heatmaps.
`heatmaps_mallows.R` produces the heatmaps of the consensus accuracy and exposure ratio from each method. It's figures are written to the Mallows_Datasets\plots directory.  

### Preflib datasets.
See the Preflib_Datasets directory.

All experiments utilizing the preflib datasets are performed in the `exp_preflib.py` script, with results written to `<dataset name>_results.csv` files. This includes running all proposed algorithms, comparative baselines, and running EPIRA with different voting rules.

### CSRankings.
See the CSRankings_Datasets directory.

All experiments utilizing the cs rankings dataset are performed in the `run_csrankings.py` script, with results written to `<dataset name>_results.csv` files. This includes running all proposed algorithms, comparative baselines, and running EPIRA with different voting rules.

### Gamma Parameter.
See the Mallows_Datasets directory.

All experiments utilizing the Mallows datasets are performed in the `gamma_values.py` script, with results written to the `mallows_gamma_results.csv` file. .

#### Helper Functions:
Functions to calculate consensus accuracy and exposure ratio are in the metric directory.
