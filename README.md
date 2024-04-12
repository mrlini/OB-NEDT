# OB-NEDT

[![DOI](https://zenodo.org/badge/785761210.svg)](https://zenodo.org/doi/10.5281/zenodo.10966259)

Code for "Log-loss boosting optimization with a Nash Equilibrium decision tree"

Use of an AdaBoost model and a log-loss optimization mechanism to improve the performance of an equilibrium-based decision tree. The two-step algorithm builds equilibrium decision trees on weighted data in the first step; during the second step, it determines the contribution of each classifier by optimizing the overall log-loss function.

- `ob_nedt.py` contains the implementation of OB-NEDT and the method `run_ob_nedt()` used to fit and predict.
- `test_ob_nedt.py` and zob_nedt_example.ipynb` show a usage example.
