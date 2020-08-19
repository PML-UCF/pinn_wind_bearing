# A Hybrid Modeling for Wind Turbine Main Bearing Fatigue with Uncertainty in Grease Observations

This repository is provided for replication of results presented in the article:

*Y. A. Yucesan and F. A. C. Viana, "[A hybrid modeling for wind turbine main bearing fatigue with uncertainty in grease observations]," Proceedings of the Annual Conference of the PHM Society 2020*

This repository includes three python scripts.

- run00 file samples visual grease inspections.
- run01 file trains the RNN using trained MLP model.
- run02 file predicts the fatigue damage accumulation of the wind turbine main bearing for 20 years.

In order to run the codes, you will also need to:
1. Install the PINN python package: https://github.com/PML-UCF/pinn
2. Download the data:
Yucesan, Yigit, 2020, "Replication Data for: A Hybrid Modeling for Wind Turbine Main Bearing Fatigue with Uncertainty in Grease Observations" (https://doi.org/10.7910/DVN/J3IUVR), Harvard Dataverse, V1, UNF:6:QG0Aog7znSpX/zq2A/nYwQ== [fileUNF]
and extract folders inside wind_bearing_dataset_2020 to the directory where this repository is cloned.
