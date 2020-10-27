# Wind Turbine Main Bearing Fatigue Life Estimation with Physics-informed Neural Networks

This repository is provided for replication of results presented in the article:

*Y. A. Yucesan and F. A. C. Viana, "A physics-informed neural network for wind turbine main bearing fatigue," International Journal of Prognostics and Health Management, Vol. 11 (1), 2020. (ISSN: 2153-2648)*

This repository includes two sets of the code.

**Basic:**
- run01 file trains the RNN using pretrained MLP model with fixed initial weights.
- run02 file predicts the fatigue damage accumulation of the wind turbine main bearing for 6 months.

**Advanced:**
- run01 file generates a random plane approximation for MLP training.
- run02 file trains the MLP with randomly generated initial weights.
- run03 file trains the RNN using trained MLP model.
- run04 file predicts the fatigue damage accumulation of the wind turbine main bearing for 30 years.

In order to run the codes, you will also need to:
1. Install the PINN python package: https://github.com/PML-UCF/pinn
2. Download the data:
Yucesan, Yigit, 2019, "Wind Turbine Main Bearing Fatigue Life Prediction with PINN" (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ENNXLZ), Harvard Dataverse, V1, UNF:6:o3b2Pkuz0uIgkQ57jEKGOA== [fileUNF]
and extract folders inside wind_bearing_dataset to the directory where this repository is cloned.
