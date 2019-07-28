# pinn_wind_bearing
Wind Turbine Main Bearing Fatigue Life Estimation with Physics-informed Neural Networks

This repository is provided for replication of results presented in the article: Y.A. Yucesan and F.A.C. Viana, "Wind turbine main bearing fatigue life estimation with physics-informed neural networks," Proceedings of the Annual Conference of the PHM Society 2019

This repository includes two versions of the code, named as: basic and advanced.

Basic code has two main files:
run01 file trains the RNN using pretrained MLP model with fixed initial weights.
run02 file predicts the fatigue damage accumulation of the wind turbine main bearing for 6 months.

Advanced code has four main files:
run01 file generates a random plane approximation for MLP training.
run02 file trains the MLP with randomly generated initial weights.
run03 file trains the RNN using trained MLP model.
run04 file predicts the fatigue damage accumulation of the wind turbine main bearing for 30 years.

Please refer to following source for required data:
Yucesan, Yigit, 2019, "Wind Turbine Main Bearing Fatigue Life Prediction with PINN", https://doi.org/10.7910/DVN/ENNXLZ, Harvard Dataverse, V1, UNF:6:o3b2Pkuz0uIgkQ57jEKGOA== [fileUNF]

Download the data and extract folders to the directory where this repository is cloned.
