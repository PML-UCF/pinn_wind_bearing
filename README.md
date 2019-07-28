<h1>Wind Turbine Main Bearing Fatigue Life Estimation with Physics-informed Neural Networks</h1>

<p>This repository is provided for replication of results presented in the article:</p>
<i>Y.A. Yucesan and F.A.C. Viana, "Wind turbine main bearing fatigue life estimation with physics-informed neural networks," Proceedings of the Annual Conference of the PHM Society 2019</i>

<p>This repository includes two versions of the code, named as: basic and advanced.</p>
Basic code has two main files:
<ul>
  <li>run01 file trains the RNN using pretrained MLP model with fixed initial weights.</li>
  <li>run02 file predicts the fatigue damage accumulation of the wind turbine main bearing for 6 months.</li>
</ul>

Advanced code has four main files:
<ul>
  <li>run01 file generates a random plane approximation for MLP training.</li>
  <li>run02 file trains the MLP with randomly generated initial weights.</li>
  <li>run03 file trains the RNN using trained MLP model.</li>
  <li>run04 file predicts the fatigue damage accumulation of the wind turbine main bearing for 30 years.</li>
</ul>

<p>Please refer to following source for required data:</p>
<i>Yucesan, Yigit, 2019, "Wind Turbine Main Bearing Fatigue Life Prediction with PINN", <a href="url">https://doi.org/10.7910/DVN/ENNXLZ</a>, Harvard Dataverse, V1, UNF:6:o3b2Pkuz0uIgkQ57jEKGOA== [fileUNF]</i>

<p>Download the data and extract folders to the directory where this repository is cloned.</p>
