Copyright 2020 Max Planck Society. All rights reserved.


Source code of the paper "Learning Constrained Dynamics with Gauss' Principle adhering Gaussian Processes",  by
A. René Geist and Sebastian Trimpe. 



#### To start working:
The code runs on Ubuntu 18.04.3 LTS

The required packages can be installed via pipenv: https://github.com/pypa/pipenv
We recommend using pipenv in combination with pyenv: https://github.com/pyenv/pyenv
After installation of pipenv, open a shell (ctrl+alt+t) move to the directory, and execute
```
pipenv install
```
Then activate the environement via
```
pipenv shell
```
List of required packages:
- GPy, version=1.9.9
- numpy, version=1.17.4
- scipy, version=1.3.3
- matplotlib, version=3.1.2
- vtk, version=8.1.2 (optional)
- mayavi, version=4.7.1  (optional)


#### Contents of this repo
**Python scripts**

*global_functions.py :* Contains functions that are frequently used in all scripts, e.g. RMSE compuyation.

*main_multioutput.py :* Optimization of GPy-GP models and our GP2 models. This script is used to compute the GP hyperparameters
used in *Table 1* via L-BFGS-b and maximum likelihood estimation.

*main_predict.py :*  Compute GPy-GP models and our GP^2 model's posterior mean and variance. This script is used to compute 
the GPs' predictions and their RMSE/constraint error depicted in *Table 1*.

*main_comparison.py :* Take precomputed model hyperparameters and compute and save predictions for different randomly 
sampled observations.

*main_ODErollout.py :* This script is used to create *Figure 3c* of the paper. Take trained GP2 model of a "subclass_ODE_mass_on_surface" system and compute RK45 trajectory predictions. 

*main_ODEplots.py :* This script is used to generate Figure 1 to Figure 5 in the supplementary material. That is, generate trajectories of the example systems 1-3.

*main_figure1.py :* (origins from "main_predict.py"") This script is used to create *Figure 3a* of the paper. That is given 
GP hyperparameters ("...params.npy" or "res_AMGPx.npy") and system ODE settings stored in the pickle-object "results" compute
predictions of the systems constrained acceleration in form of the GPs posterior variance. 

*main_figure2.py :* (origins from "main_predict.py"") This script is used to create *Figure 3b (top)* of the paper. That is given 
GP hyperparameters ("...params.npy" or "res_AMGPx.npy") and system ODE settings stored in the pickle-object "results" compute
samples of the prior and posterior GPs.

*main_figure4.py :* (origins from "main_predict.py"") This script is used to create *Figure 3b (bottom)* of the paper. That is given 
GP hyperparameters ("...params.npy" or "res_AMGPx.npy") and system ODE settings stored in the pickle-object "results" compute
 the UNCONSTRAINED system acceleration given observations of the constrained system.

*class_ODE.py :* Contains functions that are required to compute data of the constrained system. Contains function routines
to create (normalized) input data and observations, as well as general system dependent GP optimization settings.

*subclass_ODE_{unicycle, duffling_oscillator, mass_on_surface}.py :* Contains the analytical ODE of the mechanical system 
(in Udwadia-Kalaba-Equation form) as well as the system dependent constraint functions (A,b). Further this class initializes
the specific ODE parameters in the dictionary "field_param" and contains constraint functions required to ensure that computed states
lie inside the constraint state space of the system.

*class_GP.py :* Contains a vanilla implementation of Gp regression, the squared exponential covariance function, as well 
as functions required for the optimization of the GP2 hyperparameters.

*subclass_AMGP_normalized :* (subclass of "class_GP.py") Adjusts the "class_GP" such that the GP2-model is obtained. 
In particular, changes the mean and standard SE covariance function to adhere Gauss principle using the constraining 
functions from the class_ODE object.

*plot_scripts :* Contains scripts for visualization of the system ODE and GP prediction results.

**Simulation data**

*optim1_[...] :* Contains the data used for *Table 1* which is computed using "main_multioutput.py". The directories named
"0" to "9" contain the optimization results (GP hyperparameters and ODE-settings) for each run (taking new 100 random observations). 
The file "info.txt" contains the settings used for optimization. The folders "predictions_[GP-number]..." contain the
GPs prediction results (as numpy arrays) after running the script "main_predict.py" results using the previously mentioned 
hyperparameters. At the end of the folder names "predictions_[GP-number]False/True_False/True", the first False/True is set to True
 if the constraint parameters are also estimated and the second False/True is set to True if a mechanistic parametrix mean function
 has been included as additional prior knowledge. The "info.txt" file in the "predictions_[...]" directory contains a summary of the results and the used 
settings in "main_predict.py" to obtain these numbers.

*optim2_[...] :* Contains the hyperparameter optimization result of the GP2 model and GPy SE model using 200 observations
and running "main_multioutput.py". 

#### General simulation settings
**parameter name = parameter value for [Example1 | Unicycle | Duffing Oscillator] : Description**
- **noise_std=[0.1 | 0.01 | 0.1]**: Standard deviation of measurements
- **l_min = 0.5:** Mimimum value of lengthscale in optimization
- **l_max = 20:** Maximum value of lengthscale in optimization
- **sig_var_min = 0.5:** Maximum value of signal variance in optimization
- **sig_var_max = 10:** Maximum value of signal variance in optimization
- **theta_pts:** Number of optimization restarts (The optimizer keeps the hyperparameter estimate with smallest negative log-likelihood)
- **number_observations:** Number of observations used for training and predictions
-  **dev_con = [0.02 |  0.02 |  0.02}**: (Used if in the GP2 model the system parameters are estimated; flag_estimate_sys_params = True
) Maximum initial deviation from true system parameters in optimization

#### Misc:
GPy documentation :
https://gpy.readthedocs.io/en/deploy/GPy.models.html

Examples on working with GPy :
https://stats.stackexchange.com/questions/198327/sampling-from-gaussian-process-posterior
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb
http://gpss.cc/gpss13/assets/lab1.pdf

Examples on working with multi-output GPs in GPy: 
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/coregionalized_regression_tutorial.ipynb
