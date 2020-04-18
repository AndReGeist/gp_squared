# Copyright 2019 Max Planck Society. All rights reserved.

import numpy as np
import GPy
import pickle
import datetime
import os

from global_functions import *
from subclass_AMGP_normalized import *
from subclass_ODE_mass_on_surface import *
from subclass_ODE_unicycle import *
from subclass_ODE_duffling_oscillator import *
from plot_scripts import *

##################################################################################
#flag_ode = 'mass_surface'
# flag_ode = 'unicycle'
flag_ode = 'duffling_oscillator'

flag_control = False
flag_savedata = False
flag = [8]  # 1, 3, 4, 8 ONE MODEL AT THE TIME
number_runs = 10
# list_datapoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
list_datapoints = [10, 30, 50, 70, 90, 110, 130]
# list_datapoints = [10, 30, 50]
folder_path = 'data_l4dc/optim' + '2_' + flag_ode + '/'
# folder_path = 'data_l4dc/optim' + '1_' + flag_ode + '/'

l_min = 0.5
l_max = 20
sig_var_min = 0.5
sig_var_max = 10

# Load Field and data scaling used for optimization of hyperparameters
file_path = folder_path + '0/'
with open(file_path + 'result', 'rb') as f:  # Python 3: open(..., 'rb')
    optim_bounds, theta_param, res_Indep, res_AMGP, Field = pickle.load(f)
noise_std = Field.field_param['noise_std']
assert type(Field.X_train) != None and Field.field_param['ode_flag'] == flag_ode

if flag_ode == 'unicycle' and flag_control == True:
    Field.field_param['lim_min'] = [0.3, 0.3, 0.4 * np.pi, -0.4, -0.7, -0.2]
    Field.field_param['lim_max'] = [1.7, 1.7, 1.6 * np.pi, 0.4, 0.7, 0.2]
    Field.field_param['lim_num'] = [5, 5, 5, 3, 3, 3]
if flag_ode == 'mass_surface' and flag_control == True:
    Field.field_param['lim_min'] = [-1.6, -1.6, -1.6, -0.3, -0.3, -0.3, -2, -2, -2]
    Field.field_param['lim_max'] = [1.6, 1.6, 1.6, 0.3, 0.3, 0.3, 2, 2, 2]
    Field.field_param['lim_num'] = [3, 3, 1, 3, 3, 1, 2, 2, 2]
if flag_ode == 'duffling_oscillator' and flag_control == False:
    Field.field_param['lim_min'] = [-3, -3, -4, -4, 0]  # x1, x2, x1_dot, x2_dot, t
    Field.field_param['lim_max'] = [3, 3, 4, 4, 4]
    Field.field_param['lim_num'] = [5, 1, 5, 5, 3]
elif flag_control == True:
    sys.exit('Control inputs are not available for the closed-loop Duffling oscillator. Set "flag_control=False".')

Field.compute_prediction_points()

# Initialize arrays for computing simulation results
array_RMSE = np.zeros((len(list_datapoints), number_runs))
array_Rsquared = np.copy(array_RMSE)
array_constrainterror_sum = np.copy(array_RMSE)
array_constrainterror_mean = np.copy(array_RMSE)
array_constrainterror_max = np.copy(array_RMSE)

# Printout simulation settings
print('flag_ode: ', flag_ode, '\n')
print('flag: ', flag, '\n')
print('list_datapoints: ', list_datapoints, '\n')
print('number_runs: ', number_runs, '\n')
print('noise_std: ', noise_std, '\n')
print('flag_savedata: ', flag_savedata, '\n')

################################
# Initialize models and covariance for prediction grid (theta is fixed)
################################
if any(elem in flag for elem in [8]):
    file_path81 = file_path + 'res_AMGPx.npy'
    res_AMGPx = np.load(file_path81)
    Gp_AMGP = subclass_AMGP_normalized(Field)
    print('K_XX_AMGP start')
    # K_XX_AMGP = Gp_AMGP.covariance_matrix(Field.X_predict, Field.X_predict, res_AMGPx, Gp_AMGP.covariance)
    print('K_XX_AMGP done')

################################
# Make predictions for different number of trainign points
################################
for pp in range(len(flag)):
    flag_tmp = [flag[pp]]
    rank_W = Field.field_param['dim_out']
    K = GPy.kern.RBF(input_dim=Field.field_param['dim_in'], ARD=True)
    K.constrain_positive('.*rbf_variance')
    K.variance.constrain_bounded(sig_var_min, sig_var_max)
    K.lengthscale.constrain_bounded(l_min, l_max)

    for ii in range(len(list_datapoints)):  # Ietrate over increasing number of observations
        for jj in range(number_runs):  # Iterate over random sampled observations
            print('#########################################')
            print('DATA_PRED' + 'Numb. points: ' + str(list_datapoints[ii]) + ' Run: ' + str(jj))
            print('#########################################')
            Field.field_param['number_observations'] = list_datapoints[ii]
            Field.compute_training_points(list_datapoints[ii], noise_std, observation_flag='random')

            # Reformat data for GPy multioutput models
            Xt = [Field.X_train.T[:, [i]] for i in range(Field.field_param['dim_in'])]
            Yt = [Field.Y_train_noisy_tmp.T[:, [i]] for i in range(Field.field_param['dim_out'])]
            Xt_multi = [Field.X_train.T for i in range(len(Yt))]

            if any(elem in flag_tmp for elem in [1]):
                ################################
                # Standard GPy ARD-GP
                ################################
                list_mu1 = []
                list_std1 = []
                for i in range(Field.field_param['dim_out']):
                    m1 = GPy.models.GPRegression(Field.X_train.T, Field.Y_train_noisy_tmp.T[:, [i]], K.copy())

                    # Load model parameters
                    m1.update_model(False)
                    m1.initialize_parameter()
                    file_path1 = file_path + 'm1' + str(i) + '_params.npy'
                    m1[:] = np.load(file_path1)
                    m1.update_model(True)

                    # Make predictions
                    mu1, var1 = m1.predict(Field.X_predict.T)
                    list_mu1.append(mu1)
                    list_std1.append(np.sqrt(var1))

                list_dict = {'list_mu': list_mu1, 'list_std': list_std1, 'gp_type': 'ARD'}

            if any(elem in flag_tmp for elem in [2]):
                ###############################
                # Independent GPs using ICM model in GPy
                ###############################
                file_path2 = file_path + 'm2' + '_params.npy'
                ICM_indep = GPy.util.multioutput.ICM(input_dim=Field.field_param['dim_in'],
                                                     num_outputs=Field.field_param['dim_out'], kernel=K.copy())
                m2 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=ICM_indep)
                m2.update_model(False)
                m2.initialize_parameter()
                m2[:] = np.load(file_path2)
                m2.update_model(True)

                # Extend data to tell model to which output data belongs
                list_mu2 = []
                list_std2 = []
                for i in range(Field.field_param['dim_out']):
                    newX2 = np.hstack([Field.X_predict.T, i * np.ones_like(Field.X_predict.T[:, [0]])])
                    noise_dict2 = {'output_index': newX2[:, -1].astype(int)}
                    mu2, var2 = m2.predict(newX2, Y_metadata=noise_dict2)
                    list_mu2.append(mu2)
                    list_std2.append(np.sqrt(var2))

                list_dict = {'list_mu': list_mu2, 'list_std': list_std2, 'gp_type': 'ARD (Coreg)'}

            if any(elem in flag_tmp for elem in [3]):
                ###############################
                # ICM model
                ###############################
                file_path3 = file_path + 'm3' + '_params.npy'
                icm = GPy.util.multioutput.ICM(input_dim=Field.field_param['dim_in'],
                                               num_outputs=Field.field_param['dim_out'],
                                               W_rank=rank_W, kernel=K.copy())
                m3 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=icm)
                m3.update_model(False)
                m3.initialize_parameter()
                m3[:] = np.load(file_path3)
                m3.update_model(True)

                # Extend data to tell model to which output data belongs
                list_mu3 = []
                list_std3 = []
                for i in range(Field.field_param['dim_out']):
                    newX3 = np.hstack([Field.X_predict.T, i * np.ones_like(Field.X_predict.T[:, [0]])])
                    noise_dict3 = {'output_index': newX3[:, -1].astype(int)}
                    mu3, var3 = m3.predict(newX3, Y_metadata=noise_dict3)
                    list_mu3.append(mu3)
                    list_std3.append(np.sqrt(var3))

                list_dict = {'list_mu': list_mu3, 'list_std': list_std3, 'gp_type': 'ICM'}

            if any(elem in flag_tmp for elem in [4]):
                ################################
                # LCM model
                ################################
                file_path4 = file_path + 'm4' + '_params.npy'

                K1 = GPy.kern.Bias(Field.field_param['dim_in'])
                K2 = GPy.kern.Linear(Field.field_param['dim_in'])
                elcm = GPy.util.multioutput.LCM(input_dim=Field.field_param['dim_in'],
                                                num_outputs=Field.field_param['dim_out'],
                                                W_rank=rank_W, kernels_list=[K1, K2, K.copy()])
                m4 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=elcm)
                m4.update_model(False)
                m4.initialize_parameter()
                m4[:] = np.load(file_path4)
                m4.update_model(True)

                # Extend data to tell model to which output data belongs
                list_mu4 = []
                list_std4 = []
                for i in range(Field.field_param['dim_out']):
                    newX4 = np.hstack([Field.X_predict.T, i * np.ones_like(Field.X_predict.T[:, [0]])])
                    noise_dict4 = {'output_index': newX4[:, -1].astype(int)}
                    mu4, var4 = m4.predict(newX4, Y_metadata=noise_dict4)
                    list_mu4.append(mu4)
                    list_std4.append(np.sqrt(var4))

                list_dict = {'list_mu': list_mu4, 'list_std': list_std4, 'gp_type': 'LCM'}

            if any(elem in flag_tmp for elem in [8]):
                Gp_AMGP.init_update_point({'theta': res_AMGPx,  # res_AMGP_2.x
                                           'covariance_TT': Gp_AMGP.covariance,
                                           'Field': Field})
                list_dict = Gp_AMGP.compute_prediction_for_dataset(Field, res_AMGPx,  # res_AMGP_2.x
                                                                   Gp_AMGP.covariance,
                                                                   Gp_AMGP.covariance,
                                                                   Gp_AMGP.mean)
                # list_dict = Gp_AMGP.update_data_faster(Field.X_predict, Field.X_train, Field.Y_train_noisy,
                #                                     res_AMGP.x, Gp_AMGP.covariance,
                #                                     Gp_AMGP.covariance, Gp_AMGP.covariance, Gp_AMGP.mean, Field, K_XX_AMGP)

            # Compute model metrices for run ii,jj
            for j in range(Field.field_param['dim_out']):
                array_RMSE[ii, jj] = array_RMSE[ii, jj] + (1 / Field.field_param['dim_out']) * rmse(
                    list_dict['list_mu'][j].flatten(), Field.Y_field_ordered[j, :].flatten())
                array_Rsquared[ii, jj] = array_Rsquared[ii, jj] + (1 / Field.field_param['dim_out']) * r_squared(
                    list_dict['list_mu'][j].flatten(),
                    Field.Y_field_ordered[j, :].flatten())
            array_constrainterror_sum[ii, jj] = np.sum(
                compute_constraint_error(Field, list_dict, Field.field_param['params']))
            array_constrainterror_mean[ii, jj] = np.mean(
                compute_constraint_error(Field, list_dict, Field.field_param['params']))
            array_constrainterror_max[ii, jj] = np.max(
                compute_constraint_error(Field, list_dict, Field.field_param['params']))

    for ii in range(len(list_datapoints)):  # Ietrate over increasing number of observations
        print('array_RMSE: ', np.mean(array_RMSE[ii, :]))
        print('array_constrainterror_sum: ', np.mean(array_constrainterror_sum[ii, :]))

    print('array_RMSE\n', array_RMSE)
    print('array_Rsquared\n', array_Rsquared)
    print('array_constrainterror_sum\n', array_constrainterror_sum)
    print('array_constrainterror_mean\n', array_constrainterror_mean)
    print('array_constrainterror_max\n', array_constrainterror_max)

    ################################
    # Save data
    ################################
    if flag_savedata == True:
        file_path_predictions = file_path + 'predictions_' + str(flag[pp]) + '/'
        if os.path.isdir(file_path_predictions):
            pass
        else:
            os.mkdir(file_path_predictions)
        np.save(file_path_predictions + 'array_mean_RMSE.npy', array_RMSE)
        np.save(file_path_predictions + 'array_Rsquared.npy', array_Rsquared)
        np.save(file_path_predictions + 'array_constraint_error_sum.npy', array_constrainterror_sum)
        np.save(file_path_predictions + 'array_constraint_error_mean.npy', array_constrainterror_mean)
        np.save(file_path_predictions + 'array_constraint_error_max.npy', array_constrainterror_max)
        np.save(file_path_predictions + 'list_datapoints.npy', np.array(list_datapoints))

# plt.figure()
# plt.plot(dict_Indep_dataeff['RMSE_constraint'].mean(axis=1))
# plt.plot(dict_AMGP_dataeff['RMSE_constraint'].mean(axis=1))
# plt.show()
