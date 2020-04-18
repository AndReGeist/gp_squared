# Copyright 2019 Max Planck Society. All rights reserved.

import matplotlib.pyplot as plt
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

# SYSTEM-settings
#flag_ode = 'mass_surface'
#flag_ode = 'unicycle'
flag_ode = 'duffling'
flag_control = False

number_observations = 100
flag_estimate_sys_params = False
flag_mean_prior = True

# GP-settings
# 1: Predict constrained acceleration with SE-ARD (GPy)
# 3: Predict constrained acceleration with ICM (GPy)
# 4: Predict constrained acceleration with LMC (GPy)
# 8: Predict constrained acceleration with GP^2 model a.k.a. AMGP
# 9: Predict UN-constrained acceleration with GP^2 model

flag = [8]

# Script-settings
flag_training_points_in_prediction_slice = False
flag_plot_samples = False
flag_savedata = True
folder_path = 'data_l4dc/optim' + '1_2_' + flag_ode + '/'
reruns_script = 10  # Make sure the number is correct, normally 10

array_mean_RMSE = np.zeros((len(flag), reruns_script))
array_Rsquared = np.zeros((len(flag), reruns_script))
array_constraint_error = np.zeros((len(flag), reruns_script))
array_constraint_error_est = np.zeros((len(flag), reruns_script))

print(folder_path)
print('flag_estimate_sys_params: ', flag_estimate_sys_params)
print('flag_mean_prior: ', flag_mean_prior)

################################
# Load ODE TRAINING data
################################
for jj in range(reruns_script):
    print('RUN: ', jj)
    # Define slice for predictions and compute prediction points
    file_path = folder_path + str(jj) + '/'
    with open(file_path + '/result', 'rb') as f:  # Python 3: open(..., 'rb')
        optim_bounds, theta_param, res_Indep, res_AMGP, Field = pickle.load(f)
    assert type(Field.X_train) != None #and Field.field_param['ode_flag'] == flag_ode

    if flag_ode == 'unicycle':
        if flag_control is False:
            Field.field_param['lim_num'] = [100, 1, 1, 1]
            Field.field_param['lim_train_max'] = [0.5, 1, 2 * np.pi, 0.5]
        elif flag_control is True:
            #pass
            #Field.field_param['lim_min'] = [1, 1, 0, 0, 0, 0]  # [0, 0, 0, -0.5, -1, -0.5]
            #Field.field_param['lim_num'] = [1, 1, 100, 1, 1, 1]
            Field.field_param['lim_num'] = [5, 1, 8, 5, 5, 5]
            #Field.field_param['lim_train_min'] = [0, 0, 2.5, -0.5, -1, -0.5]
            #Field.field_param['lim_train_max'] = [1, 1, 2*np.pi, 0.5, 1, 0.5]

    elif flag_ode == 'mass_surface':
        assert type(Field.X_train) != None and Field.field_param['ode_flag'] == flag_ode
        if flag_control is False:
            Field.field_param['lim_num'] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
            Field.field_param['lim_train_max'] = [2, 2, 0.001, 0.5, 0.5, 0.5]
        elif flag_control is True:
            #Field.field_param['lim_min'] = [-2, -2, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]
            #Field.field_param['lim_min'] = [-2, 0, -0.01, 0, 0, 0, 0, 0, 0]
            #Field.field_param['lim_min'] = [0, 0, -0.01, -0.5, 0.5, 0.5, 0, 0, 0]
            #Field.field_param['lim_num'] = [1, 1, 1, 100, 1, 1, 1, 1, 1]
            Field.field_param['lim_num'] = [5, 5, 1, 3, 3, 1, 3, 3, 3]
            #Field.field_param['lim_train_max'] = [2, 2, 0.01, 0.5, 0.5, 0.5, 5, 5, 5]

    elif flag_ode == 'duffling':
        if flag_control is False:
            #pass
            #Field.field_param['lim_num'] = [5, 1, 5, 5, 10]
            #Field.field_param['lim_num'] = [10, 1, 10, 1, 10]
            #Field.field_param['lim_min'] = [-2, 4, -5, 0, 2]
            #Field.field_param['lim_min'] = [-2, 3, -5, 1, 0]
            Field.field_param['lim_num'] = [10, 1, 10, 1, 10]  # x1, x2, x1_dot, x2_dot, t
            #Field.field_param['lim_train_max'] = [4, 4, 5, 5, 1] # [4, 4, 5, 5, 5]
            #Field.field_param['lim_max'] = [4, 4, 5, 5, 10]
        elif flag_control is True:
            sys.exit('Control inputs are not available for the closed-loop Duffling oscillator. Set "flag_control=False".')

    if (flag_training_points_in_prediction_slice == True):
        Field.compute_training_points(number_observations, Field.field_param['noise_std'], observation_flag='random')


    Field.compute_prediction_points()

    list_error1 = []
    list_error1.append(Field.compute_constraint_error(Field.X_predict_unnormalized, Field.Y_field_unnormalized_ordered, Field.field_param['params']))
    print('Max Field error', np.max(list_error1))

    # Reformat data
    Xt = [Field.X_train.T[:,[i]] for i in range(Field.field_param['dim_in'])]
    Yt = [Field.Y_train_noisy_tmp.T[:,[i]] for i in range(Field.field_param['dim_out'])]

    Xt_multi = [Field.X_train.T for i in range(len(Yt))]

    ################################
    # PREPARE SAVING DATA
    ################################
    res_Indep = None
    res_AMGP = None

    ################################
    # GP Prediction with precomputed hyperparameters
    ################################
    list_dict_data = []
    rank_W = Field.field_param['dim_out']
    #rank_W = 2
    K = GPy.kern.RBF(input_dim=Field.field_param['dim_in'], ARD=True)

    if any(elem in flag for elem in [1]):
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

        list_dict1 = {'list_mu': list_mu1, 'list_std': list_std1, 'gp_type': 'ARD'}
        list_dict_data.append(list_dict1)

    if any(elem in flag for elem in [2]):
        ###############################
        # Independent GPs using ICM model in GPy
        ###############################
        file_path2 = file_path + 'm2' + '_params.npy'
        ICM_indep = GPy.util.multioutput.ICM(input_dim=Field.field_param['dim_in'], num_outputs=Field.field_param['dim_out'], kernel=K.copy())
        m2 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=ICM_indep)
        m2.update_model(False)
        m2.initialize_parameter()
        m2[:] = np.load(file_path2)
        m2.update_model(True)
        print('\n############### \nARD (Correg)\n############### \n', m2)

        # Extend data to tell model to which output data belongs
        list_mu2 = []
        list_std2 = []
        for i in range(Field.field_param['dim_out']):
            newX2 = np.hstack([Field.X_predict.T, i*np.ones_like(Field.X_predict.T[:,[0]])])
            noise_dict2 = {'output_index': newX2[:, -1].astype(int)}
            mu2, var2 = m2.predict(newX2, Y_metadata=noise_dict2)
            list_mu2.append(mu2)
            list_std2.append(np.sqrt(var2))

        list_dict2 = {'list_mu': list_mu2, 'list_std': list_std2, 'gp_type': 'ARD (Coreg)'}
        list_dict_data.append(list_dict2)

    if any(elem in flag for elem in [3]):
        ###############################
        #ICM model
        ###############################
        file_path3 = file_path + 'm3' + '_params.npy'
        icm = GPy.util.multioutput.ICM(input_dim=Field.field_param['dim_in'], num_outputs=Field.field_param['dim_out'],
                                       W_rank=rank_W, kernel=K.copy())
        m3 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=icm)
        m3.update_model(False)
        m3.initialize_parameter()
        m3[:] = np.load(file_path3)
        m3.update_model(True)
        print('\n############### \nICM\n############### \n', m3)

        # Extend data to tell model to which output data belongs
        list_mu3 = []
        list_std3 = []
        for i in range(Field.field_param['dim_out']):
            newX3 = np.hstack([Field.X_predict.T, i*np.ones_like(Field.X_predict.T[:,[0]])])
            noise_dict3 = {'output_index': newX3[:, -1].astype(int)}
            mu3, var3 = m3.predict(newX3, Y_metadata=noise_dict3)
            list_mu3.append(mu3)
            list_std3.append(np.sqrt(var3))

        list_dict3 = {'list_mu': list_mu3, 'list_std': list_std3, 'gp_type': 'ICM'}
        list_dict_data.append(list_dict3)

    if any(elem in flag for elem in [4]):
        ################################
        # LCM model
        ################################
        file_path4 = file_path + 'm4' + '_params.npy'

        K1 = GPy.kern.Bias(Field.field_param['dim_in'])
        K2 = GPy.kern.Linear(Field.field_param['dim_in'])
        elcm = GPy.util.multioutput.LCM(input_dim=Field.field_param['dim_in'], num_outputs=Field.field_param['dim_out'],
                                        W_rank=rank_W, kernels_list=[K1, K2, K.copy()])
        m4 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=elcm)
        m4.update_model(False)
        m4.initialize_parameter()
        m4[:] = np.load(file_path4)
        m4.update_model(True)
        print('\n############### \nLCM \n############### \n', m4)

        # Extend data to tell model to which output data belongs
        list_mu4 = []
        list_std4 = []
        for i in range(Field.field_param['dim_out']):
            newX4 = np.hstack([Field.X_predict.T, i*np.ones_like(Field.X_predict.T[:,[0]])])
            noise_dict4 = {'output_index': newX4[:, -1].astype(int)}
            mu4, var4 = m4.predict(newX4, Y_metadata=noise_dict4)
            list_mu4.append(mu4)
            list_std4.append(np.sqrt(var4))

        list_dict4 = {'list_mu': list_mu4, 'list_std': list_std4, 'gp_type': 'LCM'}
        list_dict_data.append(list_dict4)

    if any(elem in flag for elem in [6]):
        """
        Compute preditions for AMGP with slow but stable script
        """
        print('AMGP')
        file_path5 = file_path + 'm5' + '_params.npy'
        with open(file_path5, 'rb') as f:  # Python 3: open(..., 'rb')
            optim_bounds_2, theta_param_2, res_Indep_2, res_AMGP_2, Field_2 = pickle.load(f)
        #theta_AMGP = np.load(file_path4)
        Gp_AMGP = subclass_AMGP_normalized(Field)
        print('AMGP 2')
        dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGP_2.x,
                                                     Gp_AMGP.covariance, Gp_AMGP.covariance, Gp_AMGP.covariance)
        print('AMGP 3')
        list_dict_data.append(dict_post_AMGP)

    if any(elem in flag for elem in [7]):
        Gp_Indep = class_GP(Field)
        res_Indep = Gp_Indep.minimize_LogML(Field.X_train, Field.Y_train_noisy, theta_param, optim_bounds, Field, Gp_Indep.covariance)
        dict_post_Indep = Gp_Indep.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_Indep.x,
                                                     Gp_Indep.covariance, Gp_Indep.covariance, Gp_Indep.covariance)
        list_dict_data.append(dict_post_Indep)

    if any(elem in flag for elem in [8]):
        """
        Compute preditions for AMGP with faster script
        """
        if flag_estimate_sys_params == False and flag_mean_prior == False:
            file_path81 = file_path + 'res_AMGPx.npy'
            file_path82 = file_path + 'res_AMGPfun.npy'
        elif flag_estimate_sys_params == False and flag_mean_prior == True:
            file_path81 = file_path + 'res_AMGPx_mechanistic.npy'
            file_path82 = file_path + 'res_AMGPfun_mechanistic.npy'
        elif flag_estimate_sys_params == True and flag_mean_prior == False:
            file_path81 = file_path + 'res_AMGPx_params.npy'
            file_path82 = file_path + 'res_AMGPfun_params.npy'
        elif flag_estimate_sys_params == True and flag_mean_prior == True:
            file_path81 = file_path + 'res_AMGPx_params_mechanistic.npy'
            file_path82 = file_path + 'res_AMGPfun_params_mechanistic.npy'
        res_AMGPx = np.load(file_path81)
        res_AMGPfun = np.load(file_path82)
        Gp_AMGP = subclass_AMGP_normalized(Field, flag_mean_prior=flag_mean_prior)
        print('AMGP 2: ', res_AMGPx, res_AMGPfun)
        print('Test: ', res_AMGPx[-2]/res_AMGPx[-1], res_AMGPx[-2]+res_AMGPx[-1])
        Gp_AMGP.init_update_point({'theta': res_AMGPx, # res_AMGP_2.x
                                   'covariance_TT': Gp_AMGP.covariance,
                                   'Field': Field})
        print('AMGP 3')
        dict_post_AMGP = Gp_AMGP.compute_prediction_for_dataset(Field, res_AMGPx, # res_AMGP_2.x
                                                                Gp_AMGP.covariance,
                                                                Gp_AMGP.covariance,
                                                                Gp_AMGP.mean)
        # dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGPx,
        #                                      Gp_AMGP.covariance,
        #                                      Gp_AMGP.covariance, Gp_AMGP.covariance,
        #                                      Gp_AMGP.mean)
        list_dict_data.append(dict_post_AMGP)

    if any(elem in flag for elem in [9]):
        # Predict the unconstrained acceleration
        if flag_estimate_sys_params == False and flag_mean_prior == False:
            file_path81 = file_path + 'res_AMGPx.npy'
            file_path82 = file_path + 'res_AMGPfun.npy'
        elif flag_estimate_sys_params == False and flag_mean_prior == True:
            file_path81 = file_path + 'res_AMGPx_mechanistic.npy'
            file_path82 = file_path + 'res_AMGPfun_mechanistic.npy'
        elif flag_estimate_sys_params == True and flag_mean_prior == False:
            file_path81 = file_path + 'res_AMGPx_params.npy'
            file_path82 = file_path + 'res_AMGPfun_params.npy'
        elif flag_estimate_sys_params == True and flag_mean_prior == True:
            file_path81 = file_path + 'res_AMGPx_params_mechanistic.npy'
            file_path82 = file_path + 'res_AMGPfun_params_mechanistic.npy'

        res_AMGPx = np.load(file_path81)
        res_AMGPfun = np.load(file_path82)
        Gp_AMGP = subclass_AMGP_normalized(Field)
        dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGPx,
                                             Gp_AMGP.covariance_same_same_scalar,
                                             Gp_AMGP.covariance_corr, Gp_AMGP.covariance,
                                             Gp_AMGP.mean_zero)
        list_dict_data.append(dict_post_AMGP)

    if any(elem in flag for elem in [10]):
        # Predict constrained acceleration for different theta
        if flag_estimate_sys_params == False:
            file_path81 = file_path + 'res_AMGPx.npy'
            file_path82 = file_path + 'res_AMGPfun.npy'
        else:
            file_path81 = file_path + 'res_AMGPx_params.npy'
            file_path82 = file_path + 'res_AMGPfun_params.npy'
        res_AMGPx = np.load(file_path81)
        res_AMGPfun = np.load(file_path82)
        Gp_AMGP = subclass_AMGP_normalized(Field)
        #Gp_AMGP.theta_different = np.copy(res_AMGPx)
        #Gp_AMGP.theta_different = np.array([0.08, 0.05, 0.05, 0, 0, 0.1, 3])
        Gp_AMGP.theta_different = np.array([0, 0, 0, 0.1, -0.15, -0.1, 3])
        dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGPx,
                                             Gp_AMGP.covariance_AMGP_different_theta,
                                             Gp_AMGP.covariance_AMGP_different_theta_corr,
                                             Gp_AMGP.covariance,
                                             Gp_AMGP.mean_different_theta)
        list_dict_data.append(dict_post_AMGP)

    ################################
    # Compute samples
    ################################
    if flag_plot_samples == True and flag == [8]:
        n_samp = 30
        f_samples_prior = Gp_AMGP.sample_n(n_samp, Gp_AMGP.sample_prior, Field.X_predict, Gp_AMGP.mean, res_AMGPx, Gp_AMGP.covariance)
        f_samples_posterior = Gp_AMGP.sample_n(n_samp, Gp_AMGP.sample_posterior, dict_post_AMGP['mu'], dict_post_AMGP['L_pred'])
    else:
        f_samples_prior = None
        f_samples_posterior = None


    ################################
    # Compute prediction metrices
    ###############################
    for i in range(len(flag)):
        array_constraint_error[i,jj] = np.max(compute_constraint_error(Field, list_dict_data[i], Field.field_param['params']))
        if flag[i]==8:
            array_constraint_error_est[i, jj] = np.max(compute_constraint_error(Field, list_dict_data[i], res_AMGPx[-len(Field.field_param['params']):]))
        for j in range(Field.field_param['dim_out']):
            array_mean_RMSE[i,jj] = array_mean_RMSE[i,jj] + (1/Field.field_param['dim_out'])*rmse(list_dict_data[i]['list_mu'][j].flatten(), Field.Y_field_ordered[j,:].flatten())
            array_Rsquared[i,jj] = array_Rsquared[i,jj] + (1/Field.field_param['dim_out'])*r_squared(list_dict_data[i]['list_mu'][j].flatten(),
                                         Field.Y_field_ordered[j,:].flatten())

################################
# Save data
################################
if flag_savedata == True:
    # file_path = folder_path + str(jj) + '/'
    file_path_predictions = folder_path + 'predictions_' + str(flag) + \
    str(flag_estimate_sys_params) + '_' + str(flag_mean_prior) + '/'
    if os.path.isdir(file_path_predictions):
        pass
    elif jj == 0:
        os.mkdir(file_path_predictions)
    np.save(file_path_predictions + 'array_mean_RMSE.npy', array_mean_RMSE)
    np.save(file_path_predictions + 'array_Rsquared.npy', array_Rsquared)
    np.save(file_path_predictions + 'array_constraint_error.npy', array_constraint_error)

for i in range(len(flag)):
    print('Model: ', i)
    print('Max Est Constraint error min', np.min(array_constraint_error_est[i, :]))
    print('Max Est Constraint error max', np.max(array_constraint_error_est[i, :]))

    print('Max Real Constraint error min', np.min(array_constraint_error[i, :]))
    print('Max Real Constraint error max', np.max(array_constraint_error[i, :]))
    print('Max Real Constraint error mean', np.mean(array_constraint_error[i, :]))

    print('Mean RMSE min', np.min(array_mean_RMSE[i, :]))
    print('Mean RMSE max', np.max(array_mean_RMSE[i, :]))
    print('Mean RMSE mean', np.mean(array_mean_RMSE[i, :]))

    print('Rsquared min', np.min(array_Rsquared[i, :]))
    print('Rsquared max', np.max(array_Rsquared[i, :]))
    print('Rsquared mean', np.mean(array_Rsquared[i, :]))

################################
# Plots
################################
plot_output(list_dict_data, Field, file_path, flag_save=flag_savedata)
#plot_slice(list_dict_data, Field, file_path, flag_save=flag_savedata)
#for i in range(len(list_dict_data)):
#   check_constraint_n(Field, list_dict_data[i], file_path, flag_save=flag_savedata)
#plot_slice_L4DC(list_dict_data, Field, 'simulation_data/', flag_unscale=True, samples_prior=f_samples_prior, samples_posterior=f_samples_posterior)
# plt.show()

#print('list_constraint_error:', array_constraint_error)
#print('list_mean_RMSE:', array_mean_RMSE)
#print('list_Rsquared:', array_Rsquared)

