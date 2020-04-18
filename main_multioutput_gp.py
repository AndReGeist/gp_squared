# Copyright 2019 Max Planck Society. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import GPy
import pickle
import datetime
import os

from subclass_AMGP_normalized import *

from subclass_ODE_mass_on_surface import *
from subclass_ODE_unicycle import *
from subclass_ODE_duffling_oscillator import *

from plot_scripts import *

# SYSTEM-settings
flag_ode = 'mass_surface'
#flag_ode = 'unicycle'
#flag_ode = 'duffling'
#flag_ode2 = 'duffling'
flag_control = True

number_observations = 200
noise_std = 0.1  # unicycle 0.01, surface 0.1, duffing 1

flag_estimate_sys_params = True
flag_mean_prior = False

dev_con = 0.02   # unicycle 0.02, surface  0.02, duffing  0.02
l_min = 0.5
l_max = 20
sig_var_min = 0.5
sig_var_max = 10

theta_pts = 3  # Number of optimization restarts, e.g. 30

# GP-settings
# GP-settings
# 1: Learn constrained acceleration with SE-ARD (GPy)
# 3: Learn constrained acceleration with ICM (GPy)
# 4: Learn constrained acceleration with LMC (GPy)
# 6: Learn constrained acceleration with GP^2 model

flag = [6]  # 6 = AMGP

# Script-settings
folder_path = 'data_l4dc/optim' + '2_' + flag_ode + '/'
flag_loadfield = True  # Load Field for using the same previously sampled observations
flag_training_points_in_prediction_slice = False
flag_predictions = False  # Directly plot predictions
flag_savedata = True
reruns_script = 10  # Number of different observation data sets

print('flag_ode: ', flag_ode, '\n')
print('number_observations: ', number_observations, '\n')
print('noise_std: ', noise_std, '\n')
print('flag_estimate_sys_params: ', flag_estimate_sys_params, '\n')
print('flag_mean_prior: ', flag_mean_prior, '\n')
print('theta_pts: ', theta_pts, '\n')
print('flag_loadfield: ', flag_loadfield, '\n')
print('flag_training_points_in_prediction_slice: ', flag_training_points_in_prediction_slice, '\n')
print('flag_savedata: ', flag_savedata, '\n')
print('flag: ', flag, '\n')
print('reruns_script: ', reruns_script, '\n')

file_path = None
for jj in range(reruns_script):
    print('RUN: ', jj)
    if flag_savedata == True:
        file_path = folder_path + str(jj) + '/'
        if flag_loadfield == False:
            if os.path.isdir(file_path):
                pass
            else:
                os.mkdir(file_path)
            if os.path.isdir(folder_path):
                pass
            else:
                os.mkdir(folder_path)

    ################################
    # Compute/Load ODE TRAINING data
    ################################
    if flag_loadfield == True:
        file_path = folder_path + str(jj) + '/'
        with open(file_path + 'result', 'rb') as f:  # Python 3: open(..., 'rb')
            optim_bounds, theta_param, res_Indep, res_AMGP, Field = pickle.load(f)

#        assert Field.field_param['ode_flag'] == flag_ode
        if flag_ode == 'duffling' and flag_mean_prior == True:
            Field.field_param['params'] = [10, 12, 0.1, 0.15, 1, 0.3, 2 * np.pi]

        noise_std = Field.field_param['noise_std']
        assert type(Field.X_train) != None

    else:
        if flag_ode == 'unicycle':
            Field = subclass_ODE_unicycle(flag_control)
        elif flag_ode == 'mass_surface':
            Field = subclass_ODE_mass_on_surface(flag_control)
        elif flag_ode == 'duffling':
            Field = subclass_ODE_duffling_oscillator(flag_control, flag_mean_prior)

    if (flag_training_points_in_prediction_slice == False) and (flag_loadfield == False):
        Field.compute_training_points(number_observations, noise_std, observation_flag='random')

    # Define slice for predictions and compute prediction points
    if flag_predictions == True:
        if flag_ode == 'unicycle':
            if flag_control is False:
                Field.field_param['lim_num'] = [100, 1, 1, 1]
                Field.field_param['lim_train_max'] = [0.5, 1, 2 * np.pi, 0.5]
            elif flag_control is True:
                pass
                #Field.field_param['lim_min'] = [0.5, 0, 0, 0, 0, -0.5]
                #Field.field_param['lim_num'] = [100, 1, 1, 1, 1, 1]
                #Field.field_param['lim_num'] = [5, 1, 5, 5, 5, 5]
                #Field.field_param['lim_train_max'] = [0.5, 1, 2 * np.pi, 0.5, 1, 0.5]

        elif flag_ode == 'mass_surface':
            if flag_control is False:
                Field.field_param['lim_num'] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
                Field.field_param['lim_train_max'] = [2, 2, 0.001, 0.5, 0.5, 0.5]
            elif flag_control is True:
                pass
                Field.field_param['lim_num'] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
                Field.field_param['lim_train_max'] = [0, 2, 0.001, 0.5, 0.5, 0.5, 5, 5, 5]

        elif flag_ode == 'duffling':
            if flag_control is False:
                pass
                #Field.field_param['lim_num'] = [1, 1, 1, 1, 100]  # x1, x2, x1_dot, x2_dot, t
                #Field.field_param['lim_min'] = [-4, -4, -5, -5, -5]
                #Field.field_param['lim_min'] = [-4, -0, -0, -0, 3],
                #Field.field_param['lim_train_max'] = [4, 4, 5, 5, 5]
            elif flag_control is True:
                sys.exit('Control inputs are not available for the closed-loop Duffling oscillator. Set "flag_control=False".')

        Field.compute_prediction_points()

    if (flag_training_points_in_prediction_slice == True) and (flag_loadfield == False):
        Field.compute_training_points(number_observations, noise_std, observation_flag='random')

    if Field.field_param['flag_normalize_out'] == True:
        noise_var_scaled = np.diag(Field.dict_norm_Y['N_std'] * noise_std) ** 2
    else:
        noise_var_scaled = np.array([1, 1, 1])*noise_std**2

    # Compute new optimization bounds
    theta_param, optim_bounds = Field.set_optimization_parameters(theta_pts, flag_estimate_sys_params,
                                                                  dev_con=dev_con,
                                                                  l_min=l_min,
                                                                  l_max=l_max,
                                                                  sig_var_min=sig_var_min,
                                                                  sig_var_max=sig_var_max)
    if flag_estimate_sys_params == True and flag_ode == 'mass_surface':
        theta_param, optim_bounds = Field.fix_param_in_optim_bounds(theta_param, optim_bounds, [-4, -3], [0, 0])

    # Reformat data
    Xt = [Field.X_train.T[:,[i]] for i in range(Field.field_param['dim_in'])]
    Yt = [Field.Y_train_noisy_tmp.T[:, [i]] for i in range(Field.field_param['dim_out'])]

    Xt_multi = [Field.X_train.T for i in range(len(Yt))]

    ################################
    # PREPARE SAVING DATA
    ################################
    res_Indep = None
    res_AMGP = None

    ################################
    # GP Optimization
    ################################
    list_dict_data = []
    rank_W = Field.field_param['dim_out']
    K = GPy.kern.RBF(input_dim=Field.field_param['dim_in'], ARD=True)
    K.constrain_positive('.*rbf_variance')
    K.variance.constrain_bounded(sig_var_min, sig_var_max)
    K.lengthscale.constrain_bounded(l_min, l_max)

    if any(elem in flag for elem in [1]):
        ################################
        # Standard GPy ARD-GP
        ################################
        list_mu1 = []
        list_std1 = []
        for i in range(Field.field_param['dim_out']):
            m1 = GPy.models.GPRegression(Field.X_train.T, Field.Y_train_noisy_tmp.T[:, [i]], K.copy())
            m1['.*rbf.variance'].constrain_bounded(0.01, 1)
            m1['.*Gaussian_noise.variance'].constrain_fixed(noise_var_scaled[i])
            m1.optimize_restarts(num_restarts=theta_param['theta_pts'])
            print('\n############### \nARD \n###############\n','i: ', i, '\n', m1)
            if flag_savedata == True:
                np.save(file_path + 'm1'+ str(i) + '_params.npy', m1.param_array)

            if flag_predictions == True:
                mu1, var1 = m1.predict(Field.X_predict.T)
                list_mu1.append(mu1)
                list_std1.append(np.sqrt(var1))
        list_dict1 = {'list_mu': list_mu1, 'list_std': list_std1, 'gp_type': 'ARD'}
        list_dict_data.append(list_dict1)

    if any(elem in flag for elem in [2]):
        ###############################
        # Independent GPs using ICM model in GPy
        ###############################
        ICM_indep = GPy.util.multioutput.ICM(input_dim=Field.field_param['dim_in'], num_outputs=Field.field_param['dim_out'], kernel=K.copy())
        m2 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=ICM_indep)
        m2['.*W'].constrain_fixed(0)
        m2['.*rbf.variance'].constrain_positive('.*rbf_variance')
        m2['.*rbf.variance'].constrain_fixed(1.)  # For this kernel, B.kappa encodes the variance now.
        m2['.*Gaussian_noise_0.variance'].constrain_fixed(noise_var_scaled[0])
        m2['.*Gaussian_noise_1.variance'].constrain_fixed(noise_var_scaled[1])
        if flag_ode != 'duffling':
            m2['.*Gaussian_noise_2.variance'].constrain_fixed(noise_var_scaled[2])

        m2.optimize_restarts(num_restarts=theta_param['theta_pts'])
        print('\n############### \nARD (Correg)\n############### \n', m2)

        # Extend data to tell model to which output data belongs
        if flag_predictions == True:
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
        icm = GPy.util.multioutput.ICM(input_dim=Field.field_param['dim_in'], num_outputs=Field.field_param['dim_out'],
                                       W_rank=rank_W, kernel=K.copy())
        m3 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=icm)
        #m3['.*kappa'].constrain_fixed(1e-8)
        #m3['.*rbf.variance'].constrain_fixed(1.)
        m3['.*Gaussian_noise_0.variance'].constrain_fixed(noise_var_scaled[0])
        m3['.*Gaussian_noise_1.variance'].constrain_fixed(noise_var_scaled[1])
        if flag_ode != 'duffling':
            m3['.*Gaussian_noise_2.variance'].constrain_fixed(noise_var_scaled[2])
        m3.optimize_restarts(num_restarts=theta_param['theta_pts'])
        print('\n############### \nICM\n############### \n', m3)

        if flag_predictions == True:
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
        K1 = GPy.kern.Bias(Field.field_param['dim_in'])
        K2 = GPy.kern.Linear(Field.field_param['dim_in'])
        elcm = GPy.util.multioutput.LCM(input_dim=Field.field_param['dim_in'], num_outputs=Field.field_param['dim_out'],
                                        W_rank=rank_W, kernels_list=[K1, K2, K.copy()])
        m4 = GPy.models.GPCoregionalizedRegression(Xt_multi, Yt, kernel=elcm)
        m4['.*ICM.*var'].unconstrain()
        m4['.*ICM.*var'].constrain_positive('.*rbf_variance')
        m4['.*ICM0.*var'].constrain_fixed(1.)
        m4['.*ICM0.*W'].constrain_fixed(0)
        m4['.*ICM1.*var'].constrain_fixed(1.)
        m4['.*ICM1.*W'].constrain_fixed(0)
        #m4['.*kappa'].constrain_fixed(0)
        #m4['.*rbf.variance'].constrain_fixed(1.)
        m4['.*Gaussian_noise_0.variance'].constrain_fixed(noise_var_scaled[0])
        m4['.*Gaussian_noise_1.variance'].constrain_fixed(noise_var_scaled[1])
        if flag_ode != 'duffling':
            m4['.*Gaussian_noise_2.variance'].constrain_fixed(noise_var_scaled[2])
        m4.optimize_restarts(num_restarts=theta_param['theta_pts'])
        print('\n############### \nLCM \n############### \n', m4)

        # Extend data to tell model to which output data belongs
        if flag_predictions == True:
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
        # AMGP Model
        Gp_AMGP = subclass_AMGP_normalized(Field, flag_mean_prior=flag_mean_prior)
        res_AMGP = Gp_AMGP.minimize_LogML(Field.X_train, Field.Y_train_noisy, theta_param, optim_bounds, Field, Gp_AMGP.covariance)
        if flag_predictions == True:
            dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGP.x,
                                                         Gp_AMGP.covariance, Gp_AMGP.covariance, Gp_AMGP.covariance)
            list_dict_data.append(dict_post_AMGP)

    if any(elem in flag for elem in [7]):
        Gp_Indep = class_GP(Field)
        theta_param_indep = theta_param.copy()
        theta_param_indep['dim_min'] = theta_param['dim_min'][:-theta_param['num_theta_phys']]
        theta_param_indep['dim_max'] = theta_param['dim_max'][:-theta_param['num_theta_phys']]
        optim_bounds2 = optim_bounds[:-theta_param['num_theta_phys']]
        res_Indep = Gp_Indep.minimize_LogML(Field.X_train, Field.Y_train_noisy, theta_param_indep,
                                            optim_bounds2, Field, Gp_Indep.covariance)
        if flag_predictions == True:
            dict_post_Indep = Gp_Indep.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_Indep.x,
                                                         Gp_Indep.covariance, Gp_Indep.covariance, Gp_Indep.covariance)
            list_dict_data.append(dict_post_Indep)

    ################################
    # Save data
    ################################
    if flag_savedata == True:
        # Save Field
        np.save('Yt.npy', Yt)
        np.save('Xt_muti.npy', Xt_multi)
        with open(file_path + 'result', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([optim_bounds, theta_param, res_Indep, res_AMGP, Field], f)

        # Save GP parameters
        if any(elem in flag for elem in [2]):
            np.save(file_path + 'm2_params.npy', m2.param_array)
        if any(elem in flag for elem in [3]):
            np.save(file_path + 'm3_params.npy', m3.param_array)
        if any(elem in flag for elem in [4]):
            np.save(file_path + 'm4_params.npy', m4.param_array)
        if any(elem in flag for elem in [6]):
            if flag_estimate_sys_params == False and flag_mean_prior == False:
                np.save(file_path + 'res_AMGPx.npy', res_AMGP.x)
                np.save(file_path + 'res_AMGPfun.npy', res_AMGP.fun)
            elif flag_estimate_sys_params == False and flag_mean_prior == True:
                np.save(file_path + 'res_AMGPx_mechanistic.npy', res_AMGP.x)
                np.save(file_path + 'res_AMGPfun_mechanistic.npy', res_AMGP.fun)
            elif flag_estimate_sys_params == True and flag_mean_prior == False:
                np.save(file_path + 'res_AMGPx_params.npy', res_AMGP.x)
                np.save(file_path + 'res_AMGPfun_params.npy', res_AMGP.fun)
            elif flag_estimate_sys_params == True and flag_mean_prior == True:
                np.save(file_path + 'res_AMGPx_params_mechanistic.npy', res_AMGP.x)
                np.save(file_path + 'res_AMGPfun_params_mechanistic.npy', res_AMGP.fun)

        if any(elem in flag for elem in [7]):
            np.save(file_path + 'res_Indepx.npy', res_Indep.x)
            np.save(file_path + 'res_Indepfun.npy', res_Indep.fun)


    ################################
    # Plots
    ################################
    if flag_predictions == True:
        #plot_slice(list_dict_data, Field, file_path, flag_save=flag_savedata)
        plot_output(list_dict_data, Field, file_path, flag_save=flag_savedata)
        for i in range(len(list_dict_data)):
            check_constraint_n(Field, list_dict_data[i], file_path, flag_save=flag_savedata)


