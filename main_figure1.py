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
flag_ode = 'mass_surface'
#flag_ode = 'unicycle'
#flag_ode = 'duffling_oscillator'

flag_control = True
number_observations = 10
flag_estimate_sys_params = False

# GP-settings
# 1: Predict constrained acceleration with SE-ARD (GPy)
# 8: Predict constrained acceleration with GP^2 model
# 10: Predict constrained acceleration with different theta with GP^2 model
flag = [1, 8, 10]  # 8

# Script-settings
flag_training_points_in_prediction_slice = True
folder_path = 'data_l4dc/optim' + '2_' + flag_ode + '/'

################################
# Load ODE TRAINING data
################################
# Define slice for predictions and compute prediction points
file_path = folder_path + str(0) + '/'
with open(file_path + '/result', 'rb') as f:  # Python 3: open(..., 'rb')
    optim_bounds, theta_param, res_Indep, res_AMGP, Field = pickle.load(f)
assert type(Field.X_train) != None and Field.field_param['ode_flag'] == flag_ode

if flag_ode == 'unicycle':
    if flag_control is False:
        sys.exit('No data available. Set "flag_control=True".')
    elif flag_control is True:
        Field.field_param['lim_min'] = [1, 1, 0, 0, 1, 0.5]  # [0, 0, 0, -0.5, -1, -0.5]
        Field.field_param['lim_num'] = [1, 1, 100, 1, 1, 1]
        #Field.field_param['lim_num'] = [5, 1, 8, 5, 5, 5]
        Field.field_param['lim_train_min'] = [0, 0, 0, -0.5, -1, -0.5]
        Field.field_param['lim_train_max'] = [1, 1, 2*np.pi, 0.5, 1, 0.5]

elif flag_ode == 'mass_surface':
    # Change the [1st,2nd,4,5,7,8,9] entries in field_param['lim_min'] to define the prediction slice
    # The other entries are computed using the constraint equation
    # therefore code only works if field_param['lim_num'][3,6]= [1, 1]
    assert type(Field.X_train) != None and Field.field_param['ode_flag'] == flag_ode
    if flag_control is False:
        sys.exit('No data available. Set "flag_control=True".')
    elif flag_control is True:
        #Field.field_param['lim_min'] = [-2, -2, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]
        Field.field_param['lim_min'] = [-2, 0, -0.01, 0, 0, 0, 0, 0, 0]
        #Field.field_param['lim_min'] = [0, 0, -0.01, -0.5, 0.5, 0.5, 0, 0, 0]
        Field.field_param['lim_num'] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
        #Field.field_param['lim_num'] = [5, 5, 1, 5, 5, 1, 3, 3, 3]
        Field.field_param['lim_train_max'] = [0, 2, 0.01, 0.5, 0.5, 0.5, 5, 5, 5]

elif flag_ode == 'duffling_oscillator':
    if flag_control is False:
        # Change the [1,3,4,5] entries in field_param['lim_min'] to define the prediction slice
        # The second entry is computed using the constraint equation
        # therefore code aonly works if field_param['lim_num'][1]=1
        #Field.field_param['lim_min'] = [-4, -4, -5, -5, 0]
        Field.field_param['lim_min'] = [-2, 3, -5, 1, 0]
        Field.field_param['lim_num'] = [1, 1, 1, 1, 100]  # x1, x2, x1_dot, x2_dot, t
        Field.field_param['lim_train_max'] = [4, 4, 5, 5, 5] # [4, 4, 5, 5, 5]
    elif flag_control is True:
        sys.exit('Duffling oscillator has no control inputs. Set "flag_control=False".')

if (flag_training_points_in_prediction_slice == True):
    Field.compute_training_points(number_observations, Field.field_param['noise_std'], observation_flag='random')

Field.compute_prediction_points()

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

if any(elem in flag for elem in [8]):
    """
    Compute preditions for AMGP with faster script
    """
    if flag_estimate_sys_params == False:
        file_path81 = file_path + 'res_AMGPx.npy'
        file_path82 = file_path + 'res_AMGPfun.npy'
    else:
        file_path81 = file_path + 'res_AMGPx_params.npy'
        file_path82 = file_path + 'res_AMGPfun_params.npy'
    res_AMGPx = np.load(file_path81)
    res_AMGPfun = np.load(file_path82)
    Gp_AMGP = subclass_AMGP_normalized(Field)
    print('AMGP 2: ', res_AMGPx, res_AMGPfun)
    print('Test: ', res_AMGPx[-2]/res_AMGPx[-1], res_AMGPx[-2]+res_AMGPx[-1])
    # Gp_AMGP.init_update_point({'theta': res_AMGPx, # res_AMGP_2.x
    #                            'covariance_TT': Gp_AMGP.covariance,
    #                            'Field': Field})
    # print('AMGP 3')
    # dict_post_AMGP = Gp_AMGP.compute_prediction_for_dataset(Field, res_AMGPx, # res_AMGP_2.x
    #                                                         Gp_AMGP.covariance,
    #                                                         Gp_AMGP.covariance,
    #                                                         Gp_AMGP.mean)
    dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGPx,
                                         Gp_AMGP.covariance,
                                         Gp_AMGP.covariance, Gp_AMGP.covariance,
                                         Gp_AMGP.mean)
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
    Field_different = subclass_ODE_mass_on_surface(flag_control)
    from copy import deepcopy
    #Field_different.field_param = deepcopy(Field.field_param)
    Field_different = deepcopy(Field)
    Field_different.field_param['params'] = [0, 0, 0, 0.1, -0.15, -0.1, 3]
    #Field_different.dict_norm_X = deepcopy(Field.dict_norm_X)
    #Field_different.dict_norm_Y = deepcopy(Field.dict_norm_Y)
    Field_different.compute_prediction_points()
    dict_post_AMGP_different = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGPx,
                                         Gp_AMGP.covariance_AMGP_different_theta,
                                         Gp_AMGP.covariance_AMGP_different_theta_corr,
                                         Gp_AMGP.covariance,
                                         Gp_AMGP.mean_different_theta)
    list_dict_data.append(dict_post_AMGP_different)

################################
# Plots
################################
para_lw = 2
flag_unscale=True
flag_unconstrained=False

list_slice = [slice(None)] * len(Field.field_param['lim_num'])
for i in range(len(list_slice)):
    if Field.field_param['lim_num'][i] == 1:
        list_slice[i] = 0
    else:
        flag_nonconstant = i
tuple_slice = tuple(list_slice)
list_fmin = []
list_fmax = []
fig, ax = plt.subplots(Field.field_param['dim_out'], 1, figsize=(7, 9), sharex=True)  # , sharex='col', sharey='row')
# fig.set_size_inches(12/2.54, 15/2.54)
fig.set_size_inches(12 / 2.54, 15 / 2.54 / 2)

assert np.array_equal(Field.X_predict[0:2,:], Field_different.X_predict[0:2,:])
assert np.array_equal(Field.X_predict[3:5,:], Field_different.X_predict[3:5,:])
Field.Y_field_ordered = Field.un_normalize_points(Field.Y_field_ordered, Field.dict_norm_Y)
Field.Y_train_noisy_tmp = Field.un_normalize_points(Field.Y_train_noisy_tmp, Field.dict_norm_Y)
Field.X_predict = Field.un_normalize_points(Field.X_predict, Field.dict_norm_X)
Field.X_train = Field.un_normalize_points(Field.X_train, Field.dict_norm_X)

Field.Y_field_ordered_different = Field_different.un_normalize_points(Field_different.Y_field_ordered, Field.dict_norm_Y)

list_gpname = ['SE', r'GP$^2$',  r'GP$^2$']
list_alpha = [0.7, 0.5, 0.3]
list_colors_mean = ['gray', 'blue', 'goldenrod']
list_colors_var = ['lightgray', 'steelblue', 'goldenrod']
list_line = ['-', '-', '-']

marker_color = 'turquoise'
prior_sample_color = ['lightblue', 'steelblue']
field_param_lim_num_train = np.copy(Field.field_param['lim_num'])
for i in range(len(field_param_lim_num_train)):
    if field_param_lim_num_train[i] > 1:
        field_param_lim_num_train[i] = Field.field_param['number_observations']

for i in range(Field.field_param['dim_out']):
    for j in range(len(list_dict_data)):
        if flag_unscale == True:
            mue_restack = np.copy(Field.restack_prediction_list(list_dict_data[j]['list_mu']))
            std_restack = np.copy(Field.restack_prediction_list(list_dict_data[j]['list_std']))
            tmp_mean = Field.un_normalize_points(mue_restack, Field.dict_norm_Y)
            tmp_std = Field.un_normalize_std(std_restack, Field.dict_norm_Y)
            mean = tmp_mean[i, :]
            std_deviation = tmp_std[i, :]
        else:
            mean = list_dict_data[j]['list_mu'][i]
            std_deviation = list_dict_data[j]['list_std'][i]

        tmp_mu = mean.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
        tmp_std = std_deviation.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]

        low_conf = (tmp_mu - 2 * tmp_std).flatten()
        high_conf = (tmp_mu + 2 * tmp_std).flatten()
        tmp_field = Field.Y_field_ordered[i, :].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
        tmp_field_different = Field.Y_field_ordered_different[i, :].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
        tmp_positions = Field.X_predict[flag_nonconstant, :].reshape(Field.field_param['lim_num'], order='C')[
            tuple_slice]
        tmp_training_X = Field.X_train[flag_nonconstant, :]

        ax[i].fill_between(tmp_positions.flatten(), low_conf, high_conf, color=list_colors_var[j], alpha=list_alpha[j],
                           label=list_gpname[j] + ' confid.')
        ax[i].plot(tmp_positions, tmp_mu, lw=para_lw, color=list_colors_mean[j], linestyle=list_line[j],
                   label=list_gpname[j] + ' mean')

        if i == 0 and j == 0:
            ax[i].plot(tmp_positions, tmp_field_different, lw=para_lw, color='saddlebrown', linestyle=':', label=r'$\ddot{q}^{\,\prime}$')
            ax[i].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':', label=r'$\ddot{q}$')
            ax[i].plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
                       markersize=8, marker='o', linestyle='None', label=r'$y$')
        else:
            ax[i].plot(tmp_positions, tmp_field_different, lw=para_lw, color='saddlebrown', linestyle=':')
            ax[i].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':')
            ax[i].plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
                       markersize=8, marker='o', linestyle='None')

        fmin = np.min(low_conf)
        list_fmin.append(fmin)
        fmax = np.max(high_conf)
        list_fmax.append(fmax)
        ax[-1].set_xlabel(r'$q_{0}$'.format(flag_nonconstant + 1) + r' [m]')
        ax[i].set_ylabel(r'$\ddot{q}$' + r'$_{0}$'.format(i + 1) + r' [m/s$^2$]')
    ax[i].tick_params(axis='x', labelsize=10)
    ax[i].tick_params(axis='y', labelsize=10, rotation=90)
    ax[i].set_xticks([-2, 0, 2])
    ax[i].grid(True)
ax[0].set_yticks([-5, 0, 5])
ax[0].set_ylim(-6, 6)
ax[1].set_yticks([-1, 0, 1])
ax[1].set_ylim(-1.5, 2.5)
ax[2].set_yticks([-3, -1, 1])
ax[2].set_ylim(-3.2, 1.6)

handles, labels = ax[0].get_legend_handles_labels()
order1 = [3, 2, 0, 4, 6, 7]
order2 = [1, 5, 8]
#order = [1, 0, 2, 3]

legend1 = ax[0].legend([handles[idx] for idx in order1],[labels[idx] for idx in order1], loc='upper center',
            ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.7, bbox_to_anchor=(0.5, 1.7), prop={'size': 10}) #bbox_to_anchor=(0.7,1.3)
legend2 = ax[0].legend([handles[idx] for idx in order2],[labels[idx] for idx in order2], loc='upper center',
            ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.7, bbox_to_anchor=(0.5, 1.27), prop={'size': 10}) #bbox_to_anchor=(0.7,1.3)

ax[0].add_artist(legend1)
ax[0].annotate(r'$\hat{h}\,|y$', xy=(0.12, 1.46), xytext=(0.04, 1.46), xycoords='axes fraction',
            fontsize=12, ha='center', va='center',
            #bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0))
ax[0].annotate(r'$\hat{h}^{\prime}|y$', xy=(0.12, 1.11), xytext=(0.04, 1.11), xycoords='axes fraction',
            fontsize=12, ha='center', va='center',
            #bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=0.5, lengthB=0.2', lw=2.0))

ax[0].text(0.25, 0.87, r"surface ($v=0$)",
           transform=ax[0].transAxes, fontsize=10)

plt.xlim(np.min(Field.X_predict[flag_nonconstant, :]), np.max(Field.X_predict[flag_nonconstant, :]))
# plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.91, wspace=0.005)
plt.subplots_adjust(left=0.13, bottom=0.09, right=0.98, top=0.85, wspace=0.1,hspace=0.1)
# plt.tight_layout()
# fig.tight_layout()
plt.show()

