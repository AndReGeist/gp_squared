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

# SYSTEM-settings SURFACE
flag_ode = 'mass_surface'

flag_control = True
#number_observations = 10
flag_estimate_sys_params = False

# GP-settings
# 9: Predict UN-constrained acceleration with GP^2 model

flag = [9]

# Script-settings
flag_training_points_in_prediction_slice = False
folder_path = 'data_l4dc/optim' + '2_' + flag_ode + '/'

################################
# Load ODE TRAINING data
################################
# Define slice for predictions and compute prediction points
file_path = folder_path + str(0) + '/'
with open(file_path + '/result', 'rb') as f:  # Python 3: open(..., 'rb')
    optim_bounds, theta_param, res_Indep, res_AMGP, Field = pickle.load(f)
assert type(Field.X_train) != None and Field.field_param['ode_flag'] == flag_ode

assert type(Field.X_train) != None and Field.field_param['ode_flag'] == flag_ode
if flag_control is False:
    Field.field_param['lim_num'] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
    Field.field_param['lim_train_max'] = [0, 2, 0.001, 0.5, 0.5, 0.5]
elif flag_control is True:
    #Field.field_param['lim_min'] = [-2, -2, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]
    Field.field_param['lim_min'] = [0, 0, -0.01, 0, 0, 0, 0, 0, 0]
    #Field.field_param['lim_min'] = [0, 0, -0.01, -0.5, 0.5, 0.5, 0, 0, 0]
    Field.field_param['lim_num'] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
    #Field.field_param['lim_num'] = [5, 5, 1, 5, 5, 1, 3, 3, 3]
    #Field.field_param['lim_train_min'] = [0, 0, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]
    #Field.field_param['lim_train_max'] = [2, 2, 0.01, 0.5, 0.5, 0.5, 5, 5, 5]

if (flag_training_points_in_prediction_slice == True):
    Field.compute_training_points(number_observations, Field.field_param['noise_std'], observation_flag='random')


Field.compute_prediction_points()

# Reformat data
Xt = [Field.X_train.T[:,[i]] for i in range(Field.field_param['dim_in'])]
Yt = [Field.Y_train_noisy_tmp.T[:,[i]] for i in range(Field.field_param['dim_out'])]

Xt_multi = [Field.X_train.T for i in range(len(Yt))]

################################
# PREPARE SAVING DATA
################################
res_Indep = None
res_AMGP = None
list_dict_data = []
if any(elem in flag for elem in [9]):
    # Predict the unconstrained acceleration
    if flag_estimate_sys_params == False:
        file_path81 = file_path + 'res_AMGPx.npy'
        file_path82 = file_path + 'res_AMGPfun.npy'
    else:
        file_path81 = file_path + 'res_AMGPx_params.npy'
        file_path82 = file_path + 'res_AMGPfun_params.npy'
    res_AMGPx = np.load(file_path81)
    res_AMGPfun = np.load(file_path82)
    Gp_AMGP = subclass_AMGP_normalized(Field)
    dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGPx,
                                         Gp_AMGP.covariance_same_same_scalar,
                                         Gp_AMGP.covariance_corr, Gp_AMGP.covariance,
                                         Gp_AMGP.mean_zero)
    list_dict_data.append(dict_post_AMGP)


# SYSTEM-settings UNICYCLE
flag_ode = 'unicycle'
#flag_ode = 'duffling_oscillator'

flag_control = True
#number_observations = 10
flag_estimate_sys_params = False

# GP-settings
# 9: Predict UN-constrained acceleration with GP^2 model

flag = [9]

# Script-settings
flag_training_points_in_prediction_slice = False
folder_path = 'data_l4dc/optim' + '2_' + flag_ode + '/'

################################
# Load ODE TRAINING data
################################
# Define slice for predictions and compute prediction points
file_path = folder_path + str(0) + '/'
with open(file_path + '/result', 'rb') as f:  # Python 3: open(..., 'rb')
    optim_bounds, theta_param, res_Indep, res_AMGP, Field2 = pickle.load(f)
assert type(Field2.X_train) != None and Field2.field_param['ode_flag'] == flag_ode

if flag_ode == 'unicycle':
    if flag_control is False:
        Field2.field_param['lim_num'] = [100, 1, 1, 1]
        Field2.field_param['lim_train_max'] = [0.5, 1, 2 * np.pi, 0.5]
    elif flag_control is True:
        #pass
        Field2.field_param['lim_min'] = [0, 1, 0, 0, 0, 0]  # [0, 0, 0, -0.5, -1, -0.5]
        Field2.field_param['lim_num'] = [100, 1, 1, 1, 1, 1]
        #Field2.field_param['lim_num'] = [5, 1, 8, 5, 5, 5]
        Field2.field_param['lim_train_min'] = [0, 0, 0, -0.5, -1, -0.5]
        Field2.field_param['lim_train_max'] = [1, 1, 2*np.pi, 0.5, 1, 0.5]

elif flag_ode == 'mass_surface':
    assert type(Field2.X_train) != None and Field2.Field2_param['ode_flag'] == flag_ode
    if flag_control is False:
        Field2.field_param['lim_num'] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
        Field2.field_param['lim_train_max'] = [0, 2, 0.001, 0.5, 0.5, 0.5]
    elif flag_control is True:
        #Field2.field_param['lim_min'] = [-2, -2, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]
        Field2.field_param['lim_min'] = [0, 0, -0.01, 0, 0, 0, 0, 0, 0]
        #Field2.field_param['lim_min'] = [0, 0, -0.01, -0.5, 0.5, 0.5, 0, 0, 0]
        Field2.field_param['lim_num'] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
        #Field2.field_param['lim_num'] = [5, 5, 1, 5, 5, 1, 3, 3, 3]
        Field2.field_param['lim_train_min'] = [0, 0, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]
        Field2.field_param['lim_train_max'] = [2, 2, 0.01, 0.5, 0.5, 0.5, 5, 5, 5]

elif flag_ode == 'duffling_oscillator':
    if flag_control is False:
        #pass
        #Field2.field_param['lim_min'] = [-4, -4, -5, -5, 0]
        Field2.field_param['lim_min'] = [-2, 3, -5, 1, 0]
        Field2.field_param['lim_num'] = [1, 1, 1, 1, 100]  # x1, x2, x1_dot, x2_dot, t
        Field2.field_param['lim_train_max'] = [4, 4, 5, 5, 5] # [4, 4, 5, 5, 5]
    elif flag_control is True:
        sys.exit('Control inputs are not available for the closed-loop Duffling oscillator. Set "flag_control=False".')

if (flag_training_points_in_prediction_slice == True):
    Field2.compute_training_points(number_observations, Field2.Field2_param['noise_std'], observation_flag='random')


Field2.compute_prediction_points()

# Reformat data
Xt = [Field2.X_train.T[:,[i]] for i in range(Field2.field_param['dim_in'])]
Yt = [Field2.Y_train_noisy_tmp.T[:,[i]] for i in range(Field2.field_param['dim_out'])]

Xt_multi = [Field2.X_train.T for i in range(len(Yt))]

################################
# Mak prediction for unconstrained acceleration a_bar
################################
res_Indep = None
res_AMGP = None
if any(elem in flag for elem in [9]):
    # Predict the unconstrained acceleration
    if flag_estimate_sys_params == False:
        file_path81 = file_path + 'res_AMGPx.npy'
        file_path82 = file_path + 'res_AMGPfun.npy'
    else:
        file_path81 = file_path + 'res_AMGPx_params.npy'
        file_path82 = file_path + 'res_AMGPfun_params.npy'
    res_AMGPx = np.load(file_path81)
    res_AMGPfun = np.load(file_path82)
    Gp_AMGP2 = subclass_AMGP_normalized(Field2)
    dict_post_AMGP2 = Gp_AMGP2.update_data(Field2.X_predict, Field2.X_train, Field2.Y_train_noisy, res_AMGPx,
                                         Gp_AMGP2.covariance_same_same_scalar,
                                         Gp_AMGP2.covariance_corr, Gp_AMGP2.covariance,
                                         Gp_AMGP2.mean_zero)
    list_dict_data.append(dict_post_AMGP2)

################################
# Plots
################################
para_lw = 2
flag_unscale = True
flag_unconstrained = True

list_fmin = []
list_fmax = []
fig, ax = plt.subplots(Field.field_param['dim_out'], 2, figsize=(7,9), sharex='col')  # , sharex='col', sharey='row')
fig.set_size_inches(12 / 2.54, 15 / 2.54/2)

list_gpname = [r'GP$^2$', r'GP$^2$']
list_alpha = [0.5, 0.5]
list_colors_mean = ['darkgreen', 'darkgreen']
list_colors_var = ['mediumseagreen', 'mediumseagreen']
list_line = ['-', '-']

marker_color = 'turquoise'
field_param_lim_num_train = np.copy(Field.field_param['lim_num'])
for i in range(len(field_param_lim_num_train)):
    if field_param_lim_num_train[i] > 1:
        field_param_lim_num_train[i] = Field.field_param['number_observations']

# Plot 1: SURFACE
list_slice = [slice(None)] * len(Field.field_param['lim_num'])
for i in range(len(list_slice)):
    if Field.field_param['lim_num'][i] == 1:
        list_slice[i] = 0
    else:
        flag_nonconstant = i

tuple_slice = tuple(list_slice)

Field.X_predict = Field.un_normalize_points(Field.X_predict, Field.dict_norm_X)
Field.X_train = Field.un_normalize_points(Field.X_train, Field.dict_norm_X)
Field.Y_field_ordered = Field.define_observations(Field.X_predict, flag_unconstrained='a_bar').reshape((-1, Field.field_param['dim_out'])).T
Field.Y_train_noisy_tmp = Field.un_normalize_points(Field.Y_train_noisy_tmp, Field.dict_norm_Y)

for i in range(Field.field_param['dim_out']):
    if flag_unscale == True:
        mue_restack = np.copy(Field.restack_prediction_list(list_dict_data[0]['list_mu']))
        std_restack = np.copy(Field.restack_prediction_list(list_dict_data[0]['list_std']))
        tmp_mean = Field.un_normalize_points(mue_restack, Field.dict_norm_Y)
        tmp_std = Field.un_normalize_std(std_restack, Field.dict_norm_Y)
        mean = tmp_mean[i, :]
        std_deviation = tmp_std[i, :]
    else:
        mean = list_dict_data[0]['list_mu'][i]
        std_deviation = list_dict_data[0]['list_std'][i]

    tmp_mu = mean.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
    tmp_std = std_deviation.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]

    low_conf = (tmp_mu - 2 * tmp_std).flatten()
    high_conf = (tmp_mu + 2 * tmp_std).flatten()
    tmp_field = Field.Y_field_ordered[i,:].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
    tmp_positions = Field.X_predict[flag_nonconstant, :].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
    tmp_training_X = Field.X_train[flag_nonconstant, :]

    # Use vertical lines to indicate observations
    if flag_training_points_in_prediction_slice == True:
        for ii in range(tmp_training_X.shape[0]):
            ax[i,0].axvline(x=tmp_training_X[ii], linestyle='-', lw=para_lw, marker=None)
    ax[i,0].fill_between(tmp_positions.flatten(), low_conf, high_conf, color=list_colors_var[0], alpha=list_alpha[0],label=list_gpname[0] + ' confid.')
    ax[i,0].plot(tmp_positions, tmp_mu, lw=para_lw, color=list_colors_mean[0], linestyle=list_line[0], label=list_gpname[0] + ' mean')
    if i == 0 and 0==0:
        ax[i,0].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':', label=r'$\bar{a}$')
        #ax[i,0].plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
        #           markersize=8, marker='o', linestyle='None', label='observations')
    else:
        ax[i,0].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':')
        #ax[i,0].plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
        #           markersize=8, marker='o', linestyle='None')
    fmin = np.min(low_conf)
    list_fmin.append(fmin)
    fmax = np.max(high_conf)
    list_fmax.append(fmax)
    ax[-1,0].set_xlabel(r'$q_{0}$'.format(flag_nonconstant+1)+r' [m]')
    ax[i,0].set_ylabel(r'$\bar{a}$'+r'$_{0}$'.format(i+1)+r' [m/s$^2$]')
ax[0, 0].tick_params(axis='x', labelsize=10)
ax[0, 0].tick_params(axis='y', labelsize=10)
ax[1, 0].tick_params(axis='y', labelsize=10)
ax[2, 0].tick_params(axis='y', labelsize=10)
ax[2,0].set_xticks([0, 1, 2])
ax[2,0].set_yticks([-10.5, -9.8, -9])

# PLOT 2: UNICYCLE
list_slice = [slice(None)] * len(Field2.field_param['lim_num'])
for i in range(len(list_slice)):
    if Field2.field_param['lim_num'][i] == 1:
        list_slice[i] = 0
    else:
        flag_nonconstant = i

tuple_slice2 = tuple(list_slice)

Field2.X_predict = Field2.un_normalize_points(Field2.X_predict, Field2.dict_norm_X)
Field2.X_train = Field2.un_normalize_points(Field2.X_train, Field2.dict_norm_X)
Field2.Y_field_ordered = Field2.define_observations(Field2.X_predict, flag_unconstrained='a_bar').reshape((-1, Field2.field_param['dim_out'])).T
Field2.Y_train_noisy_tmp = Field2.un_normalize_points(Field2.Y_train_noisy_tmp, Field2.dict_norm_Y)

for i in range(Field2.field_param['dim_out']):
    if flag_unscale == True:
        mue_restack = np.copy(Field2.restack_prediction_list(list_dict_data[1]['list_mu']))
        std_restack = np.copy(Field2.restack_prediction_list(list_dict_data[1]['list_std']))
        tmp_mean = Field2.un_normalize_points(mue_restack, Field2.dict_norm_Y)
        tmp_std = Field2.un_normalize_std(std_restack, Field2.dict_norm_Y)
        mean2 = tmp_mean[i, :]
        std_deviation2 = tmp_std[i, :]
    else:
        mean2 = list_dict_data[1]['list_mu'][i]
        std_deviation2 = list_dict_data[1]['list_std'][i]

    tmp_mu = mean2.reshape(Field2.field_param['lim_num'], order='C')[tuple_slice2]
    tmp_std = std_deviation2.reshape(Field2.field_param['lim_num'], order='C')[tuple_slice2]

    low_conf = (tmp_mu - 2 * tmp_std).flatten()
    high_conf = (tmp_mu + 2 * tmp_std).flatten()
    tmp_field = Field2.Y_field_ordered[i,:].reshape(Field2.field_param['lim_num'], order='C')[tuple_slice2]
    tmp_positions = Field2.X_predict[flag_nonconstant, :].reshape(Field2.field_param['lim_num'], order='C')[tuple_slice2]
    tmp_training_X = Field2.X_train[flag_nonconstant, :]

    # Use vertical lines to indicate observations
    if flag_training_points_in_prediction_slice == True:
        for ii in range(tmp_training_X.shape[0]):
            ax[i,1].axvline(x=tmp_training_X[ii], linestyle='-', lw=para_lw, marker=None)
    ax[i,1].fill_between(tmp_positions.flatten(), low_conf, high_conf, color=list_colors_var[1], alpha=list_alpha[1])
    ax[i,1].plot(tmp_positions, tmp_mu, lw=para_lw, color=list_colors_mean[1], linestyle=list_line[1])
    if i == 0 and 1==0:
        ax[i,1].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':')
        #ax[i,1].plot(tmp_training_X, Field2.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
        #           markersize=8, marker='o', linestyle='None', label='observations')
    else:
        ax[i,1].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':')
        #ax[i,1].plot(tmp_training_X, Field2.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
        #           markersize=8, marker='o', linestyle='None')
    #fmin = np.min(low_conf)
    #list_fmin.append(fmin)
    #fmax = np.max(high_conf)
    #list_fmax.append(fmax)
    ax[-1,1].set_xlabel(r'$q_{0}$'.format(flag_nonconstant+1)+r' [m/s]')
    #ax[i,1].set_ylabel(r'$\ddot{q}$'+r'$_{0}$'.format(i+1))
#ax[2,1].set_xticks([0, 1, 2]

ax[0, 1].tick_params(axis='x', labelsize=10)
ax[0, 1].tick_params(axis='y', labelsize=10)
ax[1, 1].tick_params(axis='y', labelsize=10)
ax[2, 1].tick_params(axis='y', labelsize=10) # , rotation=90

ax[0,0].set_xticks([0, 2])
ax[0,1].set_xticks([0, 1])

ax[0,0].set_yticks([-0.4, 0, 0.4])
ax[0,1].set_yticks([-0.5, 0, 0.3])
ax[1,0].set_yticks([-0.4, 0, 0.4])
ax[1,1].set_yticks([-0.4, 0, 0.4])
ax[2,0].set_yticks([-10.2, -9.8, -9.4])
ax[2,1].set_yticks([-0.4, 0, 0.4])

ax[0,0].set_xlim(0, 2)
ax[0,1].set_xlim(0, 1)

ax[0,0].set_ylim(-0.45, 0.45)
ax[0,1].set_ylim(-0.55, 0.35)
ax[1,0].set_ylim(-0.45, 0.45)
ax[1,1].set_ylim(-0.45, 0.45)
ax[2,0].set_ylim(-10.25, -9.35)
ax[2,1].set_ylim(-0.45, 0.45)

ax[0,0].text(0.07, 0.8, r"surface ($v=0$)",
           transform=ax[0,0].transAxes, fontsize=10)
ax[0,1].text(0.07, 0.8, r"unicycle ($q_3=0$)",
           transform=ax[0,1].transAxes, fontsize=10)

for i in range(Field.field_param['dim_out']):
    for j in range(len(list_dict_data)):
        if i==0:
            handles, labels = ax[0,0].get_legend_handles_labels()
            #order = [1, 0, 4, 2, 3, 5]
            order = [1, 0, 2]
            ax[0,0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center',
                         ncol=3, bbox_to_anchor=(1.15, 1.5), frameon=False, prop={'size': 10}) #bbox_to_anchor=(0.7,1.3)
        #ax[i,0].set_ylim(min(list_fmin), max(list_fmax))
        #ax[i,0].grid(True)
    #plt.xlim(np.min(Field.X_predict[flag_nonconstant, :]), np.max(Field.X_predict[flag_nonconstant, :]))
    #plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.91, wspace=0.005)
    plt.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.9, wspace=0.3, hspace=0.1)
    #plt.tight_layout()
    #fig.tight_layout()
    plt.show()

