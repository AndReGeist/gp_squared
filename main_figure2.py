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
flag_ode = 'unicycle'
#flag_ode = 'duffling_oscillator'

flag_control = True
number_observations = 5
n_samp = 20 # 20

flag_training_points_in_prediction_slice = True
flag_savedata = False
flag_estimate_sys_params = False

flag = [8]  # 8
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
        #pass
        Field.field_param['lim_min'] = [1, 1, 0, 0, 0, 0]  # [0, 0, 0, -0.5, -1, -0.5]
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
        Field.field_param['lim_train_max'] = [2, 2, 0.01, 0.5, 0.5, 0.5, 5, 5, 5]

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
if any(elem in flag for elem in [8]):
    """
    Compute preditions for AMGP
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
    dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGPx,
                                         Gp_AMGP.covariance,
                                         Gp_AMGP.covariance, Gp_AMGP.covariance,
                                         Gp_AMGP.mean, flag_Lpred=True)
    list_dict_data.append(dict_post_AMGP)

################################
# Compute samples
################################
f_samples_prior = Gp_AMGP.sample_n(n_samp, Gp_AMGP.sample_prior, Field.X_predict, Gp_AMGP.mean, res_AMGPx, Gp_AMGP.covariance)
f_samples_posterior = Gp_AMGP.sample_n(n_samp, Gp_AMGP.sample_posterior, dict_post_AMGP['mu'], dict_post_AMGP['L_pred'])

print(Field.X_train)

################################
# Plot Samples
################################
flag_unscale = True
flag_plot_mean = False
flag_plot_all = False
plot_dimension = 0

para_lw = 2
list_slice = [slice(None)] * len(Field.field_param['lim_num'])
for i in range(len(list_slice)):
    if Field.field_param['lim_num'][i] == 1:
        list_slice[i] = 0
    else:
        flag_nonconstant = i
tuple_slice = tuple(list_slice)
list_fmin = []
list_fmax = []

if flag_unscale == True:
    Field.Y_field_ordered = Field.un_normalize_points(Field.Y_field_ordered, Field.dict_norm_Y)
    Field.Y_train_noisy_tmp = Field.un_normalize_points(Field.Y_train_noisy_tmp, Field.dict_norm_Y)
    Field.X_predict = Field.un_normalize_points(Field.X_predict, Field.dict_norm_X)
    Field.X_train = Field.un_normalize_points(Field.X_train, Field.dict_norm_X)

list_gpname = [r'GP$^2$']
list_alpha = [0.5]
list_colors_mean = ['blue']
list_colors_var = ['steelblue']
list_line = ['--', '-']
marker_color = 'turquoise'
sample_color = ['lightblue', 'steelblue']
field_param_lim_num_train = np.copy(Field.field_param['lim_num'])
for i in range(len(field_param_lim_num_train)):
    if field_param_lim_num_train[i] > 1:
        field_param_lim_num_train[i] = Field.field_param['number_observations']

for iii in range(len(f_samples_prior)):
    f_samples_prior[iii] = Field.un_normalize_points(f_samples_prior[iii], Field.dict_norm_Y)
for iii in range(len(f_samples_posterior)):
    f_samples_posterior[iii] = Field.un_normalize_points(f_samples_posterior[iii], Field.dict_norm_Y)

list_error1 = []
list_error2 = []
for iii in range(len(f_samples_prior)):
    list_error1.append(Field.compute_constraint_error(Field.X_predict, f_samples_prior[iii], Field.field_param['params']))
    list_error2.append(Field.compute_constraint_error(Field.X_predict, f_samples_posterior[iii], Field.field_param['params']))

print('max constraint error', np.max(list_error1), np.max(list_error2))
list_tmp = []
for i in range(len(list_error1)):
    list_tmp.append(np.max(list_error1[i]))
    list_tmp.append(np.max(list_error2[i]))
print('max constraint error', np.max(list_tmp))

if flag_plot_all == True:
    fig, ax = plt.subplots(Field.field_param['dim_out'], 1, figsize=(7, 9),
                           sharex=True)  # , sharex='col', sharey='row')
    fig.set_size_inches(12 / 2.54, 15 / 2.54 / 2)
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

            if flag_plot_mean == True:
                tmp_mu = mean.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
                tmp_std = std_deviation.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]

                low_conf = (tmp_mu - 2 * tmp_std).flatten()
                high_conf = (tmp_mu + 2 * tmp_std).flatten()
            tmp_field = Field.Y_field_ordered[i, :].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            tmp_positions = Field.X_predict[flag_nonconstant, :].reshape(Field.field_param['lim_num'], order='C')[
                tuple_slice]
            tmp_training_X = Field.X_train[flag_nonconstant, :]

            for iii in range(len(f_samples_prior)):
                ax[i].plot(tmp_positions, f_samples_prior[iii][i, :], color=sample_color[0])
            for iii in range(len(f_samples_posterior)):
                ax[i].plot(tmp_positions, f_samples_posterior[iii][i, :], color=sample_color[1])
            if flag_plot_mean == True:
                ax[i].fill_between(tmp_positions.flatten(), low_conf, high_conf, color=list_colors_var[j], alpha=list_alpha[j],
                                   label=list_gpname[j] + ' confidence')
                ax[i].plot(tmp_positions, tmp_mu, lw=para_lw, color=list_colors_mean[j], linestyle=list_line[j],
                           label=list_gpname[j] + ' mean')
            if i == 0 and j == 0:
                ax[i].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':', label='groundtruth')
                ax[i].plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
                           markersize=8, marker='o', linestyle='None', label='observations')
            else:
                ax[i].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':')
                ax[i].plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
                           markersize=8, marker='o', linestyle='None')
            #fmin = np.min(low_conf)
            #list_fmin.append(fmin)
            #fmax = np.max(high_conf)
            #list_fmax.append(fmax)
            ax[-1].set_xlabel(r'$q_{0}$'.format(flag_nonconstant + 1))
            ax[i].set_ylabel(r'$\ddot{q}$' + r'$_{0}$'.format(i + 1))

    for i in range(Field.field_param['dim_out']):
        for j in range(len(list_dict_data)):
            if i == 0:
                handles, labels = ax[0].get_legend_handles_labels()
                # order = [1, 0, 4, 2, 3, 5]
                order = [1, 0, 2, 3]
                # ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center',
                #             ncol=2, bbox_to_anchor=(0.5, 1.25), prop={'size': 10}) #bbox_to_anchor=(0.7,1.3)
            # ax[i].set_ylim(min(list_fmin), max(list_fmax))
            ax[i].grid(True)


if flag_plot_all == False:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4),
                           sharex=True)  # , sharex='col', sharey='row')
    fig.set_size_inches(12 / 2.54, 15 / 2.54 / 2)
    i = plot_dimension
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

        if flag_plot_mean == True:
            tmp_mu = mean.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            tmp_std = std_deviation.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]

            low_conf = (tmp_mu - 2 * tmp_std).flatten()
            high_conf = (tmp_mu + 2 * tmp_std).flatten()
        tmp_field = Field.Y_field_ordered[i, :].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
        tmp_positions = Field.X_predict[flag_nonconstant, :].reshape(Field.field_param['lim_num'], order='C')[
            tuple_slice]
        tmp_training_X = Field.X_train[flag_nonconstant, :]

        for iii in range(len(f_samples_prior)):
            if iii == 0:
                ax.plot(tmp_positions, f_samples_prior[iii][i, :], color=sample_color[0], label= r'GP$^2$ prior samples')
            else:
                ax.plot(tmp_positions, f_samples_prior[iii][i, :], color=sample_color[0])
        for iii in range(len(f_samples_posterior)):
            if iii == 0:
                ax.plot(tmp_positions, f_samples_posterior[iii][i, :], color=sample_color[1], label= r'GP$^2$ posterior samples')
            else:
                ax.plot(tmp_positions, f_samples_posterior[iii][i, :], color=sample_color[1])
        if flag_plot_mean == True:
            ax.fill_between(tmp_positions.flatten(), low_conf, high_conf, color=list_colors_var[j],
                               alpha=list_alpha[j],
                               label=list_gpname[j] + ' confidence')
            ax.plot(tmp_positions, tmp_mu, lw=para_lw, color=list_colors_mean[j], linestyle=list_line[j],
                       label=list_gpname[j] + ' mean')
        if i == 0 and j == 0:
            ax.plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':', label=r'$\ddot{q}$' + r'$_{0}$'.format(i + 1))
            ax.plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
                       markersize=8, marker='o', linestyle='None', label=r'$y$')
        else:
            ax.plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':')
            ax.plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
                       markersize=8, marker='o', linestyle='None')
        # fmin = np.min(low_conf)
        # list_fmin.append(fmin)
        # fmax = np.max(high_conf)
        # list_fmax.append(fmax)
        ax.set_xlabel(r'$q_{0}$'.format(flag_nonconstant + 1) + r' [rad]')
        ax.set_ylabel(r'$\ddot{q}$' + r'$_{0}$'.format(i + 1) + r' [m/s$^2$]')
        ax.set_xticks([0, 3, 6])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10, rotation=90)

    for j in range(len(list_dict_data)):
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            order = [2, 3, 0, 1]
            ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center',
                        ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.4), prop={'size': 10}) #bbox_to_anchor=(0.7,1.3)
        # ax.set_ylim(min(list_fmin), max(list_fmax))
        #ax.grid(True)
ax.text(0.03, 0.87, r"unicycle ($v=1.4 \mathrm{m}/\mathrm{s}$)",
           transform=ax.transAxes, fontsize=10)
plt.xlim(np.min(Field.X_predict[flag_nonconstant, :]), np.max(Field.X_predict[flag_nonconstant, :]))
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.98, top=0.8, wspace=0.005)
plt.show()

