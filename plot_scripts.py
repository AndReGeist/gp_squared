# Copyright 2019 Max Planck Society. All rights reserved.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Qt5Agg')  # Uncomment to disable pycharm SciView (easy to save figures)
#import seaborn as sns
#sns.set()
#sns.set_style("darkgrid", {"axes.facecolor": ".9"})

def matrix_show(matrix, title, vmin, vmax, cmap):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=cmap, vmax=vmax, vmin=vmin, origin='lower') #cmap='RdBu_r',
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[1]):
            c = round(matrix[j, i], 2)
            ax.text(i, j, str(c), va='center', ha='center')
    plt.title(title)
    plt.show()

def matrix_show2(matrix, title, vmin, vmax, cmap):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=cmap, vmax=vmax, vmin=vmin, origin='lower')  # cmap='RdBu_r',
    plt.colorbar(cax)
    plt.title(title)
    plt.show()

def matrix_show_n(matrix_list, title_list, vmin, vmax, cmap):
    fig, ax = plt.subplots(1, len(matrix_list))
    for j in range(len(matrix_list)):
        ax[j].matshow(matrix_list[j], cmap=cmap, vmax=vmax, vmin=vmin, origin='lower')  # cmap='RdBu_r',
        for i in range(matrix_list[j].shape[1]):
            for jj in range(matrix_list[j].shape[1]):
                c = round(matrix_list[j][jj, i], 2)
                ax[j].text(i, jj, str(c), va='center', ha='center')
        ax[j].title.set_text(title_list[j])
        ax[j].grid()
    plt.show()


def plot_field_slice(list_dict_data, field, X_train):
    nx = field.list_dim_num[0]
    index = int((nx - 1) / 2)
    X_position = field.positions[0, index * nx:(index + 1) * nx]

    fig, ax = plt.subplots(3, len(list_dict_data))  # , sharex='col', sharey='row')
    # axes are in a two-dimensional array, indexed by [row, col]
    for j in range(len(list_dict_data)):
        mu_x_tmp = list_dict_data[j]['mu_x'].reshape(nx, nx)[index, :]
        std_x_tmp = list_dict_data[j]['std_x'].reshape(nx, nx)[index, :]
        ax[0, j].fill_between(X_position, mu_x_tmp - 2 * std_x_tmp, mu_x_tmp + 2 * std_x_tmp, color='salmon')
        ax[0, j].plot(X_position, mu_x_tmp, color='red')
        ax[0, j].plot(X_position, field.u[index, :], color='green')
        for xc in X_train[0, :]:
            ax[0, j].axvline(x=xc)
        ax[0, j].grid()
        ax[0, j].set_ylabel(r"$f_1 = \dot{\theta}$ (rad/s)")
        ax[0, j].set_xlabel(r"$x_1 = \theta$ (rad)")
        ax[0, j].title.set_text(list_dict_data[j]['gp_type'] + ': ' + 'f1 at field slice y close to 0')

        mu_y_tmp = list_dict_data[j]['mu_y'].reshape(nx, nx)[index, :]
        std_y_tmp = list_dict_data[j]['std_y'].reshape(nx, nx)[index, :]
        ax[1, j].fill_between(X_position, mu_y_tmp - 2 * std_y_tmp, mu_y_tmp + 2 * std_y_tmp, color='salmon')
        ax[1, j].plot(X_position, mu_y_tmp, color='red')
        ax[1, j].plot(X_position, field.v[index, :], color='green')
        for xc in X_train[0, :]:
            ax[1, j].axvline(x=xc)
        ax[1, j].grid()
        ax[1, j].set_ylabel(r"$f_2 = \ddot{\theta}$ (rad/s)")
        ax[1, j].set_xlabel(r"$x_1 = \theta$ (rad)")
        ax[1, j].title.set_text(list_dict_data[j]['gp_type'] + ': ' + 'f2 at field slice y close to 0')

        if 'mu_z' in list_dict_data[j]:
            print('check')
            mu_z_tmp = list_dict_data[j]['mu_z'].reshape(nx, nx)[index, :]
            std_z_tmp = list_dict_data[j]['std_z'].reshape(nx, nx)[index, :]
            ax[2, j].fill_between(X_position, mu_z_tmp - 2 * std_z_tmp, mu_z_tmp + 2 * std_z_tmp, color='salmon')
            ax[2, j].plot(X_position, mu_z_tmp, color='red')
            ax[2, j].plot([X_position[0], X_position[-1]], [-0.5, -0.5], color='green')
            for xc in X_train[0, :]:
                ax[2, j].axvline(x=xc)
            ax[2, j].grid()
            ax[2, j].set_ylabel(r"$f_2 = \ddot{\theta}$ (rad/s)")
            ax[2, j].set_xlabel(r"$x_1 = \theta$ (rad)")
            ax[2, j].title.set_text(list_dict_data[j]['gp_type'] + ': ' + 'fc at field slice y close to 0')
    plt.show()

def compute_constraint_error(Field, dict_GP_post, constraint_params):
    numb_points = len(dict_GP_post['list_mu'][0])
    error_constraint = np.zeros((numb_points, 1))
    value_Af = np.zeros((numb_points, 1))
    value_b = np.zeros((numb_points, 1))
    acci = np.zeros((Field.field_param['dim_out'], 1))
    if Field.field_param['flag_normalize_in'] == True:
        X_predict = Field.un_normalize_points(Field.X_predict, Field.dict_norm_X)
    else:
        X_predict = Field.X_predict
    Field.check_if_states_fulfill_constraint(X_predict)

    for i in range(numb_points):
        for j in range(len(dict_GP_post['list_mu'])):
            if Field.field_param['flag_normalize_out'] == True:
                acci[j] = (Field.dict_norm_Y['N_std_inv'][j,j] * dict_GP_post['list_mu'][j][i]) + Field.dict_norm_Y['N_mue'][j]
            else:
                acci[j] = dict_GP_post['list_mu'][j][i]
            A = Field.constraint_A(X_predict[:, i], constraint_params)
            b = Field.constraint_b(X_predict[:, i], constraint_params)
            value_Af[i] = A.dot(acci)
            value_b[i] = b
            error_constraint[i] = np.abs(value_Af[i] - value_b[i])
    return error_constraint

def check_constraint_n(Field, dict_GP_post, path, flag_save=False):
    error_constraint = compute_constraint_error(Field, dict_GP_post, Field.field_param['params'])
    plt.figure()
    plt.title('title')
    plt.plot(error_constraint, markersize=10, marker='s', linestyle='None', label='Af - b')
    #plt.plot(value_Af, markersize=8, marker='D', linestyle='None', label='Af')
    #plt.plot(value_b, markersize=6, marker='o', linestyle='None', label='b')
    plt.legend(loc='upper right')
    plt.xlabel('input point number')
    plt.ylabel('output error values')
    if flag_save==True:
        plt.savefig(path + '/' + 'constraint' + '_' + dict_GP_post['gp_type'] + '_' + str(Field.field_param['lim_min']) + '_' + str(Field.field_param['lim_min']) + '.png', bbox_inches='tight')
    else:
        plt.show()

def plot_output(list_dict_data, Field, path, flag_save):
    dim_out = Field.field_param['dim_out']
    # axes are in a two-dimensional array, indexed by [row, col]
    max_prediction_error = 0

    ground_truth = Field.Y_field_ordered
    #if flag_unconstrained == 'true':
    #    ground_truth[0,:] = 0
    #    ground_truth[1, :] = 9.81
    if len(list_dict_data) == 1:
        list_dict_data.append(list_dict_data[0])

    fig, ax = plt.subplots(dim_out+1, len(list_dict_data), sharex='row')  # , sharex='col', sharey='row')
    for j in range(len(list_dict_data)):
        for i in range(len(list_dict_data[j]['list_mu'])):
            if i == 0:
                ax[0, j].title.set_text(list_dict_data[j]['gp_type'])
            ax[i, j].plot(ground_truth[i, :], markerfacecolor='blue', markersize=10, marker='s', linestyle='None', label='func '+str(i))

            #if np.array_equal(Field.X_train, Field.positions):
            #    ax[i, j].plot(Field.Y_train_noisy_ordered[i, :], markerfacecolor='none', markeredgecolor='red', markersize=10, marker='s', linestyle='None', label='noisy func '+str(i))
            ax[i, j].plot(list_dict_data[j]['list_mu'][i], markerfacecolor='orange', markeredgecolor='orange', marker='o', linestyle='None', label='mean '+str(i))
            ax[i, j].errorbar(np.arange(Field.X_predict.shape[1]), list_dict_data[j]['list_mu'][i],
                              2*list_dict_data[j]['list_std'][i], linestyle='None', color='orange', label='var '+str(i))

            #ax[i, j].plot(Field.indeces, Field.Y_train_noisy_ordered[i,:], markerfacecolor='none', markeredgecolor='red', markersize=10, marker='s', linestyle='None', label='observations')

            ax[i, j].grid()
            ax[i, j].legend(loc='upper right')
            ax[i, j].set_ylabel('output function value' + 'Nr: ' + str(i+1))
            ax[i, j].set_xlabel('input point number')
            prediction_error = np.abs(list_dict_data[j]['list_mu'][i] - ground_truth[i, :].reshape(-1, 1))
            emax = np.max(prediction_error)
            if emax > max_prediction_error:
                max_prediction_error = emax
            ax[dim_out, j].plot(prediction_error, marker='o', linestyle='None', label='error '+str(i))
            ax[dim_out, j].legend(loc='upper right')
            ax[dim_out, j].set_ylabel('output error values')
            ax[dim_out, j].set_xlabel('input point number')

    for j in range(len(list_dict_data)):
        ax[dim_out, j].set_ylim([0, max_prediction_error])

    if flag_save==True:
        plt.savefig(path + '/plot_output' + '.png',
                    bbox_inches='tight')
    else:
        plt.show()

def plot_slice(list_dict_data, Field, path, flag_save=False):
    '''
    Plots both pendulum accelerations over pendulum angle at slice x_dot = dim_min[2] and y_dot = dim_min[3]
    Required settings: 'dim_num': [NUMBER, NUMBER, 1, 1]

    :param list_dict_data:
    :param Field:
    :param flag_unconstrained:
    :return:
    '''
    para_lw = 2

    if len(list_dict_data) == 1:
        list_dict_data.append(list_dict_data[0])

    list_slice = [slice(None)] * len(Field.field_param['lim_num'])
    for i in range(len(list_slice)):
        if Field.field_param['lim_num'][i] == 1:
            list_slice[i] = 0
        else:
            flag_nonconstant = i
    tuple_slice = tuple(list_slice)
    list_emax = []
    list_fmin = []
    list_fmax = []

    fig, ax = plt.subplots(Field.field_param['dim_out'], len(list_dict_data), sharex=True)  # , sharex='col', sharey='row')
    for i in range(Field.field_param['dim_out']):
        for j in range(len(list_dict_data)):
            if i == 0:
                ax[0, j].title.set_text(list_dict_data[j]['gp_type'])
            mean = list_dict_data[j]['list_mu'][i]
            std_deviation = list_dict_data[j]['list_std'][i]

            tmp_mu = mean.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            tmp_std = std_deviation.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]

            low_conf = (tmp_mu - 2 * tmp_std).flatten()
            high_conf = (tmp_mu + 2 * tmp_std).flatten()
            tmp_field = Field.Y_field_ordered[i,:].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            # if Field.field_param['ode_flag'] == 'unicycle_without_control' or Field.field_param['ode_flag'] == 'unicycle_controlled':
            #     # Reducing the state space to the actual dimensions important for plotting
            #     X_predict_new = np.zeros((Field.field_param['dim_in']-, Field.X_predict.shape[1]))
            #     X_predict_new[0, :] = np.sqrt(Field.X_predict[0,:]**2 + Field.X_predict[1,:]**2)
            #     X_predict_new[1:, :] = Field.X_predict[2:,:]
            # if Field.field_param['ode_flag'] == 'mass_surface':
            #     X_predict_new = Field.X_predict
            tmp_positions = Field.X_predict[flag_nonconstant, :].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            ax[i, j].plot(Field.X_train[flag_nonconstant, :], Field.Y_train_noisy_tmp[i, :], color='red', markeredgecolor='black',
                          markersize=4, marker='o', linestyle='None', label='observations')
            ax[i, j].fill_between(tmp_positions.flatten(), low_conf, high_conf, color='skyblue', label='GP confidence')
            ax[i, j].plot(tmp_positions, tmp_mu, lw=para_lw, color='blue', label='GP mean')
            ax[i, j].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':', label='groundtruth')

            #ax[i, j].set_ylabel(str(i)+'-th output dimension')
            ax[i, j].grid()
            #for i in range(len(Field.field_param_old['lim_num'])):
            #    if Field.field_param_old['lim_num'][i] == Field.field_param_old['lim_num']
            #    ax[i, j].plot()
            fmin = np.min(low_conf)
            list_fmin.append(fmin)
            fmax = np.max(high_conf)
            list_fmax.append(fmax)

    for i in range(Field.field_param['dim_out']):
        for j in range(len(list_dict_data)):
            #ax[i, j].set_ylim(min(list_fmin), max(list_fmax))
            #ax[-1, j].set_ylim(0, max(list_emax))
            pass
    plt.xlim(np.min(Field.X_predict[flag_nonconstant, :]), np.max(Field.X_predict[flag_nonconstant, :]))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85, wspace=0.12, hspace=0.1)
    plt.tight_layout()

    if flag_save==True:
        plt.savefig(path + '/slice_' + str(flag_nonconstant) + str(Field.field_param['lim_min']) + '.png',
                    bbox_inches='tight')

    fig, ax = plt.subplots(1, len(list_dict_data), sharey=True)  # , sharex='col', sharey='row')
    for i in range(Field.field_param['dim_out']):
        for j in range(len(list_dict_data)):
            mean = list_dict_data[j]['list_mu'][i]
            tmp_mu = mean.reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            tmp_field = Field.Y_field_ordered[i, :].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            prediction_error = np.abs(tmp_field - tmp_mu).reshape((-1, 1))
            ax[j].plot(tmp_positions, prediction_error, marker='o', linestyle='None', label='error ' + str(i))
            ax[j].legend(loc='upper right')
            ax[j].set_ylabel('output error values')
            ax[j].set_xlabel('input dimension')
            ax[j].grid()
            emax = np.max(prediction_error)
            list_emax.append(emax)
    plt.tight_layout()
    if flag_save==True:
        plt.savefig(path + '/error_' + str(flag_nonconstant) + str(Field.field_param['lim_min']) + '.png', bbox_inches='tight')
    else:
        plt.show()

def plot_slice_L4DC(list_dict_data, Field, path, flag_unscale=True, flag_unconstrained=False, 
                    samples_prior=None, samples_posterior=None):
    '''
    Plots both pendulum accelerations over pendulum angle at slice x_dot = dim_min[2] and y_dot = dim_min[3]
    Required settings: 'dim_num': [NUMBER, NUMBER, 1, 1]

    :param list_dict_data:
    :param Field:
    :param flag_unconstrained:
    :return:
    '''
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
    fig, ax = plt.subplots(Field.field_param['dim_out'], 1, figsize=(7,9), sharex=True)  # , sharex='col', sharey='row')
    #fig.set_size_inches(12/2.54, 15/2.54)
    fig.set_size_inches(12 / 2.54, 15 / 2.54/2)

    if flag_unscale == True and flag_unconstrained == False:
        Field.Y_field_ordered = Field.un_normalize_points(Field.Y_field_ordered, Field.dict_norm_Y)
        Field.Y_train_noisy_tmp = Field.un_normalize_points(Field.Y_train_noisy_tmp, Field.dict_norm_Y)
        Field.X_predict = Field.un_normalize_points(Field.X_predict, Field.dict_norm_X)
        Field.X_train = Field.un_normalize_points(Field.X_train, Field.dict_norm_X)

    if flag_unscale == True and flag_unconstrained == True:
        Field.X_predict = Field.un_normalize_points(Field.X_predict, Field.dict_norm_X)
        Field.X_train = Field.un_normalize_points(Field.X_train, Field.dict_norm_X)
        Field.Y_field_ordered = Field.define_observations(Field.X_predict, flag_unconstrained='a_bar').reshape((-1, Field.field_param['dim_out'])).T
        Field.Y_train_noisy_tmp = Field.un_normalize_points(Field.Y_train_noisy_tmp, Field.dict_norm_Y)

    if flag_unconstrained == False:
        list_gpname = ['SE-ARD', r'GP$^2$']
        list_alpha = [0.7, 0.5]
        list_colors_mean = ['gray', 'blue']
        list_colors_var = ['lightgray', 'steelblue']
        list_line = ['--', '-']
    else:
        list_gpname = [r'GP$^2$']
        list_alpha = [0.5]
        list_colors_mean = ['blue']
        list_colors_var = ['steelblue']
        list_line = ['--', '-']
    marker_color = 'turquoise'
    prior_sample_color = ['lightblue', 'steelblue']
    field_param_lim_num_train = np.copy(Field.field_param['lim_num'])
    for i in range(len(field_param_lim_num_train)):
        if field_param_lim_num_train[i] > 1:
            field_param_lim_num_train[i] = Field.field_param['number_observations']

    if samples_prior is not None:
        for iii in range(len(samples_prior)):
            samples_prior[iii] = Field.un_normalize_points(samples_prior[iii], Field.dict_norm_Y)
    if samples_posterior is not None:
        for iii in range(len(samples_posterior)):
            samples_posterior[iii] = Field.un_normalize_points(samples_posterior[iii], Field.dict_norm_Y)

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
            tmp_field = Field.Y_field_ordered[i,:].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            tmp_positions = Field.X_predict[flag_nonconstant, :].reshape(Field.field_param['lim_num'], order='C')[tuple_slice]
            tmp_training_X = Field.X_train[flag_nonconstant, :]

            if flag_unconstrained == True:
                # Use vertical lines to indicate observations
                for ii in range(tmp_training_X.shape[0]):
                    ax[i].axvline(x=tmp_training_X[ii], linestyle=':', marker=None)
            if samples_prior != None:
                for iii in range(len(samples_prior)):
                    ax[i].plot(tmp_positions, samples_prior[iii][i,:], color=prior_sample_color[0])
            if samples_posterior != None:
                for iii in range(len(samples_posterior)):
                    ax[i].plot(tmp_positions, samples_posterior[iii][i, :], color=prior_sample_color[1])
            ax[i].fill_between(tmp_positions.flatten(), low_conf, high_conf, color=list_colors_var[j], alpha=list_alpha[j],label=list_gpname[j] + ' confidence')
            ax[i].plot(tmp_positions, tmp_mu, lw=para_lw, color=list_colors_mean[j], linestyle=list_line[j], label=list_gpname[j] + ' mean')
            if i == 0 and j==0:
                ax[i].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':', label='groundtruth')
                ax[i].plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
                           markersize=8, marker='o', linestyle='None', label='observations')
            else:
                ax[i].plot(tmp_positions, tmp_field, lw=para_lw, color='black', linestyle=':')
                ax[i].plot(tmp_training_X, Field.Y_train_noisy_tmp[i, :], color=marker_color, markeredgecolor='black',
                           markersize=8, marker='o', linestyle='None')
            fmin = np.min(low_conf)
            list_fmin.append(fmin)
            fmax = np.max(high_conf)
            list_fmax.append(fmax)
            ax[-1].set_xlabel(r'$q_{0}$'.format(flag_nonconstant+1))
            ax[i].set_ylabel(r'$\ddot{q}$'+r'$_{0}$'.format(i+1))

    for i in range(Field.field_param['dim_out']):
        for j in range(len(list_dict_data)):
            if i==0:
                handles, labels = ax[0].get_legend_handles_labels()
                #order = [1, 0, 4, 2, 3, 5]
                order = [1, 0, 2, 3]
                #ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center',
                #             ncol=2, bbox_to_anchor=(0.5, 1.25), prop={'size': 10}) #bbox_to_anchor=(0.7,1.3)
            #ax[i].set_ylim(min(list_fmin), max(list_fmax))
            ax[i].grid(True)
    plt.xlim(np.min(Field.X_predict[flag_nonconstant, :]), np.max(Field.X_predict[flag_nonconstant, :]))
    #plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.91, wspace=0.005)
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95, wspace=0.005)
    #plt.tight_layout()
    #fig.tight_layout()
    plt.show()

def plot_comparison_old(list_dict_data, list_datapoints):
    '''
    Plots both pendulum accelerations over pendulum angle at slice x_dot = dim_min[2] and y_dot = dim_min[3]
    Required settings: 'dim_num': [NUMBER, NUMBER, 1, 1]

    :param list_dict_data:
    :param Field:
    :param flag_unconstrained:
    :return:
    '''
    para_lw = 3
    list_fmin = []
    list_fmax = []
    fig, ax = plt.subplots(1, 3, figsize=(12,6))  # , sharex='col', sharey='row'), sharex=True

    #list_colors_mean = ['green', 'blue']
    #list_colors_var = ['lightgreen', 'steelblue']
    #list_colors_mean = ['lightgreen', 'blue']
    #list_colors_var = ['green', 'blue']
    list_alpha = [0.2, 0.3]
    list_gpname = ['Indep. GPs', 'Constr. Indep. GPs']
    list_colors_mean = ['firebrick', 'blue']
    list_colors_var = ['salmon', 'steelblue']
    list_ylabels = [r'mean RMSE (m/s$^2$)', 'std. dev. RMS (m/s$^2$)', 'constraint RMSE']
    list_GP_labels = []

    for i in range(3):
        for j in range(2):
            mean = list_dict_data[j][i].mean(axis=1)
            std_deviation = list_dict_data[j][i].std(axis=1)
            low_conf = mean - std_deviation
            high_conf = mean + std_deviation

            ax[i].fill_between(list_datapoints, low_conf, high_conf, color=list_colors_var[j], alpha=list_alpha[j],label=list_gpname[j])
            ax[i].plot(list_datapoints, mean, color=list_colors_mean[j], markeredgecolor='black', markersize=8, marker='o', label=list_gpname[j])
            if i == 2 and j==1:
                mean2 = list_dict_data[1][-1].mean(axis=1)
                std_deviation2 = list_dict_data[1][-1].std(axis=1)
                low_conf2 = mean2 - std_deviation2
                high_conf2 = mean2 + std_deviation2
                ax[i].fill_between(list_datapoints, low_conf2, high_conf2, color='green', alpha=0.3,
                                   label=list_gpname[j])
                ax[i].plot(list_datapoints, mean2, color='green', markeredgecolor='black', markersize=8,
                           marker='o', label='Constr. Indep. GPs (Estim. constraint)')

            fmin = np.min(low_conf)
            list_fmin.append(fmin)
            fmax = np.max(high_conf)
            list_fmax.append(fmax)
            ax[i].set_xlabel('# training points')
            ax[i].set_ylabel(list_ylabels[i])
            ax[i].grid(True)

    #for i in range(3):
    #        ax[i].set_ylim(min(list_fmin), max(list_fmax))
    #plt.legend(loc='upper center', ncol=2)  # bbox_to_anchor=(0.7,1.3)
    #plt.xlim(np.min(Field.X_predict_tmp[flag_nonconstant, :]), np.max(Field.X_predict_tmp[flag_nonconstant, :]))
    #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.9)
    plt.tight_layout()
    plt.show()

def plot_data_comparison(array, list_datapoints, flag_ode, flag):
    '''

    '''
    para_lw = 3
    list_fmin = []
    list_fmax = []

    #list_colors_mean = ['green', 'blue']
    #list_colors_var = ['lightgreen', 'steelblue']
    #list_colors_mean = ['lightgreen', 'blue']
    #list_colors_var = ['green', 'blue']
    list_alpha = [0.2, 0.3]
    list_gpname = ['SE-ARD', 'ICM', 'LMC', r'GP$^2$']
    list_marker = ['s', '^', 'v', 'o']
    list_ylabels = [r'mean RMSE (m/s$^2$)', 'std. dev. RMS (m/s$^2$)', 'constraint RMSE']
    list_GP_labels = []
    list_flag_ode = ['surface', 'unicycle', 'duffing']

    fig, ax = plt.subplots(1, 3, figsize=(12,6))  # , sharex='col', sharey='row'), sharex=True, sharey=True
    for xx in range(len(flag_ode)):
        for pp in range(len(flag)):
            mean = array[xx][pp][:,0]
            low_conf = array[xx][pp][:,1]
            high_conf = array[xx][pp][:,2]
            asymmetric_error = [low_conf, high_conf]
            # color=list_colors_mean[pp], color=list_colors_var[j],
            ax[xx].fill_between(list_datapoints, low_conf, high_conf, alpha=0.3) # label=list_gpname[pp]
            #ax[xx].errorbar(list_datapoints, mean, yerr=asymmetric_error, label=list_gpname[xx])
            ax[xx].plot(list_datapoints, mean, markeredgecolor='black', markersize=5, marker=list_marker[pp], label=list_gpname[pp])
            #if xx == 0 and pp ==0:

            fmin = np.min(low_conf)
            list_fmin.append(fmin)
            fmax = np.max(high_conf)
            list_fmax.append(fmax)
            ax[xx].set_xticks(list_datapoints[0::2])
            #ax[xx].set_ylabel(list_ylabels[i])
            ax[xx].text(.5, .9, list_flag_ode[xx],
                    horizontalalignment='center',
                    transform=ax[xx].transAxes)
            ax[xx].grid(True)
            ax[xx].tick_params(axis='x', labelsize=10)
            ax[xx].tick_params(axis='y', labelsize=10, rotation=90)

    ax[0].set_ylabel('mean RMSE')
    ax[1].set_xlabel(r'$\#$ training points')
    ax[2].legend(loc='center right', ncol=1, prop={'size': 10})  # bbox_to_anchor=(0.7,1.3)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.97, wspace= 0.3, hspace = 1)

    fig, ax = plt.subplots(1, 3, figsize=(12,6))  # , sharex='col', sharey='row'), sharex=True, sharey=True
    for xx in range(len(flag_ode)):
        for pp in range(len(flag)):
            mean = array[xx][pp][:,3]
            low_conf = array[xx][pp][:,4]
            high_conf = array[xx][pp][:,5]
            asymmetric_error = [low_conf, high_conf]
            # color=list_colors_mean[pp], color=list_colors_var[j],
            ax[xx].fill_between(list_datapoints, low_conf, high_conf, alpha=0.3) # label=list_gpname[pp]
            #ax[xx].errorbar(list_datapoints, mean, yerr=asymmetric_error, label=list_gpname[xx])
            ax[xx].plot(list_datapoints, mean, markeredgecolor='black', markersize=5, marker=list_marker[pp], label=list_gpname[pp])
            #if xx == 0 and pp ==0:

            fmin = np.min(low_conf)
            list_fmin.append(fmin)
            fmax = np.max(high_conf)
            list_fmax.append(fmax)
            ax[xx].set_xticks(list_datapoints[0::2])
            #ax[xx].set_ylabel(list_ylabels[i])
            ax[xx].text(.5, .9, list_flag_ode[xx],
                    horizontalalignment='center',
                    transform=ax[xx].transAxes)
            ax[xx].grid(True)
            #ax[xx].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
            ax[xx].tick_params(axis='x', labelsize=10)
            ax[xx].tick_params(axis='y', labelsize=10, rotation=90)

    ax[0].set_ylabel('constraint error')
    ax[1].set_xlabel(r'$\#$ training points')
    ax[2].legend(loc='center right', ncol=1, prop={'size': 10})  # bbox_to_anchor=(0.7,1.3)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.97, wspace= 0.3, hspace = 1)
    plt.show()

def plot_uncosntrained_acceleration_pendulum(list_dict_data, Field):
    dim_out = Field.field_param['dim_out']
    fig, ax = plt.subplots(2,1, sharex='col')  # , sharex='col', sharey='row')
    # axes are in a two-dimensional array, indexed by [row, col]
    max_prediction_error = 0

    ground_truth = Field.Y_field_ordered
    # if flag_unconstrained == 'true':
    #    ground_truth[0,:] = 0
    #    ground_truth[1, :] = 9.81
    angle = np.arctan2(Field.positions[0,:], Field.positions[1,:])
    angle = (180/np.pi)*angle
    index_angle = angle.argsort()
    #plt.rc('text', usetex=True)

    #plt.rcParams['font.family'] = 'Times New Roman'
    para_lw = 2
    # AXIS [0,0]
    for i in range(len(list_dict_data[0]['list_mu'])):
        ax[i].plot(angle[index_angle], list_dict_data[0]['list_mu'][i][index_angle], linestyle='-', color='darkred', label='GP posterior mean', lw=3)#marker='o', markerfacecolor='blue', markeredgecolor='blue',
        low_conf = (list_dict_data[0]['list_mu'][i][index_angle] - list_dict_data[0]['list_std'][i][
            index_angle]**2).flatten()
        high_conf = (list_dict_data[0]['list_mu'][i][index_angle] + list_dict_data[0]['list_std'][i][
            index_angle]**2).flatten()
        ax[i].fill_between(angle[index_angle], low_conf, high_conf, color='skyblue',
                              label='GP posterior variance')

        ax[i].grid()
        ax[0].set_ylabel(r'$\bar{a}_1$ (m/s$^2$)')
        ax[1].set_ylabel(r'$\bar{a}_2$ (m/s$^2$)')
        ax[1].set_xlabel('pendulum angle ($^\circ$)')
    ax[0].plot([-180, 180], [0, 0], color='black', linestyle='--',
               label='unconstrained acc.',lw=2)
    ax[1].plot([-180, 180], [9.81, 9.81], color='black', linestyle='--',
               label='unconstrained acc.',lw=2)
    ax[0].legend(loc='lower right', ncol=2)
    ax[1].legend(loc='lower right', ncol=2)
    plt.xlim(-180, 180)
    ax[1].set_ylim(0, 12)
    plt.subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.98, wspace= 0.03)
    plt.show()

def plot_DataEFF(Indep_list_theta, Indep_list_MLE, Indep_list_RMSE, \
                     AMGP_list_theta, AMGP_list_MLE, AMGP_list_RMSE,\
                     number_runs, list_datapoints):
    from pylab import text
    # Create an axes instance
    fig, ax = plt.subplots(1, 2, sharey='row') # sharey=True  # , sharex='col', sharey='row')

    # Create the boxplot
    bp1 = ax[0].boxplot(Indep_list_RMSE, patch_artist=True, labels=list_datapoints)
    bp2 = ax[1].boxplot(AMGP_list_RMSE, patch_artist=True, labels=list_datapoints)
    #ax[0].set_title('Independent GPs')
    #ax[1].set_title('UK-GP')

    ## change outline color, fill color and linewidth of the boxes
    for box in bp1['boxes']:
        # change outline color
        box.set(color='skyblue')
        box.set(edgecolor='black', linewidth=0.5)
        # change fill color
        #box.set_facecolor('#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp1['whiskers']:
        whisker.set(color='black', linewidth=1)

    ## change color and linewidth of the caps
    for cap in bp1['caps']:
        cap.set(color='black', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp1['medians']:
        median.set(color='darkred', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp1['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp2['boxes']:
        # change outline color
        box.set(color='skyblue')
        box.set(edgecolor='black', linewidth=0.5)
        # change fill color
        #box.set(facecolor='#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp2['whiskers']:
        whisker.set(color='black', linewidth=1)

    ## change color and linewidth of the caps
    for cap in bp2['caps']:
        cap.set(color='black', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp2['medians']:
        median.set(color='darkred', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp2['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ax[0].yaxis.grid(True)
    ax[0].set_xlabel('number of observations')
    ax[0].set_ylabel('Prediction RMSE')
    ax[1].yaxis.grid(True)
    ax[1].set_xlabel('number of observations')
    #ax[1].set_ylabel('Prediction RMSE')
    text(0.78, 0.94, 'Independent GPs', ha='center', va='center', transform=ax[0].transAxes)
    text(0.9, 0.94, 'UK-GP', ha='center', va='center', transform=ax[1].transAxes)
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.97, wspace= 0.05, hspace= 0.6)
    plt.show()

    def compute_GP_updates(Field, field_param):
        if res_Indep != [None]:
            Gp_Indep = class_GP(field_param)
        if res_AMGP != [None]:
            if field_param['flag_normalize_in'] == True or Field.field_param['flag_normalize_out'] == True:
                Gp_AMGP = subclass_AMGP_normalized(field_param, constraint_A, constraint_b, func_M)
            else:
                Gp_AMGP = subclass_AMGP(field_param, constraint_A, constraint_b, func_M)

        if field_param['flag_lengthscales'] == 'same' and field_param['flag_signalvar'] == 'same':
            if res_Indep != [None]:
                dict_post_Indep = Gp_Indep.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_Indep.x,
                                                       Gp_Indep.covariance_same_same, Gp_Indep.covariance_same_same,
                                                       Gp_Indep.covariance_same_same, Gp_Indep.mean, Field)
            if res_AMGP != [None]:
                dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy,
                                                     res_AMGP.x, Gp_AMGP.covariance_ukgp_same_same,
                                                     Gp_AMGP.covariance_ukgp_same_same,
                                                     Gp_AMGP.covariance_ukgp_same_same, Gp_AMGP.mean, Field)
        if field_param['flag_lengthscales'] == 'all' and field_param['flag_signalvar'] == 'same':
            if res_Indep != [None]:
                dict_post_Indep = Gp_Indep.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_Indep.x,
                                                       Gp_Indep.covariance_all_same, Gp_Indep.covariance_all_same,
                                                       Gp_Indep.covariance_all_same, Gp_Indep.mean, Field)
            if res_AMGP != [None]:
                dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy,
                                                     res_AMGP.x, Gp_AMGP.covariance_ukgp_all_same,
                                                     Gp_AMGP.covariance_ukgp_all_same,
                                                     Gp_AMGP.covariance_ukgp_all_same, Gp_AMGP.mean, Field)
        if field_param['flag_lengthscales'] == 'same' and field_param['flag_signalvar'] == 'all':
            if res_Indep != [None]:
                dict_post_Indep = Gp_Indep.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_Indep.x,
                                                       Gp_Indep.covariance_same_all, Gp_Indep.covariance_same_all,
                                                       Gp_Indep.covariance_same_all, Gp_Indep.mean, Field)
            if res_AMGP != [None]:
                dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy,
                                                     res_AMGP.x, Gp_AMGP.covariance_ukgp_same_all,
                                                     Gp_AMGP.covariance_ukgp_same_all,
                                                     Gp_AMGP.covariance_ukgp_same_all, Gp_AMGP.mean, Field)
        if field_param['flag_lengthscales'] == 'all' and field_param['flag_signalvar'] == 'all':
            if res_Indep != [None]:
                dict_post_Indep = Gp_Indep.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_Indep.x,
                                                       Gp_Indep.covariance_all_all, Gp_Indep.covariance_all_all,
                                                       Gp_Indep.covariance_all_all, Gp_Indep.mean, Field)
            if res_AMGP != [None]:
                dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy,
                                                     res_AMGP.x, Gp_AMGP.covariance_ukgp_all_all,
                                                     Gp_AMGP.covariance_ukgp_all_all,
                                                     Gp_AMGP.covariance_ukgp_all_all, Gp_AMGP.mean, Field)

        # UNSCALE DATA
        if field_param['flag_normalize_in'] == True:
            Field.X_predict = Field.un_normalize_points(Field.X_predict, Field.dict_norm_X)
        if field_param['flag_normalize_out'] == True:
            Field.Y_predict = Field.un_normalize_points(Field.Y_field_ordered, Field.dict_norm_Y)

        if res_Indep != [None] and res_AMGP != [None]:
            list_dict_data = [dict_post_Indep, dict_post_AMGP]
        elif res_AMGP != [None]:
            list_dict_data = [dict_post_AMGP, dict_post_AMGP]
        elif res_Indep != [None]:
            list_dict_data = [dict_post_Indep, dict_post_Indep]

def analyse_model(Field, Gp_AMGP, Gp_Indep, filename, number_observations=10, noise_std=0,
                  res_AMGP=None, res_Indep=None, flag_save=False):

    list_lim_min = [[], [], [], []]
    list_lim_num = [[], [], [], []]
    list_train_lim_max = [[], [], [], []]
    if Field.field_param['ode_flag'] == 'unicycle' and Field.field_param['flag_control'] is False:
        list_lim_min[0] = [0, 1, 0]
        list_lim_num[0] = [1, 100, 1]
        list_lim_min[1] = [1, 1, 0.5]
        list_lim_num[1] = [1, 100, 1]
        list_lim_min[2] = [0, 3, -0.3]
        list_lim_num[2] = [100, 1, 1]
    if Field.field_param['ode_flag'] == 'unicycle' and Field.field_param['flag_control'] is True:
        list_train_lim_max[0] = Field.compute_points(np.array([0.5, 1, 2 * np.pi, 0.5, 1, 0.5]).reshape(-1,1)).flatten().tolist()
        list_lim_num[0] = [100, 1, 1, 1, 1, 1]
        list_train_lim_max[1] = Field.compute_points(np.array([0.5, 1, 2 * np.pi, 0.5, 1, 0.5]).reshape(-1,1)).flatten().tolist()
        list_lim_num[1] = [100, 1, 1, 1, 1, 1]
        list_train_lim_max[2] = Field.compute_points(np.array([1, 1, np.pi, 0.5, 1, 0.5]).reshape(-1,1)).flatten().tolist()
        list_lim_num[2] = [1, 1, 100, 1, 1, 1]

    if Field.field_param['ode_flag'] == 'mass_surface' and Field.field_param['flag_control'] is False:
        list_lim_num[0] = [100, 1, 1, 1, 1]
        list_lim_num[1] = [100, 1, 1, 1, 1]
        list_lim_num[2] = [1, 100, 1, 1, 1]
    if Field.field_param['ode_flag'] == 'mass_surface' and Field.field_param['flag_control'] is True:
        list_train_lim_max[0] = Field.compute_points(np.array([0, -2, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]).reshape(-1,1)).flatten().tolist()
        list_lim_num[0] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
        list_lim_num[1] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
        list_train_lim_max[1] = Field.compute_points(np.array([0, 0, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]).reshape(-1,1)).flatten().tolist()

        list_train_lim_max[2] = np.array([0, -2, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]).reshape(-1,1).flatten().tolist()
        list_lim_num[2] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
        list_lim_num[3] = [100, 1, 1, 1, 1, 1, 1, 1, 1]
        list_train_lim_max[3] = np.array([0, 0, -0.01, -0.5, -0.5, -0.5, -5, -5, -5]).reshape(-1,1).flatten().tolist()


    for i in range(len(list_lim_min)):
        #Field.field_param['lim_min'] = list_lim_min[i]
        Field.field_param['lim_train_max'] = list_train_lim_max[i]
        Field.field_param['lim_num'] = list_lim_num[i]

        Field.compute_training_points(number_observations, noise_std, observation_flag='grid')
        Field.compute_prediction_points()

        if res_AMGP != [None]:
            dict_post_AMGP = Gp_AMGP.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_AMGP.x,
                                                 Gp_AMGP.covariance, Gp_AMGP.covariance, Gp_AMGP.covariance)
        if res_Indep != [None]:
            dict_post_Indep = Gp_Indep.update_data(Field.X_predict, Field.X_train, Field.Y_train_noisy, res_Indep.x,
                                                   Gp_Indep.covariance, Gp_Indep.covariance, Gp_Indep.covariance)

        if res_Indep != [None] and res_AMGP != [None]:
            list_dict_data = [dict_post_Indep, dict_post_AMGP]
        elif res_AMGP != [None]:
            list_dict_data = [dict_post_AMGP, dict_post_AMGP]
        elif res_Indep != [None]:
            list_dict_data = [dict_post_Indep, dict_post_Indep]

        # UNSCALE DATA
        if res_Indep != [None]:
            check_constraint_n(Field, dict_post_Indep, 'simulation_data/' + filename, flag_save=flag_save)
        if res_AMGP != [None]:
            check_constraint_n(Field, dict_post_AMGP, 'simulation_data/' + filename, flag_save=flag_save)

        plot_slice(list_dict_data, Field, 'simulation_data/' + filename, flag_save=flag_save)