# Copyright 2019 Max Planck Society. All rights reserved.

import pickle

from subclass_ODE_unicycle import *
from subclass_ODE_mass_on_surface import *

from subclass_AMGP_normalized import *
from plot_scripts import *
from matplotlib.lines import Line2D
#flag = 'slice'
#flag = 'load_prediction'
#flag = 'plot_comparison'
#flag = 'Data_Comparison'
flag = 'ode_trajectory'

from mayavi import mlab
from matplotlib.cm import get_cmap  # for viridis

#########################
# Plot Slice
#########################
if flag == 'slice':
    #filename = '2019-11-08_12-09-43_mass_surface_indep_samesameFalse_TrueTruenum_obs60_L-BFGS-B_theta_pts1'
    filename = '2019-11-08_13-29-23_mass_surface_amgp_samesameFalse_TrueTruenum_obs40_L-BFGS-B_theta_pts1'


    number_observations = 10
    noise_std = 1
    observation_flag = 'grid' # EIther 'grid' or None
    flag_control = True

    with open('simulation_data/' + \
              filename +\
              '/result',
              'rb') as f:  # Python 3: open(..., 'rb')
                optim_bounds, theta_param, res_Indep, res_AMGP, Field = pickle.load(f)

    Field.compute_prediction_points()
    Field.field_param['number_observations'] = number_observations
    Field.compute_training_points(number_observations, noise_std, observation_flag=observation_flag)
    Gp_Indep = None
    Gp_AMGP = None
    if res_Indep != [None]:
        Gp_Indep = class_GP(Field)
    if res_AMGP != [None]:
        Gp_AMGP = subclass_AMGP_normalized(Field)
        res_AMGP.x[(Gp_AMGP.num_lengthscales + Gp_AMGP.numb_signalvar):(Gp_AMGP.num_lengthscales + Gp_AMGP.numb_signalvar + 3)] = \
            [noise_std, noise_std, noise_std]

    analyse_model(Field, Gp_AMGP, Gp_Indep, filename,
                  number_observations=10,
                  noise_std=noise_std,
                  res_AMGP=res_AMGP,
                  res_Indep=res_Indep)

#####################
# Load data for Figure 1
#####################
if flag == 'load_prediction':
    file_path = 'data_l4dc/optim1_unicycle/predictions/'
    array_mean_RMSE = np.load(file_path + 'array_mean_RMSE.npy')
    array_Rsquared = np.load(file_path + 'array_Rsquared.npy')
    array_constraint_error = np.load(file_path + 'array_constraint_error.npy')

    print('array_mean_RMSE\n', array_mean_RMSE)
    print('array_Rsquared\n', array_Rsquared)
    print('array_constraint_error\n', array_constraint_error)

#####################
# Plot Data_Comparison
#####################
if flag == 'Data_Comparison':

    #flag_ode = ['unicycle', 'duffling_oscillator', 'mass_surface']
    flag_ode = ['mass_surface', 'unicycle', 'duffling_oscillator']

    list_datapoints = [10, 30, 50, 70, 90, 110, 130]

    #folder_path = 'data_l4dc/optim' + '2_' + flag_ode + '/'
    #file_path = folder_path + '0/'

    flag = [1,3,4,8]  # 1, 3, 4, 8 ONE MODEL AT THE TIME
    array = [[[] for i in range(len(flag))] for i in range(len(flag_ode))]

    def compute_metrics_of_model(file_path, list_datapoints):
        array_RMSE = np.load(file_path + 'array_mean_RMSE.npy')
        array_constrainterror_sum = np.load(file_path + 'array_constraint_error_sum.npy')

        tmp = np.zeros((len(list_datapoints),6))
        for ii in range(len(list_datapoints)):  # Iterate over increasing number of observations
            tmp[ii,0] = np.mean(array_RMSE[ii, :])
            tmp[ii, 1] = np.min(array_RMSE[ii, :])
            tmp[ii, 2] = np.max(array_RMSE[ii, :])
            tmp[ii, 3] = np.mean(array_constrainterror_sum[ii, :])
            tmp[ii, 4] = np.min(array_constrainterror_sum[ii, :])
            tmp[ii, 5] = np.max(array_constrainterror_sum[ii, :])
        return tmp

    for xx in range(len(flag_ode)):
        for pp in range(len(flag)):
            print('MODEL: ', pp)
            file_path = 'data_l4dc/optim' + '2_' + flag_ode[xx] + '/0/predictions_' + str(flag[pp]) + '/'

            array[xx][pp] = compute_metrics_of_model(file_path, list_datapoints)

    plot_data_comparison(array, list_datapoints, flag_ode, flag)


# #########################
# # PLOT ODE TRAJECTORY
# #########################
if flag == 'ode_trajectory':
    import seaborn as sns
    # sns.set()
    # sns.set_style("darkgrid")
    # sns.set_context("paper")

    from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
    folder_name = 'data_l4dc/optim2_mass_surface/'
    file_name = folder_name + '0/'

    with open(file_name + 'result', 'rb') as f:  # Python 3: open(..., 'rb')
        optim_bounds, theta_param, res_Indep, res_AMGP, Field = pickle.load(f)
    res_AMGPx = np.load(file_name + 'res_AMGPx.npy')
    field_param = Field.field_param

    folder_name = 'data_l4dc/optim2_mass_surface/ode_predictions/'
    X = np.load(folder_name +   'X.npy')
    Y = np.load(folder_name +   'Y.npy')
    Z = np.load(folder_name +   'Z.npy')

    XXy = np.load(folder_name +   'XXy.npy')
    XXy_params = np.load(folder_name +   'XXy_params.npy')
    XXy_phys = np.load(folder_name +   'XXy_phys.npy')
    XXy_GPy = np.load(folder_name +   'XXy_GPy.npy')

    dXX = np.load(folder_name +   'dXXy.npy')
    dXX_params = np.load(folder_name +   'dXXy_params.npy')
    dXX_phys = np.load(folder_name +   'dXXy_phys.npy')
    dXX_GPy = np.load(folder_name +   'dXXy_GPy.npy')

    t_steps = np.load(folder_name +   't_steps.npy')

    ###########################
    # Plot the numerical solution
    ###########################
    color_scale = 'PuBu'
    #color_scale = 'binary'
    #line_color_phys = 'turquoise'
    #line_color_GP2 = 'blue'
    #line_color_GP = 'darkred'

    line_color_phys = 'black'
    line_color_GP2 = 'blue'
    line_color_GP2_params = 'steelblue'
    line_color_GP = 'gray'

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    #matplotlib.rcParams['figure.figsize'] = 14, 5
    matplotlib.rcParams['font.size'] = 14

    fig, ax = plt.subplots(3, 1, sharex=True)
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]

    ax[0].plot(t_steps, XXy[0, :], color='red', linestyle='-')
    ax[0].plot(t_steps, XXy[1, :], color='green', linestyle='-')
    ax[0].plot(t_steps, XXy[2, :], color='blue', linestyle='-')

    ax[0].plot(t_steps, XXy_params[0, :], color='red', linestyle='-.')
    ax[0].plot(t_steps, XXy_params[1, :], color='green', linestyle='-.')
    ax[0].plot(t_steps, XXy_params[2, :], color='blue', linestyle='-.')

    ax[0].plot(t_steps, XXy_GPy[0, :], color='red', linestyle=':')
    ax[0].plot(t_steps, XXy_GPy[1, :], color='green', linestyle=':')
    ax[0].plot(t_steps, XXy_GPy[2, :], color='blue', linestyle=':')

    ax[0].plot(t_steps, XXy_phys[0, :], color='red', linestyle='--')
    ax[0].plot(t_steps, XXy_phys[1, :], color='green', linestyle='--')
    ax[0].plot(t_steps, XXy_phys[2, :], color='blue', linestyle='--')

    #ax[0].rcParams.update({'font.size': 14})  # increase the font size
    ax[0].set_ylabel('position [m]')
    ax[0].legend(custom_lines, [r'$q_1$', r'$q_2$', r'$q_3$'], loc='lower left')
    ax[0].grid()

    # ax[0].title('velocities')
    #ax[1].rcParams.update({'font.size': 14})  # increase the font size
    ax[1].set_ylabel('velocity [m/s]')
    ax[1].plot(t_steps, XXy[3, :], color='red', linestyle='-')
    ax[1].plot(t_steps, XXy[4, :], color='green', linestyle='-')
    ax[1].plot(t_steps, XXy[5, :], color='blue', linestyle='-')

    ax[1].plot(t_steps, XXy_params[3, :], color='red', linestyle='-.')
    ax[1].plot(t_steps, XXy_params[4, :], color='green', linestyle='-.')
    ax[1].plot(t_steps, XXy_params[5, :], color='blue', linestyle='-.')

    ax[1].plot(t_steps, XXy_GPy[3, :], color='red', linestyle=':')
    ax[1].plot(t_steps, XXy_GPy[4, :], color='green', linestyle=':')
    ax[1].plot(t_steps, XXy_GPy[5, :], color='blue', linestyle=':')

    ax[1].plot(t_steps, XXy_phys[3, :], color='red', linestyle='--')
    ax[1].plot(t_steps, XXy_phys[4, :], color='green', linestyle='--')
    ax[1].plot(t_steps, XXy_phys[5, :], color='blue', linestyle='--')
    ax[1].legend(custom_lines, [r'$\dot{q}_1$', r'$\dot{q}_2$', r'$\dot{q}_3$'], loc='lower left')
    ax[1].grid()

    # ax[2].title('accelerations')
    #ax[2].rcParams.update({'font.size': 14})  # increase the font size
    ax[2].set_xlabel('time [s]')
    ax[2].set_ylabel(r'acceleration [m/s$^2$]')
    ax[2].plot(t_steps, dXX[0, :], color='red', linestyle='-',)
    ax[2].plot(t_steps, dXX[1, :], color='green', linestyle='-')
    ax[2].plot(t_steps, dXX[2, :], color='blue', linestyle='-')

    ax[2].plot(t_steps, dXX_params[0, :], color='red', linestyle='-.')
    ax[2].plot(t_steps, dXX_params[1, :], color='green', linestyle='-.')
    ax[2].plot(t_steps, dXX_params[2, :], color='blue', linestyle='-.')

    ax[2].plot(t_steps, dXX_GPy[0, :], color='red', linestyle=':')
    ax[2].plot(t_steps, dXX_GPy[1, :], color='green', linestyle=':')
    ax[2].plot(t_steps, dXX_GPy[2, :], color='blue', linestyle=':')

    ax[2].plot(t_steps, dXX_phys[0, :], color='red', linestyle='--')
    ax[2].plot(t_steps, dXX_phys[1, :], color='green', linestyle='--')
    ax[2].plot(t_steps, dXX_phys[2, :], color='blue', linestyle='--')
    ax[2].set_xlim(np.min(t_steps), np.max(t_steps))
    ax[2].legend(custom_lines, [r'$\ddot{q}_1$', r'$\ddot{q}_2$', r'$\ddot{q}_3$'], loc='lower left')
    ax[2].grid()
    plt.subplots_adjust(left=0.1, bottom=0.07, right=0.95, top=0.95, wspace=0.03, hspace=0.1)

    plt.show()

    fig = plt.figure(1)
    grid = plt.GridSpec(2, 8)

    # 3D surface plot
    plt.subplot(grid[0:, 0:3])
    X_train_unscaled = Field.un_normalize_points(Field.X_train, Field.dict_norm_X)
    ax1 = fig.add_subplot(grid[0:, 0:3], projection='3d')
    ax1.text2D(0.05, 0.8, "3D plot of surface \n and RK45 trajectories", transform=ax1.transAxes) #title('3D plot of surface')
    mycmap = plt.get_cmap(color_scale)  # 'binary' 'gist_earth' 'twilight_shifted'
    surf1 = ax1.plot_surface(X, Y, Z, alpha=0.9, cmap=mycmap)
    #ax1.plot(X_train_unscaled[0, :], X_train_unscaled[1, :], X_train_unscaled[2, :], linewidth=1, marker="o", color='red')
    ax1.plot(XXy_phys[0, :], XXy_phys[1, :], XXy_phys[2, :], linewidth=1, marker="o", color=line_color_phys, label='Analy. ODE')
    ax1.plot(XXy[0, :], XXy[1, :], XXy[2, :], linewidth=1, marker="o", color=line_color_GP2, label=r'GP$^2$')
    ax1.plot(XXy_params[0, :], XXy_params[1, :], XXy_params[2, :], linewidth=1, marker="o", color=line_color_GP2_params, label=r'GP$^2 params$')
    ax1.plot(XXy_GPy[0, :], XXy_GPy[1, :], XXy_GPy[2, :], linewidth=1, marker="o", color=line_color_GP, label='SE')

    #fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.set_xlabel(r"$q_1$ [m]")
    ax1.set_ylabel(r"$q_2$ [m]")
    ax1.set_zlabel(r"$q_3$ [m]")
    ax1.grid(False)
    ax1.legend(ncol=2, prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.65, 1.05))

    # 2D trajectory plot
    ax23 = plt.subplot(grid[0:, 3:6])
    #plt.title('2D plot of surface and RK45 trajectories')f_sample
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel(r"$q_1$ [m]")
    plt.ylabel(r"$q_2$ [m]")
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10, rotation=90)

    min_x = np.min((XXy_phys[0, :], XXy[0, :], XXy_GPy[0, :]))
    max_x = np.max((XXy_phys[0, :], XXy[0, :], XXy_GPy[0, :]))
    min_y = np.min((XXy_phys[1, :], XXy[1, :], XXy_GPy[1, :]))
    max_y = np.max((XXy_phys[1, :], XXy[1, :], XXy_GPy[1, :]))

    bord = 0.2
    #v = np.linspace(-.15, 0.4, 10, endpoint=True)
    #plt.imshow(Z.T, extent=[min_x-bord, max_x+bord, min_y-bord, max_y+bord], origin='lower', cmap=color_scale)
    #plt.contourf(X, Y, Z, v, origin='lower', xlim=[min_x - bord, max_x + bord], ylim=[min_y - bord, max_y + bord], cmap=color_scale)
    #CS = plt.contour(X, Y, Z, 6, colors='k', origin='lower', xlim=[min_x - bord, max_x + bord], ylim=[min_y - bord, max_y + bord])
    #plt.clabel(CS, inline=1, fontsize=10)
    #plt.colorbar(ticks=v)
    contours = plt.contour(X, Y, Z, 8, colors='gray')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(Z.T, extent=[-2, 2, -2, 2], origin='lower', cmap=color_scale, alpha=0.5)


    plt.plot(XXy_phys[0, :], XXy_phys[1, :], color=line_color_phys)
    plt.quiver(XXy_phys[0, :], XXy_phys[1, :], XXy_phys[3, :], XXy_phys[4, :], color=line_color_phys, label='Analy. ODE')

    plt.plot(XXy[0, :], XXy[1, :], color=line_color_GP2)
    plt.quiver(XXy[0, :], XXy[1, :], XXy[3, :], XXy[4, :], color=line_color_GP2, label=r'GP$^2$')

    plt.plot(XXy_params[0, :], XXy_params[1, :], color=line_color_GP2_params)
    plt.quiver(XXy_params[0, :], XXy_params[1, :], XXy_params[3, :], XXy_params[4, :], color=line_color_GP2_params, label=r'GP$^2$')

    plt.plot(XXy_GPy[0, :], XXy_GPy[1, :], color=line_color_GP)
    plt.quiver(XXy_GPy[0, :], XXy_GPy[1, :], XXy_GPy[3, :], XXy_GPy[4, :], color=line_color_GP, label='SE')

    #plt.xlim(min_x-bord, max_x+bord)
    #plt.ylim(min_y-bord, max_y+bord)
    #plt.legend(ncol=1, prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, 1.1))
    #plt.axis('equal')
    #ax23.text(0.05, 0.9, "2D plot of surface \n and RK45 trajectories", transform=ax23.transAxes, fontsize=10) #title('3D plot of surface')
    #plt.gca().set_aspect('equal')  # , adjustable='box'
    #plt.grid()

    # Constraint errors
    ax2 = plt.subplot(grid[0:, 6:8])
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel('time in s', fontsize=10)
    plt.ylabel('constraint error', fontsize=10)
    def compute_errors(Field, XXy, dXX, flag):
        constr_error = np.zeros(XXy.shape[1])
        if flag == 'constraint':
            for i in range(XXy.shape[1]):
                constr_error[i] = np.abs(
                    Field.constraint_A(XXy[:, i], Field.field_param['params']) @ dXX[:, i] -
                    Field.constraint_b(XXy[:, i], Field.field_param['params']))
        elif flag == 'z':
            for i in range(XXy.shape[1]):
                constr_error[i] = np.abs(
                    Field.func_z(XXy[:2, i], Field.field_param['params']) - XXy[2, i])
        elif flag == 'velocity':
            for i in range(XXy.shape[1]):
                constr_error[i] = np.abs(
                    Field.func_z_dot(XXy[:, i], Field.field_param['params']) - XXy[5, i])
        #elif flag=='RMSE':
        #    rmse(XXy[:, i])
        return constr_error

    constr_error = compute_errors(Field, XXy, dXX, 'constraint')
    constr_error_params = compute_errors(Field, XXy_params, dXX_params, 'constraint')
    constr_error_GPy = compute_errors(Field, XXy_GPy, dXX_GPy, 'constraint')
    constr_error_phys = compute_errors(Field, XXy_phys, dXX_phys, 'constraint')

    constr_z = compute_errors(Field, XXy, dXX, 'z')
    constr_z_params = compute_errors(Field, XXy_params, dXX_params, 'z')
    constr_z_GPy = compute_errors(Field, XXy_GPy, dXX_GPy, 'z')
    constr_z_phys = compute_errors(Field, XXy_phys, dXX_phys, 'z')
    
    plt.semilogy(t_steps, constr_error_phys,  color=line_color_phys, label=r'Analytical ODE')
    plt.semilogy(t_steps, constr_error,  color=line_color_GP2, label=r'GP$^2$')
    plt.semilogy(t_steps, constr_error_params,  color=line_color_GP2_params, label=r'GP$^2 \theta_p$')
    plt.semilogy(t_steps, constr_error_GPy,  color=line_color_GP, label=r'SE-ARD')
    plt.semilogy(t_steps, constr_z,  color=line_color_GP2, label=r'(GP$^2$)')
    plt.semilogy(t_steps, constr_z_params,  color=line_color_GP2_params, label=r'(GP$^2 \theta_p$)')
    plt.semilogy(t_steps, constr_z_phys,  color=line_color_phys, label=r'(ODE)')
    plt.semilogy(t_steps, constr_z_GPy,  color=line_color_GP, label=r'(GPy)')
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10, rotation=90)
    handles, labels = ax2.get_legend_handles_labels()
    order = [1, 0, 4, 2, 3, 5]
    #order = [1, 0, 2, 3]
    plt.legend(ncol=1, prop={'size': 10}, loc='center', bbox_to_anchor=(0.6, 0.6))
    plt.grid()
    #plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95, wspace=0.005, hspace=0.001)
    plt.show()

    fig2 = mlab.figure(bgcolor=(1, 1, 1))
    # manually set viridis for the surface
    su = mlab.surf(X[49:,:90], Y[49:,:90], Z[49:,:90], opacity=0.9, colormap='Blues')
    #ax1.plot(X_train_unscaled[0, :], X_train_unscaled[1, :], X_train_unscaled[2, :], linewidth=1, marker="o", color='red')
    sc = mlab.points3d(XXy_phys[0, :], XXy_phys[1, :], XXy_phys[2, :], scale_factor=0.1, scale_mode='none', color=(0,0,0))
    sc = mlab.points3d(XXy[0, :], XXy[1, :], XXy[2, :], scale_factor=0.1, scale_mode='none', color=(0,0,1))
    sc = mlab.points3d(XXy_params[0, :], XXy_params[1, :], XXy_params[2, :], scale_factor=0.1, scale_mode='none', color=(1,0,0))
    sc = mlab.points3d(XXy_GPy[0, :], XXy_GPy[1, :], XXy_GPy[2, :], scale_factor=0.1, scale_mode='none', color=(0.5,0.5,0.5))

    cat1_extent = (0, 2, -2, 1.6, -3.2, 0.5)
    mlab.outline(su, color=(.7, .7, .7), extent=cat1_extent)
    #cmap_name = 'GnBu_r'
    #cdat = np.array(get_cmap(cmap_name, 256).colors)
    #cdat = (cdat * 255).astype(int)
    #su.module_manager.scalar_lut_manager.lut.table = cdat

    mlab.show()

    #fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.set_xlabel(r"$q_1$ [m]")
    ax1.set_ylabel(r"$q_2$ [m]")
    ax1.set_zlabel(r"$q_3$ [m]")
    ax1.grid(False)
    ax1.legend(ncol=2, prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.65, 1.05))


    plt.tight_layout()

    plt.show()