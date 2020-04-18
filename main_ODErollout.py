# Copyright 2019 Max Planck Society. All rights reserved.

# This script computes a trajectory given the surface on particle ODE using
# either the analytical ODE or GP models fed into an ODE solver

from plot_scripts import *
import pickle
import copy

import datetime

import os
from scipy.integrate import solve_ivp
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

from class_ODE import *
from subclass_ODE_unicycle import *
from subclass_AMGP_normalized import *
from global_functions import *
from plot_scripts import *
from mayavi import mlab

##################################################################################
# SYSTEM-settings
flag_ode = 'mass_surface'
#flag_ode = 'unicycle'
#flag_ode = 'duffling_oscillator'

# GP-settings
#flag_GP = 'indep'
flag_GP = 'amgp'
flag_GP_params = 'amgp_params'

#flag_plot = 'paper'
flag_plot = 'analyze'

flag = [0, 1]
# Should the constraint knowledge be used to ensure that the GPy state estimate
# fulfills the constraints?
flag_constrain_GPy_estimate = False


# GP file
folder_name = 'data_l4dc/optim2_mass_surface/'
file_name = folder_name + '0/'

with open(file_name + 'result', 'rb') as f:  # Python 3: open(..., 'rb')
    optim_bounds, theta_param, res_Indep, res_AMGP, Field = pickle.load(f)
field_param = Field.field_param
    
# Initialize GPs
if flag_GP == 'indep':
    Gp_Indep = subclass_AMGP_normalized(Field)
    params_Indep = {'theta': res_Indepx,  # res_AMGP_2.x
                   'covariance_TT': Gp_Indep.covariance,
                   'Field': Field}
    Gp_Indep.init_update_point(params_Indep)

if flag_GP == 'amgp':
    res_AMGPx = np.load(file_name + 'res_AMGPx.npy')
    Gp_AMGP = subclass_AMGP_normalized(Field)
    params_AMGP = {'theta': res_AMGPx,  # res_AMGP_2.x
                               'covariance_TT': Gp_AMGP.covariance,
                               'Field': Field}
    Gp_AMGP.init_update_point(params_AMGP)

if flag_GP_params == 'amgp_params':
    res_AMGPx_params = np.load(file_name + 'res_AMGPx_params.npy')
    Gp_AMGP_params = subclass_AMGP_normalized(Field)
    params_AMGP_params = {'theta': res_AMGPx_params,  # res_AMGP_2.x
                               'covariance_TT': Gp_AMGP_params.covariance,
                               'Field': Field}
    Gp_AMGP_params.init_update_point(params_AMGP_params)

# Define Simulation parameter
t_end = 6
t_steps = np.linspace(0, t_end, 100)
x0 = 0.4
y0 = 0.5
z0 = 0
dx0 = 0.9
dy0 = 0
dz0 = 0

x0 = 0.4
y0 = 0.8
z0 = 0
dx0 = 0.8
dy0 = 1
dz0 = 0

# x0 = 1.5
# y0 = 1.5
# z0 = 0
# dx0 = 0.9
# dy0 = 0.5
# dz0 = 0

X0 = Field.compute_points(np.array([x0, y0, z0, dx0, dy0, dz0]).reshape(-1,1)).flatten()  # the initial condition
m = 3
u = np.array([0, 0, 0])
params_AMGP['u'] = u
params_AMGP_params['u'] = u

###########################
# Run ODE solver
###########################
# GP2 model
XXy = Field.compute_trajectory(t_steps, np.copy(X0), Gp_AMGP.update_point_ODE, params_AMGP)
if any(elem in flag for elem in [0]):
    dXX = np.zeros((3, XXy.shape[1]))  # Acceleration at solution
    for i in range(XXy.shape[1]):
        dXX[:, i] = Gp_AMGP.update_point_ODE(t_steps[i], XXy[:, i], params_AMGP)[3:]

XXy_params = Field.compute_trajectory(t_steps, np.copy(X0), Gp_AMGP_params.update_point_ODE, params_AMGP_params)
if any(elem in flag for elem in [0]):
    dXX_params = np.zeros((3, XXy_params.shape[1]))  # Acceleration at solution
    for i in range(XXy.shape[1]):
        dXX_params[:, i] = Gp_AMGP_params.update_point_ODE(t_steps[i], XXy_params[:, i], params_AMGP_params)[3:]

# UKE ODE model
params_phys = {'theta': Field.field_param['params'], 'u': u}
decorated_ODE = Field.decorator_changeout_for_solveivp(Field.func_ODE, 3)
XXy_phys = Field.compute_trajectory(t_steps, np.copy(X0), decorated_ODE, params_phys)

# SE-ARD model
if any(elem in flag for elem in [1]):
    Field.compute_training_points(1000, 0.01, observation_flag='random')
    K = GPy.kern.RBF(input_dim=Field.field_param['dim_in'], ARD=True)
    ################################
    # Standard GPy ARD-GP
    ################################
    m1 = GPy.models.GPRegression(Field.X_train.T, Field.Y_train_noisy_tmp.T[:, [0]], K.copy())
    # Load model parameters
    m1.update_model(False)
    m1.initialize_parameter()
    file_path1 = file_name+ 'm1' + str(0) + '_params.npy'
    m1[:] = np.load(file_path1)
    m1.update_model(True)
    m2 = GPy.models.GPRegression(Field.X_train.T, Field.Y_train_noisy_tmp.T[:, [1]], K.copy())
    # Load model parameters
    m2.update_model(False)
    m2.initialize_parameter()
    file_path1 = file_name + 'm1' + str(1) + '_params.npy'
    m2[:] = np.load(file_path1)
    m2.update_model(True)
    m3 = GPy.models.GPRegression(Field.X_train.T, Field.Y_train_noisy_tmp.T[:, [2]], K.copy())
    # Load model parameters
    m3.update_model(False)
    m3.initialize_parameter()
    file_path1 = file_name + 'm1' + str(2) + '_params.npy'
    m3[:] = np.load(file_path1)
    m3.update_model(True)

    params_AMGP['m1'] = m1
    params_AMGP['m2'] = m2
    params_AMGP['m3'] = m3
    params_AMGP['flag_constrain_GPy_estimate'] = flag_constrain_GPy_estimate
    XXy_GPy = Field.compute_trajectory(t_steps, np.copy(X0), Gp_AMGP.adjust_GPy_ARD_for_solveivp, params_AMGP)
    XXy_GPy_params = Field.compute_trajectory(t_steps, np.copy(X0), Gp_AMGP_params.adjust_GPy_ARD_for_solveivp, params_AMGP)


###########################
# Compute acceleration
###########################
if any(elem in flag for elem in [1]):
    dXX_GPy = np.zeros((3, XXy_GPy.shape[1]))  # Acceleration at solution
    for i in range(XXy_GPy.shape[1]):
        dXX_GPy[:, i] = Gp_AMGP.adjust_GPy_ARD_for_solveivp(t_steps[i], XXy_GPy[:, i], params_AMGP)[3:]

dXX_phys = np.zeros((3, XXy_phys.shape[1]))  # Acceleration at solution
for i in range(XXy_phys.shape[1]):
    dXX_phys[:, i] = decorated_ODE(t_steps[i], XXy_phys[:, i], params_phys)[3:]

###########################
# Compute energy
###########################
Energy_kin, Energy_pot, Energy_tot = Field.compute_energy(XXy)
Energy_kin_params, Energy_pot_params, Energy_tot_params = Field.compute_energy(XXy_params)
Energy_kin_GPy, Energy_pot_GPy, Energy_tot_GPy = Field.compute_energy(XXy_GPy)
Energy_kin_phys, Energy_pot_phys, Energy_tot_phys = Field.compute_energy(XXy_phys)

###########################
# Compute Surface
###########################
numb_field = 100
Field.field_param['lim_num'] = [numb_field, numb_field, 1, 1, 1, 1, 1, 1, 1]
Field.compute_prediction_points()
X = Field.X_predict_unnormalized[0, :].reshape((numb_field, numb_field))
Y = Field.X_predict_unnormalized[1, :].reshape((numb_field, numb_field))
Z = Field.X_predict_unnormalized[2, :].reshape((numb_field, numb_field))


np.save(folder_name + 'ode_predictions/' + 'X.npy', X)
np.save(folder_name + 'ode_predictions/' + 'Y.npy', Y)
np.save(folder_name + 'ode_predictions/' + 'Z.npy', Z)

np.save(folder_name + 'ode_predictions/' + 'XXy.npy', XXy)
np.save(folder_name + 'ode_predictions/' + 'XXy_params.npy', XXy_params)
np.save(folder_name + 'ode_predictions/' + 'XXy_phys.npy', XXy_phys)
np.save(folder_name + 'ode_predictions/' + 'XXy_GPy.npy', XXy_GPy)

np.save(folder_name + 'ode_predictions/' + 'dXXy.npy', dXX)
np.save(folder_name + 'ode_predictions/' + 'dXXy_params.npy', dXX_params)
np.save(folder_name + 'ode_predictions/' + 'dXXy_phys.npy', dXX_phys)
np.save(folder_name + 'ode_predictions/' + 'dXXy_GPy.npy', dXX_GPy)

np.save(folder_name + 'ode_predictions/' + 't_steps.npy', t_steps)


if flag_plot == 'analyze':
    ###########################
    # Plot the numerical solution
    ###########################
    fig = plt.figure(1)

    grid = plt.GridSpec(3, 9)

    # 3D surface plot
    color_scale = 'PuBuGn'
    color_scale = 'BrBG_r'
    color_scale = 'Pastel1'
    color_scale = 'GnBu_r'
    X_train_unscaled = Field.un_normalize_points(Field.X_train, Field.dict_norm_X)
    ax1 = fig.add_subplot(grid[0:3, 0:3], projection='3d')
    mycmap = plt.get_cmap(color_scale)  # 'binary' 'gist_earth' 'twilight_shifted'
    surf1 = ax1.plot_surface(X, Y, Z, alpha=0.9, cmap=mycmap)
    #ax1.plot(X_train_unscaled[0, :], X_train_unscaled[1, :], X_train_unscaled[2, :], linewidth=1, marker="o", color='red')
    ax1.plot(XXy[0, :], XXy[1, :], XXy[2, :], linewidth=1, marker="o", color='red')
    ax1.plot(XXy_GPy[0, :], XXy_GPy[1, :], XXy_GPy[2, :], linewidth=1, marker="o", color='orange')
    ax1.plot(XXy_phys[0, :], XXy_phys[1, :], XXy_phys[2, :], linewidth=1, marker="o", color='green')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')

    # color_scale = 'GnBu_r'
    # X_train_unscaled = Field.un_normalize_points(Field.X_train, Field.dict_norm_X)
    # ax1 = fig.add_subplot(grid[0:3, 0:3], projection='3d')
    # mycmap = plt.get_cmap(color_scale)  # 'binary' 'gist_earth' 'twilight_shifted'
    # surf1 = ax1.mlab.surf(X, Y, Z, alpha=0.9, cmap=mycmap)
    # #ax1.plot(X_train_unscaled[0, :], X_train_unscaled[1, :], X_train_unscaled[2, :], linewidth=1, marker="o", color='red')
    # ax1.mlab.points3d(XXy[0, :], XXy[1, :], XXy[2, :], linewidth=1, marker="o", color='red')
    # ax1.mlab.points3d(XXy_GPy[0, :], XXy_GPy[1, :], XXy_GPy[2, :], linewidth=1, marker="o", color='orange')
    # ax1.mlab.points3d(XXy_phys[0, :], XXy_phys[1, :], XXy_phys[2, :], linewidth=1, marker="o", color='green')
    # fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    # ax1.set_xlabel('x axis')
    # ax1.set_ylabel('y axis')
    # ax1.set_zlabel('z axis')

    # 2D trajectory plot
    plt.subplot(grid[2, 4:6])
    plt.title('Trajectory with velocity')
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel("x")
    plt.ylabel("y")

    plt.imshow(Z.T, extent=[-2, 2, -2, 2], origin='lower', cmap=color_scale)

    plt.plot(XXy_phys[0, :], XXy_phys[1, :], color='green')
    plt.quiver(XXy_phys[0, :], XXy_phys[1, :], XXy_phys[3, :], XXy_phys[4, :], color='green')

    plt.plot(XXy[0, :], XXy[1, :], color='red')
    plt.quiver(XXy[0, :], XXy[1, :], XXy[3, :], XXy[4, :], color='red')

    plt.plot(XXy_params[0, :], XXy_params[1, :], color='red')
    plt.quiver(XXy_params[0, :], XXy_params[1, :], XXy_params[3, :], XXy_params[4, :], color='red')

    plt.plot(XXy_GPy[0, :], XXy_GPy[1, :], color='red')
    plt.quiver(XXy_GPy[0, :], XXy_GPy[1, :], XXy_GPy[3, :], XXy_GPy[4, :], color='orange')

    plt.axis('equal')
    plt.gca().set_aspect('equal')  # , adjustable='box'
    plt.grid()

    # Energies
    plt.subplot(grid[0, 4:6])
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.plot(t_steps, Energy_kin, color='red', linestyle='-', label=r'kinetic (GP$^2$)')
    plt.plot(t_steps, Energy_pot, color='green', linestyle='-', label=r'potential (GP$^2$)')
    plt.plot(t_steps, Energy_tot, color='blue', linestyle='-', label=r'total (GP$^2$)')
    plt.plot(t_steps, Energy_kin_GPy, color='red', linestyle=':', label=r'kinetic (GP)')
    plt.plot(t_steps, Energy_pot_GPy, color='green', linestyle=':', label=r'potential (GP)')
    plt.plot(t_steps, Energy_tot_GPy, color='blue', linestyle=':', label=r'total (GP)')
    plt.plot(t_steps, Energy_kin_phys, color='red', linestyle='--', label=r'kinetic (ODE)')
    plt.plot(t_steps, Energy_pot_phys, color='green', linestyle='--', label=r'potential (ODE)')
    plt.plot(t_steps, Energy_tot_phys, color='blue', linestyle='--', label=r'total (ODE)')
    plt.legend()
    plt.grid()

    # Constraint errors
    plt.subplot(grid[1, 4:6])
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel('time')
    plt.ylabel('constraint error')
    constr_error = np.zeros(XXy.shape[1])
    for i in range(XXy.shape[1]):
        constr_error[i] = np.abs(Field.constraint_A(XXy[:, i], Field.field_param['params']) @ dXX[:, i] - Field.constraint_b(XXy[:, i], Field.field_param['params']))

    constr_error_surf_phys = np.zeros(XXy_phys.shape[1])
    for i in range(XXy_phys.shape[1]):
        constr_error_surf_phys[i] = np.abs(Field.func_z(XXy_phys[:2, i], Field.field_param['params'])-XXy_phys[2, i])
    constr_error_surf_GPy = np.zeros(XXy_GPy.shape[1])
    for i in range(XXy_GPy.shape[1]):
        constr_error_surf_GPy[i] = np.abs(Field.func_z(XXy_GPy[:2, i], Field.field_param['params'])-XXy_GPy[2, i])

    plt.plot(t_steps, constr_error)
    #plt.plot(t_steps, constr_error_phys)
    #plt.plot(t_steps, constr_error_GPy)
    plt.grid()

    # plt.title('positions')
    plt.subplot(grid[0, 7:])
    plt.plot(t_steps, XXy[0, :], color='red', linestyle='-', label=r'$q_1$ (GP$^2$)')
    plt.plot(t_steps, XXy[1, :], color='green', linestyle='-')
    plt.plot(t_steps, XXy[2, :], color='blue', linestyle='-')

    plt.plot(t_steps, XXy_params[0, :], color='red', linestyle='-.', label=r'$q_1$ (GP$^2$ $\theta_p$)')
    plt.plot(t_steps, XXy_params[1, :], color='green', linestyle='-.')
    plt.plot(t_steps, XXy_params[2, :], color='blue', linestyle='-.')

    plt.plot(t_steps, XXy_GPy[0, :], color='red', linestyle=':')
    plt.plot(t_steps, XXy_GPy[1, :], color='green', linestyle=':', label=r'$q_2$ (ARD)')
    plt.plot(t_steps, XXy_GPy[2, :], color='blue', linestyle=':')

    plt.plot(t_steps, XXy_phys[0, :], color='red', linestyle='--')
    plt.plot(t_steps, XXy_phys[1, :], color='green', linestyle='--')
    plt.plot(t_steps, XXy_phys[2, :], color='blue', linestyle='--', label=r'$q_3$ (UKE)')

    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel('time')
    plt.ylabel('positions')
    plt.legend()
    plt.grid()

    # plt.title('velocities')
    plt.subplot(grid[1, 7:])
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel('time')
    plt.ylabel('velocities')
    plt.plot(t_steps, XXy[3, :], color='red', linestyle='-', label=r'$\dot{q}_1$ (GP$^2$)')
    plt.plot(t_steps, XXy[4, :], color='green', linestyle='-')
    plt.plot(t_steps, XXy[5, :], color='blue', linestyle='-')

    plt.plot(t_steps, XXy_params[3, :], color='red', linestyle='-.', label=r'$\dot{q}_1$ (GP$^2$ $\theta_p$)')
    plt.plot(t_steps, XXy_params[4, :], color='green', linestyle='-.')
    plt.plot(t_steps, XXy_params[5, :], color='blue', linestyle='-.')

    plt.plot(t_steps, XXy_GPy[3, :], color='red', linestyle=':')
    plt.plot(t_steps, XXy_GPy[4, :], color='green', linestyle=':', label=r'$\dot{q}_2$ (ARD)')
    plt.plot(t_steps, XXy_GPy[5, :], color='blue', linestyle=':')

    plt.plot(t_steps, XXy_phys[3, :], color='red', linestyle='--')
    plt.plot(t_steps, XXy_phys[4, :], color='green', linestyle='--')
    plt.plot(t_steps, XXy_phys[5, :], color='blue', linestyle='--', label=r'$\dot{q}_3$ (UKE)')
    plt.legend()
    plt.grid()

    # plt.title('accelerations')
    plt.subplot(grid[2, 7:])
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel('time')
    plt.ylabel(r'accelerations')
    plt.plot(t_steps, dXX[0, :], color='red', linestyle='-', label=r'$\ddot{q_1}$ (GP$^2$)')
    plt.plot(t_steps, dXX[1, :], color='green', linestyle='-')
    plt.plot(t_steps, dXX[2, :], color='blue', linestyle='-')

    plt.plot(t_steps, dXX_params[0, :], color='red', linestyle='-.', label=r'$\ddot{q_1}$ (GP$^2$ $\theta_p$)')
    plt.plot(t_steps, dXX_params[1, :], color='green', linestyle='-.')
    plt.plot(t_steps, dXX_params[2, :], color='blue', linestyle='-.')

    plt.plot(t_steps, dXX_GPy[0, :], color='red', linestyle=':')
    plt.plot(t_steps, dXX_GPy[1, :], color='green', linestyle=':', label=r'$\ddot{q_2}$ (ARD)')
    plt.plot(t_steps, dXX_GPy[2, :], color='blue', linestyle=':')

    plt.plot(t_steps, dXX_phys[0, :], color='red', linestyle='--')
    plt.plot(t_steps, dXX_phys[1, :], color='green', linestyle='--')
    plt.plot(t_steps, dXX_phys[2, :], color='blue', linestyle='--', label=r'$\ddot{q_3}$ (UKE)')
    plt.legend()
    plt.grid()

    plt.show()

if flag_plot == 'paper':
    ###########################
    # Plot the numerical solution
    ###########################
    #color_scale = 'PuBu'
    color_scale = 'binary'
    line_color_phys = 'black'
    line_color_GP2 = 'blue'
    line_color_GP = 'darkred'


    fig = plt.figure(1)

    grid = plt.GridSpec(2, 2)
    # Trajectory plot
    plt.subplot(grid[0:2, 0])
    plt.title('Trajectory with velocity')
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel("x")
    plt.ylabel("y")

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
    contours = plt.contour(X, Y, Z, 3, colors='gray')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(Z.T, extent=[-2, 2, -2, 2], origin='lower', cmap=color_scale, alpha=0.5)


    plt.plot(XXy_phys[0, :], XXy_phys[1, :], color=line_color_phys)
    plt.quiver(XXy_phys[0, :], XXy_phys[1, :], XXy_phys[3, :], XXy_phys[4, :], color=line_color_phys)

    plt.plot(XXy[0, :], XXy[1, :], color=line_color_GP2)
    plt.quiver(XXy[0, :], XXy[1, :], XXy[3, :], XXy[4, :], color=line_color_GP2)

    plt.plot(XXy_GPy[0, :], XXy_GPy[1, :], color=line_color_GP)
    plt.quiver(XXy_GPy[0, :], XXy_GPy[1, :], XXy_GPy[3, :], XXy_GPy[4, :], color=line_color_GP)

    #plt.xlim(min_x-bord, max_x+bord)
    #plt.ylim(min_y-bord, max_y+bord)

    #plt.axis('equal')
    plt.gca().set_aspect('equal')  # , adjustable='box'
    plt.grid()

    # Constraint errors
    plt.subplot(grid[0, 1])
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel('time')
    plt.ylabel('constraint error')
    def compute_errors(Field, XXy, dXX, flag):
        constr_error = np.zeros(XXy.shape[1])
        if flag == 'constraint':
            for i in range(XXy.shape[1]):
                constr_error[i] = np.abs(
                    Field.constraint_A(XXy[:, i], Field.field_param['params']) @ dXX[:, i] -
                    Field.constraint_b(XXy[:, i], Field.field_param['params']))
        elif flag == 'position':
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
    constr_error_GPy = compute_errors(Field, XXy_GPy, dXX_GPy, 'constraint')
    constr_error_phys = compute_errors(Field, XXy_phys, dXX_phys, 'constraint')

    plt.semilogy(t_steps, constr_error,  color=line_color_GP2, label=r'(GP$^2$)')
    plt.semilogy(t_steps, constr_error_phys,  color=line_color_phys, label=r'(ODE)')
    plt.semilogy(t_steps, constr_error_GPy,  color=line_color_GP, label=r'(GPy)')
    plt.grid()
    plt.tight_layout()

    # plt.title('accelerations')
    plt.subplot(grid[1, 1])
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel('time')
    plt.ylabel(r'accelerations')
    plt.plot(t_steps, dXX[0, :], color=line_color_GP2, linestyle='-', label=r'(GP$^2$)')
    plt.plot(t_steps, dXX[1, :], color=line_color_GP2, linestyle=':', label=r'(GP$^2$)')
    plt.plot(t_steps, dXX[2, :], color=line_color_GP2, linestyle='--', label=r'(GP$^2$)')

    plt.plot(t_steps, dXX_GPy[0, :], color=line_color_GP, linestyle='-', label=r'(GP)')
    plt.plot(t_steps, dXX_GPy[1, :], color=line_color_GP, linestyle=':', label=r'(GP)')
    plt.plot(t_steps, dXX_GPy[2, :], color=line_color_GP, linestyle='--', label=r'(GP)')

    plt.plot(t_steps, dXX_phys[0, :], color=line_color_phys, linestyle='-', label=r'(ODE)')
    plt.plot(t_steps, dXX_phys[1, :], color=line_color_phys, linestyle=':', label=r'(ODE)')
    plt.plot(t_steps, dXX_phys[2, :], color=line_color_phys, linestyle='--', label=r'(ODE)')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.show()
