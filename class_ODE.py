# Copyright 2019 Max Planck Society. All rights reserved.

from plot_scripts import *
import sys
from scipy.integrate import solve_ivp

class class_ODE:
    def __init__(self, flag_control=False):
        self.field_param = None

    def set_optimization_parameters(self, theta_pts,
                                    flag_estimate_sys_params=False,
                                    flag_optimizer='L-BFGS-B', #'DiffEvo',
                                    dev_con=0.1, l_min=1, l_max=10, sig_var_min=1, sig_var_max=5):
        """
        optim_bounds: bounds of the optimizer
        theta_param: range in which the initial hyperparameteres lie for optimization
        :param l_min: minimum lengthscale in optimization
        :param l_max: maximum lengthscale in optimization
        :param sig_var_min: maximum signal variance in optimization
        :param theta_pts: number of optimizer restarts
        :param flag_optimizer: set optimization method
        :param flag_estimate_sys_params: set if constraint parameteres will be estimated
        :param dev_con:
        :return:
        """

        theta_param = {'dim_min': [l_min] * self.field_param['numb_lengthscales'] + \
                                 [sig_var_min] * self.field_param['numb_signalvar'],
                       'dim_max': [l_max]*self.field_param['numb_lengthscales'] + \
                                  [sig_var_max]*self.field_param['numb_signalvar'],
                       'flag_optimizer': flag_optimizer,
                       'num_theta_phys': len(self.field_param['params']),
                       'theta_pts': theta_pts,
                       'flag_joblib_parallel': 'False'}
        theta_param['dim_min'] += [self.field_param['noise_std'] for x in range(self.field_param['dim_out'])]
        theta_param['dim_max'] += [self.field_param['noise_std']+1e-16 for x in range(self.field_param['dim_out'])]

        optim_bounds = [(l_min, l_max)] * self.field_param['numb_lengthscales'] + \
                       [(sig_var_min, sig_var_max)] * self.field_param['numb_signalvar'] + \
                       [(self.field_param['noise_std'], self.field_param['noise_std'])]*self.field_param['dim_out']

        if flag_estimate_sys_params == False:
            theta_param['dim_min'] += self.field_param['params']
            theta_param['dim_max'] += [x+1e-16 for x in self.field_param['params']]
            optim_bounds += [(x, x) for x in self.field_param['params']]

        elif flag_estimate_sys_params == True:
            theta_param['dim_min'] += [x-dev_con for x in self.field_param['params']]
            theta_param['dim_max'] += [x+dev_con for x in self.field_param['params']]
            optim_bounds += [(x-dev_con, x+dev_con) for x in self.field_param['params']]
        return theta_param, optim_bounds

    def fix_param_in_optim_bounds(self, theta_param, optim_bounds, flag, flag_value):
        """Wrapper to fix constraint parameters"""
        for i in range(len(flag)):
            theta_param['dim_min'][flag[i]] = flag_value[i]
            theta_param['dim_max'][flag[i]] = flag_value[i] + 1e-16
            optim_bounds[flag[i]] = (flag_value[i], flag_value[i])
        return theta_param, optim_bounds

    def compute_points(self, tmp_array):
        '''
        Takes input data points of shape dim_in x number_points and adjusts them to lie on constraint manifold.
        '''
        print('System has no constraints?')
        return tmp_array

    def compute_grid_points(self):
       allX = [np.linspace(self.field_param['lim_min'][ii], self.field_param['lim_max'][ii],
                           self.field_param['lim_num'][ii])
               for ii in range(len(self.field_param['lim_num']))]
       XX = np.meshgrid(*allX, indexing='ij')  # Create multi-dimensional coord-matrices
       X_predict_tmp = np.vstack(map(np.ravel, XX))  # Compute grid coordinate points
       return self.compute_points(X_predict_tmp)

    def stack_inputdata_for_covariance_computation(self, data1, data2):
        """Stacks the input data in a special format for efficient compuation of the covariance matrix"""
        assert data1.shape[0] == data1.shape[0] == self.field_param['dim_in']
        assert data1.shape[1] >= 1
        assert data2.shape[1] >= 1
        data_tmp = np.zeros((2*self.field_param['dim_in'], data1.shape[1] * data2.shape[1]))
        k=0
        for ii in range(data1.shape[1]):
            for jj in range(data2.shape[1]):
                data_tmp[:self.field_param['dim_in'],k] = data1[:, ii]
                data_tmp[self.field_param['dim_in']:,k] = data2[:, jj]
                k+=1
        return data_tmp

    def compute_training_points(self, number_observations, noise_std, observation_flag='grid'):
        self.field_param['number_observations'] = number_observations
        self.field_param['noise_std'] = noise_std

        if observation_flag == 'grid':
            # Pick random training points from prediction positions
            X_grid = self.compute_grid_points()
            self.indeces = np.random.randint(0, X_grid.shape[1], self.field_param['number_observations'])
            self.X_train = X_grid[:, self.indeces]

        elif observation_flag == 'random':
            # Sample training points in interval randomly
            X_train_tmp = np.zeros((len(self.field_param['lim_min']), self.field_param['number_observations']))
            if 'lim_train_min' not in self.field_param.keys():
                self.field_param['lim_train_min'] = self.field_param['lim_min']
            if 'lim_train_max' not in self.field_param.keys():
                self.field_param['lim_train_max'] = self.field_param['lim_max']

            for i in range(len(self.field_param['lim_min'])):
                if self.field_param['lim_num'][i] > 1:
                    X_train_tmp[i,:] = (self.field_param['lim_train_max'][i] - self.field_param['lim_train_min'][i]) *\
                                            np.random.rand(1, self.field_param['number_observations']) + \
                                            self.field_param['lim_train_min'][i]
                else:
                    X_train_tmp[i, :] = self.field_param['lim_min'][i]*np.ones((1, self.field_param['number_observations']))
            self.X_train = self.compute_points(X_train_tmp)

        else:
            sys.exit('Non valid observation_flag given to compute_training_points')

        self.check_if_states_fulfill_constraint(self.X_train)

        # Compute ODE values at points
        Y_train = self.define_observations(self.X_train)
        self.Y_train_noisy = Y_train + np.random.normal(0, noise_std, size=(Y_train.shape[0], 1),)
        #Y_train_ordered = Y_train.reshape((-1, self.field_param['dim_out'])).T
        self.Y_train_noisy_tmp = self.Y_train_noisy.reshape((-1, self.field_param['dim_out'])).T

        if self.field_param['flag_normalize_in'] == True:
            if hasattr(self, 'dict_norm_X'):
                pass
            else:
                self.dict_norm_X = self.normalization(self.X_train)
            self.X_train = self.normalize_points(self.X_train, self.dict_norm_X)

        #self.X_traintrain_formatted = self.stack_inputdata_for_covariance_computation(self.X_train, self.X_train)

        if self.field_param['flag_normalize_out'] == True:
                if hasattr(self, 'dict_norm_Y'):
                    pass
                else:
                    self.dict_norm_Y = self.normalization(self.Y_train_noisy_tmp)
                self.Y_train_noisy_tmp = self.normalize_points(self.Y_train_noisy_tmp,
                                                                      self.dict_norm_Y)
                self.Y_train_noisy = self.Y_train_noisy_tmp.T.reshape(-1,1)

    def compute_prediction_points(self):
        '''
        Computes scaled prediction points using compute grid points
        :return:
        '''
        self.X_predict = self.compute_grid_points()
        self.X_predict_unnormalized = np.copy(self.X_predict)
        self.Y_field = self.define_observations(self.X_predict)
        self.Y_field_ordered = self.Y_field.reshape((-1, self.field_param['dim_out'])).T
        self.check_if_states_fulfill_constraint(self.X_predict)

        if self.field_param['flag_normalize_in'] == True:
            self.X_predict = self.normalize_points(self.X_predict, self.dict_norm_X)
            X_test = self.un_normalize_points(self.X_predict, self.dict_norm_X)
            self.check_if_states_fulfill_constraint(X_test)

        if self.field_param['flag_normalize_out'] == True:
            self.Y_field_unnormalized = np.copy(self.Y_field)
            self.Y_field_unnormalized_ordered = np.copy(self.Y_field_ordered)
            self.Y_field_ordered = self.normalize_points(self.Y_field_ordered, self.dict_norm_Y)
            self.Y_field = self.Y_field_ordered.T.reshape(-1, 1)

    def normalization(self, tmp_array):
        assert tmp_array.shape[0] == self.field_param['dim_in'] or tmp_array.shape[0] == self.field_param['dim_out']
        dict_norm = {}
        dict_norm['N_mue'] = tmp_array.mean(axis=1).reshape((-1, 1))
        dict_norm['N_std'] = np.diag(1 / tmp_array.std(axis=1))
        dict_norm['N_std_inv'] = np.diag(tmp_array.std(axis=1))
        dict_norm['N_var'] = np.diag(1 / tmp_array.var(axis=1))
        dict_norm['N_var_inv'] = np.diag(tmp_array.var(axis=1))
        for i in range(tmp_array.shape[0]):
            if (tmp_array[i,:] == tmp_array[i,0]).all():
                dict_norm['N_std'][i,i] = 0
                dict_norm['N_std_inv'][i,i] = 0
                dict_norm['N_var'][i,i] = 0
                dict_norm['N_var_inv'][i,i] = 0
        return dict_norm

    def normalize_points(self, tmp_array, dict_norm):
        assert tmp_array.shape[0] == self.field_param['dim_in'] or tmp_array.shape[0] == self.field_param['dim_out']
        tmp_array2 = np.zeros(tmp_array.shape)
        for i in range(tmp_array.shape[1]):
            tmp_array2[:,i] = dict_norm['N_std'] @ (tmp_array[:, i] - dict_norm['N_mue'].flatten())
        return tmp_array2

    def un_normalize_points(self, tmp_array, dict_norm):
        assert tmp_array.shape[0] == self.field_param['dim_in'] or tmp_array.shape[0] == self.field_param['dim_out']
        tmp_array2 = np.zeros(tmp_array.shape)
        for i in range(tmp_array.shape[1]):
            tmp_array2[:,i] = (dict_norm['N_std_inv'] @ tmp_array[:, i]) + dict_norm['N_mue'].flatten()
        return tmp_array2

    def un_normalize_std(self, tmp_array, dict_norm):
        assert tmp_array.shape[0] == self.field_param['dim_in'] or tmp_array.shape[0] == self.field_param['dim_out']
        tmp_array2 = np.zeros(tmp_array.shape)
        for i in range(tmp_array.shape[1]):
            tmp_array2[:,i] = np.diag(dict_norm['N_std_inv'] @ np.diag(tmp_array[:, i]**2) @ dict_norm['N_std_inv'])
        return np.sqrt(tmp_array2)

    def define_observations(self, X_points, flag_unconstrained=None):
        """Use ODE model to compute acceleration at points
        :param X_points: 
        :return Y_poin
        """
        Y_points = np.zeros((self.field_param['dim_out'] * X_points.shape[1], 1))
        for ii in np.arange(X_points.shape[1]):
            Y_points[self.field_param['dim_out'] * ii: self.field_param['dim_out'] * (ii + 1)] = \
                self.func_ODE(0, X_points[:, ii], self.field_param['params'], flag_unconstrained).reshape((self.field_param['dim_out'], 1))
        return Y_points

    def restack_prediction_list(self, list_result):
        """Currently the ouput of some GP functions is a list of arrays. This functions stacks these arrays such that
        an array of size D_out x number_points is obtained. """
        tmp = list_result[0]
        for i in np.arange(1, self.field_param['dim_out']):
            tmp = np.hstack((tmp, list_result[i]))
        return tmp.T

    def compute_constraint_error(self, X, Y, theta):
        assert (X.shape[1] == Y.shape[1])
        constraint_error = np.zeros((X.shape[1]))
        for i in range(X.shape[1]):
            constraint_error[i] = (np.abs(self.constraint_A(X[:, i],theta)@Y[:,[i]] - self.constraint_b(X[:, i],theta))).flatten()
        return constraint_error

    def compute_trajectory(self, t_steps, X0, func_ODE, params):
        print('Start Solve ODE step-wise')
        XXy = np.zeros((len(X0), t_steps.shape[0]))
        for i in range(t_steps.shape[0]):
            XX0 = solve_ivp(lambda t, y: func_ODE(t, y, params), [t_steps[i], t_steps[i] + t_steps[1]], X0,
                            t_eval=[t_steps[i] + t_steps[1]], method='RK45')  # 'RK45','Radau','BDF','LSODA'
            X0 = XX0.y.flatten()  # Define new initial state
            XXy[:, i] = X0.flatten()
        print('Finished Solve ODE step-wise')
        return XXy

