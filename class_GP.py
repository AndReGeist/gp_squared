# Copyright 2019 Max Planck Society. All rights reserved.

from scipy import optimize
import time
from plot_scripts import *

class class_GP:
    def __init__(self, Field):
        self.field_param = Field.field_param
        self.gp_type = 'indep'
        self.sim_result = []  # Initialize for callback function
        self.Dout = Field.field_param['dim_out']
        self.Din = Field.field_param['dim_in']

        self.num_lengthscales = Field.field_param['numb_lengthscales']
        self.flag_lengthscales = Field.field_param['flag_lengthscales']

        self.numb_signalvar = Field.field_param['numb_signalvar']
        self.flag_signalvar = Field.field_param['flag_signalvar']

        if self.field_param['flag_normalize_in'] == True:
            self.dict_norm_X = Field.dict_norm_X
        if self.field_param['flag_normalize_out'] == True:
            self.dict_norm_Y = Field.dict_norm_Y

        self.un_normalize_points = Field.un_normalize_points
        #self.X_traintrain_formatted = Field.X_traintrain_formatted

        self.mean = self.mean_zero
        if Field.field_param['flag_lengthscales'] == 'same' and Field.field_param['flag_signalvar'] == 'same':
            self.covariance = self.covariance_same_same
        elif Field.field_param['flag_lengthscales'] == 'all' and Field.field_param['flag_signalvar'] == 'same':
            self.covariance = self.covariance_all_same
        elif Field.field_param['flag_lengthscales'] == 'same' and Field.field_param['flag_signalvar'] == 'all':
            self.covariance = self.covariance_same_all
        elif Field.field_param['flag_lengthscales'] == 'all' and Field.field_param['flag_signalvar'] == 'all':
            self.covariance = self.covariance_all_all

    def mean_zero(self, X, theta):
        mean_vec = np.zeros((self.Dout * X.shape[1], 1))
        return mean_vec

    def covariance_all_all(self, X, Y, theta):
        """Computes squared exponential covariance function with
        DIFFERENT lengthscale for outputs and
        DIFFERENT signal variances for outputs.
        Returns block diagonal covariance matrix (function)"""
        return np.diag([((theta[self.num_lengthscales + i] ** 2) * np.math.exp(
            np.einsum('ji,jk', X-Y,
                      np.einsum('ij,jk', self.Leng[i], X-Y)))) for i in range(self.Dout)])

    def covariance_same_all(self, X, Y, theta):
        return np.math.exp(np.einsum('ji,jk', (X - Y), np.einsum('ij,jk', self.Leng, (X - Y)))) *\
               np.diag([theta[self.num_lengthscales + i] ** 2 for i in range(self.Dout)])

    def covariance_all_same(self, X, Y, theta):
        return (theta[self.num_lengthscales] ** 2) * \
               np.diag([(np.math.exp(np.einsum('ji,jk', X-Y,
                                               np.einsum('ij,jk', self.Leng[i], X-Y)))) for i in range(self.Dout)])

    def covariance_same_same(self, X, Y, theta):
        """Computes squared exponential covariance function with
                SAME lengthscale for outputs and
                SAME signal variances for outputs.
                Returns block diagonal covariance matrix (function)"""
        return self.covariance_same_same_scalar(X, Y, theta) * np.eye(self.Dout, self.Dout)

    def covariance_same_same_scalar(self, X, Y, theta):
        return (theta[self.num_lengthscales] ** 2) * np.exp(np.einsum('ji,jk', X-Y, np.einsum('ij,jk', self.Leng, X-Y)))

    def compute_cov_matrix(self, theta, data_1, data_2, covariance):
        cov_mat = np.empty((data_1.shape[1] * self.Dout, data_2.shape[1] * self.Dout))
        for ii in range(data_1.shape[1]):
            for jj in range(data_2.shape[1]):
                cov_mat[ii * self.Dout:(ii + 1) * self.Dout, jj * self.Dout:(jj + 1) * self.Dout] = \
                    covariance(data_1[:, [ii]], data_2[:, [jj]], theta)
        return cov_mat

    def reform_covariance(self, X):
        n1 = 2*self.Din; n2 = self.Dout
        X1 = X[:self.Din].reshape((-1,1))
        X2 = X[self.Din:].reshape((-1,1))
        return self.covariance_tmp(X1, X2, self.theta_tmp)

    def covariance_matrix(self, data_1, data_2, theta, covariance, flag_noise=False):
        '''
        Create a covariance matrix using the covariance function "covariance"
        :param data_1: first dataset m x n1
        :param data_2: second dataset m x n2
        :param D: number of GP states for a single state
        assert data_1.shape[0] == 2 and  data_2.shape[0] == 2, 'Data point matrix dimension error!
        '''
        assert data_1.shape[0] == data_2.shape[0] == self.Din, 'Check inputs of covariance_matrix.'
        if self.flag_lengthscales == 'all':
            self.Leng = []
            for i in range(self.Dout):
                self.Leng.append(np.diag((-1 / (2 * theta[(self.Din * i):(self.Din * (i + 1))] ** 2))))
        elif self.flag_lengthscales == 'same':
            self.Leng = np.diag((-1 / (2 * theta[:self.num_lengthscales] ** 2)))
        cov_mat = self.compute_cov_matrix(theta, data_1, data_2, covariance)

        if flag_noise == True:
            index_var = self.num_lengthscales + self.numb_signalvar

            if self.field_param['flag_normalize_out'] == True:
                noise_var = (np.diag(self.dict_norm_Y['N_std'])*theta[index_var:(index_var + self.Dout)])**2
            else:
                noise_var = theta[index_var:(index_var + self.Dout)]**2

            cov_mat = cov_mat + np.diag(np.tile([noise_var], data_1.shape[1]).flatten()) + \
                           1e-16 * np.eye(data_1.shape[1]*self.Dout, data_1.shape[1]*self.Dout)
        return cov_mat

    def update_data(self, X_pred, X_train, Y_train, theta,
                    covariance_XX=None, covariance_TX=None, covariance_TT=None,
                    mean_pred_func=None, flag_Lpred=False):
        '''
        Compute GP posterior
        Credits to: Carl Rasmussen, Katherine Bailey, Nando de Freitas, and Kevin Murphy
        :param X_pred:
        :param X_train:
        :param Y_train:
        :return mue: Mean of posterior distribution
        :return std: Vector of standard deviations
        '''
        if mean_pred_func == None:
            mean_pred_func = self.mean

        mean_train = self.mean(X_train, theta)
        mean_pred = mean_pred_func(X_pred, theta)
        K_TT = self.covariance_matrix(X_train, X_train, theta, covariance_TT,
                                      flag_noise=True)
        K_TX = self.covariance_matrix(X_train, X_pred, theta, covariance_TX)
        K_XX = self.covariance_matrix(X_pred, X_pred, theta, covariance_XX)
        #matrix_show2(K_TT, 'K_TT', np.min(K_TT), np.max(K_TT), 'RdYlGn_r')  # Plots covariance matrix

        L = np.linalg.cholesky(K_TT)
        Lk = np.linalg.solve(L, K_TX)

        # Compute the standard deviation of NOISY posterior
        index_var = self.num_lengthscales + self.numb_signalvar
        if self.field_param['flag_normalize_out'] == True:
            noise_var = (np.diag(self.dict_norm_Y['N_std'])*theta[index_var:(index_var + self.Dout)])**2
        else:
            noise_var = theta[index_var:(index_var + self.Dout)]**2
        matrix_noise_var = np.diag(np.tile([noise_var], X_pred.shape[1]).flatten())
        if flag_Lpred == True:
            L_pred = np.linalg.cholesky(K_XX + 1e-14 * np.eye(X_pred.shape[1]*self.Dout, X_pred.shape[1]*self.Dout) - np.dot(Lk.T, Lk))
        else:
            L_pred = None
        var = np.diag(K_XX - np.sum(Lk ** 2, axis=0) + matrix_noise_var)
        std = np.sqrt(var).reshape(-1, 1)

        # Compute process mean
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, (Y_train - mean_train)))
        mu = mean_pred + np.dot(K_TX.T, alpha)
        mu = mu.reshape(-1, 1)

        # Likelihood
        log_p = -0.5 * np.dot((Y_train - mean_train).T, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * K_TT.shape[
            0] * np.log(2 * np.pi)

        list_mu = self.extract_dims(mu, self.Dout)
        list_std = self.extract_dims(std, self.Dout)
        return {'list_mu': list_mu, 'mu': mu, 'std': std, 'list_std': list_std, 'L_pred': L_pred,
                'gp_type': self.gp_type, 'log_p': log_p, 'K_XX': K_XX, 'K_TT': K_TT, 'theta': theta}

    def update_data_faster(self, X_pred, X_train, Y_train, theta, K_XX):
        '''
        Compute GP posterior
        Credits to: Carl Rasmussen, Katherine Bailey, Nando de Freitas, and Kevin Murphy
        :param X_pred:
        :param X_train:
        :param Y_train:
        :return mue: Mean of posterior distribution
        :return std: Vector of standard deviations
        '''
        mean_pred_func = self.mean

        mean_train = self.mean(X_train, theta)
        mean_pred = mean_pred_func(X_pred, theta)
        K_TT = self.covariance_matrix(X_train, X_train, theta, self.covariance,
                                      flag_noise=True)
        K_TX = self.covariance_matrix(X_train, X_pred, theta, self.covariance)

        L = np.linalg.cholesky(K_TT)
        Lk = np.linalg.solve(L, K_TX)

        # Compute the standard deviation of NOISY posterior
        index_var = self.num_lengthscales + self.numb_signalvar
        if self.field_param['flag_normalize_out'] == True:
            noise_var = (np.diag(self.dict_norm_Y['N_std'])*theta[index_var:(index_var + self.Dout)])**2
        else:
            noise_var = theta[index_var:(index_var + self.Dout)]**2
        matrix_noise_var = np.diag(np.tile([noise_var], X_pred.shape[1]).flatten())
        cov_matrix = K_XX + matrix_noise_var - np.sum(Lk ** 2, axis=0)
        var = np.diag(cov_matrix)
        std = np.sqrt(var).reshape(-1, 1)

        # Compute process mean
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, (Y_train - mean_train)))
        mu = mean_pred + np.dot(K_TX.T, alpha)
        mu = mu.reshape(-1, 1)

        # Likelihood
        log_p = -0.5 * np.dot((Y_train - mean_train).T, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * K_TT.shape[
            0] * np.log(2 * np.pi)

        list_mu = self.extract_dims(self.mu, self.Dout)
        list_std = self.extract_dims(self.std, self.Dout)
        return {'list_mu': list_mu, 'mu':self.mu, 'std': self.std, 'cov_matrix': cov_matrix, 'list_std': list_std, 'gp_type': self.gp_type,
                'log_p': log_p, 'K_XX': K_XX, 'K_TT': K_TT, 'theta': theta}

    def init_update_point(self, params):
        '''
        Initialize mean computation for ODE script
        :param X_pred:
        :param X_train:
        :param Y_train:
        :return mue: Mean of posterior distribution
        :return std: Vector of standard deviations
        '''
        theta = params['theta']
        covariance_TT = params['covariance_TT']
        Field = params['Field']
        #if self.field_param['flag_normalize_in'] == True:
        #    self.dict_norm_X = Field.dict_norm_X
        #if self.field_param['flag_normalize_out'] == True:
        #    self.dict_norm_Y = Field.dict_norm_Y

        self.mean_train = self.mean(Field.X_train, theta)
        K_TT = self.covariance_matrix(Field.X_train, Field.X_train, theta, covariance_TT,
                                      flag_noise=True)
        self.LL = np.linalg.cholesky(K_TT)
        self.LL_transp = self.LL.T

        # Compute the standard deviation of NOISY posterior
        index_var = self.num_lengthscales + self.numb_signalvar
        if self.field_param['flag_normalize_out'] == True:
            noise_var = (np.diag(self.dict_norm_Y['N_std'])*theta[index_var:(index_var + self.Dout)])**2
        else:
            noise_var = theta[index_var:(index_var + self.Dout)]**2
        self.matrix_noise_var = np.diag(np.tile([noise_var], 1).flatten())

    def update_point_predictions(self, X, Field, theta,
                                 covariance_XX,
                                 covariance_TX,
                                 mean_pred_func):
        '''
        Compute GP prediction for plots. Requires to run 'init_update_point' first.
        :param Field:
        :param theta: GP hyperparameters
        :return mue: Mean of posterior distribution
        :return std: Vector of standard deviations
        '''

        K_XX = covariance_XX(X, X, theta)
        mean_pred = mean_pred_func(X, theta)
        K_TX = self.covariance_matrix(Field.X_train, X, theta, covariance_TX)

        Lk = np.linalg.solve(self.LL, K_TX)
        std = np.sqrt(np.diag(K_XX + self.matrix_noise_var - np.sum(Lk ** 2, axis=0)))

        # Compute process mean
        alpha = np.linalg.solve(self.LL_transp, np.linalg.solve(self.LL, (Field.Y_train_noisy - self.mean_train)))
        mu = mean_pred + np.dot(K_TX.T, alpha)
        return np.vstack((mu.flatten(), std)).T  # First column is mu

    def myfunc(self, X):
        '''
        '''
        X = X.reshape(-1, 1)
        Field = self.Field_myfunc
        theta = self.theta_myfunc
        K_XX = self.covariance(X, X, theta)
        mean_pred = self.mean(X, theta)
        K_TX = self.covariance_matrix(Field.X_train, X, theta, self.covariance)

        Lk = np.linalg.solve(self.LL, K_TX)
        std = np.sqrt(np.diag(K_XX + self.matrix_noise_var - np.sum(Lk ** 2, axis=0)))

        # Compute process mean
        alpha = np.linalg.solve(self.LL_transp, np.linalg.solve(self.LL, (Field.Y_train_noisy - self.mean_train)))
        mu = mean_pred + np.dot(K_TX.T, alpha)
        return np.hstack((mu.flatten(), std)).T # First column is mu

    def compute_prediction_for_dataset(self, Field, theta,
                                       covariance_XX,
                                       covariance_TX,
                                       mean_pred_func):
        '''
        Compute GP posterior
        Credits to: Carl Rasmussen, Katherine Bailey, Nando de Freitas, and Kevin Murphy
        :param X_pred:
        :param X_train:
        :param Y_train:
        :return mue: Mean of posterior distribution
        :return std: Vector of standard deviations
        '''
        #tmp_array = np.zeros((Field.X_predict.shape[1]*self.Dout, 2))
        #for i in range(Field.X_predict.shape[1]):
            #tmp_array[i*self.Dout:((i+1)*self.Dout),:] = self.update_point_predictions(Field.X_predict[:,[i]], Field, theta,
            #                                                             covariance_XX, covariance_TX,
            #                                                             mean_pred_func)
        self.Field_myfunc = Field
        self.theta_myfunc = theta
        tmp_array = np.apply_along_axis(self.myfunc, 0, Field.X_predict)
        list_mu = [tmp_array[[i], :].T for i in range(self.Dout)]
        list_std = [tmp_array[[i+self.Dout], :].T for i in range(self.Dout)]
        return {'list_mu': list_mu,
                'list_std': list_std,
                'gp_type': self.gp_type,
                'theta': theta}

    def update_point_ODE(self, t, X, params):
        '''
        Compute GP prediction for ODE script. Requires to run 'init_update_point' first.
        :param X_pred:
        :param X_train:
        :param Y_train:
        :return mue: Mean of posterior distribution
        :return std: Vector of standard deviations
        '''
        theta = params['theta']
        Field = params['Field']
        X = np.concatenate((X, params['u'])).reshape(-1,1)

        X_scaled = Field.normalize_points(X, Field.dict_norm_X)

        mean_pred = self.mean(X_scaled, theta)
        K_TX = self.covariance_matrix(Field.X_train, X_scaled, theta, self.covariance)

        # Compute process mean
        alpha = np.linalg.solve(self.LL_transp, np.linalg.solve(self.LL, (Field.Y_train_noisy - self.mean_train)))
        mu = mean_pred + np.dot(K_TX.T, alpha)
        mu = mu.reshape(-1, 1)

        if Field.field_param['flag_normalize_out'] == True:
            mu_unscaled = Field.un_normalize_points(mu, Field.dict_norm_Y)
        return np.vstack((X[3:6], mu_unscaled)).flatten()

    def negLogLikelihood(self, X_train, data_Y, covariance_func):
        '''Returns a function that computes the negative log-likelihood
         for training data X_train and Y_train'''
        def step(theta):
            mean_train = self.mean(X_train, theta)
            K_TT = self.covariance_matrix(X_train, X_train, theta, covariance_func, flag_noise=True)
            L = np.linalg.cholesky(K_TT)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, (data_Y - mean_train)))
            return 0.5 * np.dot((data_Y - mean_train).T, alpha) + np.sum(np.log(np.diag(L))) +\
                   0.5 * K_TT.shape[0] * np.log(2 * np.pi)
        return step

    def minimize_LogML(self, X_data, Y_data, theta_param, optim_bounds, Field, covariance_func, filename=None):
        '''Minimize the negative logarithmic likelihood of the GP. The optimization is performed
        several times using random initialized hyperparameters
        '''
        if self.field_param['flag_normalize_in'] == True:
            self.dict_norm_X = Field.dict_norm_X
        if self.field_param['flag_normalize_out'] == True:
            self.dict_norm_Y = Field.dict_norm_Y

        theta_train = np.random.sample((theta_param['theta_pts'], len(theta_param['dim_min'])))
        for i in range(theta_param['theta_pts']):
            for j in range(len(theta_param['dim_min'])):
                theta_train[i, j] = (theta_param['dim_max'][j] - \
                                     theta_param['dim_min'][j]) * theta_train[i, j] + theta_param['dim_min'][j]
        list_result = []
        list_log = []
        self.sim_result = []
        self.list_sim_result = []

        for i in range(theta_param['theta_pts']):
            self.Nfeval = 1  # Set optimization iteration counter to 1
            if theta_param['flag_optimizer'] == 'L-BFGS-B':
                result = optimize.minimize(self.negLogLikelihood(X_data, Y_data, covariance_func), theta_train[i, :],
                                           bounds=optim_bounds, method='L-BFGS-B',
                                           callback=self.callbackF, options={'ftol': 1e-6, 'gtol': 1e-04, 'maxiter': 20, 'maxiter': 10}) # options={'ftol': 1e-8, 'gtol': 1e-04}
                list_result.append(result)
                list_log.append(result['fun'])
                self.list_sim_result.append(self.sim_result)  # Save result from optimize.minimze callback
                if filename != None:
                    with open('simulation_data/' + filename + 'info.txt', "a+") as text_file:
                        print('\n' + self.gp_type, 'Theta Nr. ', str(i), ': ', result['fun'], result['x'], file=text_file)
                print('\n' + self.gp_type, 'Theta Nr. ', str(i), ': ', result['fun'], result['x'])
            elif theta_param['flag_optimizer'] == 'DiffEvo':
                result = optimize.differential_evolution(self.negLogLikelihood(X_data, Y_data, covariance_func), bounds=optim_bounds)
                # callback=self.callbackF) #, options={'ftol': 1e-6})
                list_result.append(result)
                list_log.append(result['fun'])
                if filename != None:
                    with open('simulation_data/' + filename + 'info.txt', "a+") as text_file:
                        print('\n' + self.gp_type, 'Theta Nr. ', str(i), ': ', result['fun'], result['x'], file=text_file)
                print('\n' + self.gp_type, 'Theta Nr. ', str(i), ': ', result['fun'], result['x'])

        theta_out = list_result[list_log.index(min(list_log))]
        if filename != None:
            with open('simulation_data/' + filename + 'info.txt', "a+") as text_file:
                print('\n' + 'SUMMARY', file=text_file)
                for i in range(len(list_result)):
                   print(list_result[i].x, file=text_file)
                print('list_log', list_log, file=text_file)
                print('Best theta: ', theta_out.x, theta_out.fun, file=text_file)
        print('\n' + 'SUMMARY')
        for i in range(len(list_result)):
            print(list_result[i].x)
        print('list_log', list_log)
        print('Best theta: ', theta_out.x, theta_out.fun)
        return theta_out

    def callbackF(self, Xi):
        '''Callback function for scipy.optimize,
        saves hyperparameters over optimization iterations'''
        self.Nfeval = self.Nfeval
        self.sim_result.append([self.Nfeval, Xi])
        np.set_printoptions(precision=2, suppress=True)
        print(self.Nfeval, Xi, '\n')  # self.negLogLikelihood(X_train, data_Y)
        self.Nfeval += 1

    def sample_prior(self, X, mean_func, theta, covariance):
        '''
        Draw sample function from Gaussian process
        :return: function prediction
        '''
        assert X.shape[0] == self.Din  # Check if X is a data set
        K_XX = self.covariance_matrix(X, X, theta, covariance) + 1e-12 * np.eye(X.shape[1]*self.Dout, X.shape[1]*self.Dout)
        L_K = np.linalg.cholesky(K_XX)
        mean = mean_func(X, theta)
        f_sample = mean + np.dot(L_K, np.random.normal(size=(K_XX.shape[0], 1)))
        return self.extract_dims_array(f_sample, self.Dout)

    def sample_posterior(self, mean, L_K):
        '''
        Draw sample function from Gaussian process
        :return: function prediction
        '''
        #assert X.shape[0] == self.Din  # Check if X is a data set
        f_sample = mean + np.dot(L_K, np.random.normal(size=(L_K.shape[0], 1)))
        #f_sample = np.random.multivariate_normal(mean.flatten(), L_K).reshape((-1,1))
        return self.extract_dims_array(f_sample, self.Dout)

    def sample_n(self, n, sample_func, *args, **kwargs):
        """
        Generates n samples from a GP (Multivariate Gaussian distribution)
        :param n:
        :param X:
        :param mean:
        :return:
        """
        list_list_samples = []
        for i in range(n):
            list_list_samples.append(sample_func(*args, **kwargs))  # sample() MISSING!
        return list_list_samples

    def extract_dims(self, vec, Ndim):
        '''
        Extracts every dim's (first, second, third, ...) entry from a vector and
        stores it in a list.
        :param vec: vector with sequential vector components, e.g. vec=[x1 y1 x2 y2 x3 y3...]
        :return list_components: list of separated vectors e.g. for Ndim=2,
            list_components=[vec_x, vec_y], vec_x = [x1 x2 x3 ...], vec_y = [y1 y2 y3 ...]
        '''
        assert vec.shape[1] == 1
        list_components = []
        if Ndim == 1:
            list_components.append(vec.reshape(-1, 1))
        else:
            for i in range(Ndim):
                list_components.append(vec[i::self.field_param['dim_out']].reshape(-1, 1))
        return list_components

    def extract_dims_array(self, vec, Ndim):
        '''
        Extracts every dim's (first, second, third, ...) entry from a vector and
        stores it in a array.
        :param vec: vector with sequential vector components, e.g. vec=[x1 y1 x2 y2 x3 y3...]
        '''
        assert vec.shape[1] == 1
        if Ndim == 1:
            vec_array = vec.reshape(1, -1)
        else:
            vec_array = vec[0::self.field_param['dim_out']].reshape(1, -1)
            for i in range(1, Ndim):
                vec_array = np.vstack((vec_array, vec[i::self.field_param['dim_out']].reshape(1, -1)))
        return vec_array

    def plot_likelihood(self, X_train, data_Y):
        hyper_min = [0.1, 0.1, 0.1, 0.1, 0.1, 0]
        hyper_max = [1, 1, 1, 1, 1, 1, 0]
        hyper_num = [3, 3, 3, 3, 3, 1]
        allT = [np.logspace(hyper_min[ii], hyper_max[ii], hyper_num[ii])
                for ii in range(len(hyper_num))]
        TT = np.meshgrid(*allT)
        self.thetas = np.vstack(map(np.ravel, TT))
        log_array = np.zeros((self.thetas.shape[1]))
        for i in range(self.thetas.shape[1]):
            print('Plot likelihood:' + str(i + 1) + ' out of ' + str(self.thetas.shape[1]))
            # params = {'lengthscales': self.thetas[:4,i],
            #          'sigma': self.thetas[4,i], 'std_obs': 0}
            print('hyperparameter: /n', self.thetas[:, i])
            log_array[i] = self.negLogLikelihood_theta(X_train, data_Y, self.thetas[:, i])

        fig, ax = plt.subplots()
        fig.canvas.draw()
        plt.title('Negative Log ML')
        plt.plot(log_array, markersize=10, linewidth=2, marker='s', linestyle='None', label='Negative Log ML')
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # for i in range(len(labels)):
        #    labels[i] = '[' + str(self.thetas[:,i]) + ']'
        # ax.set_xticklabels(labels)
        # ax.set_ylim(ymin=0.00001)
        plt.legend(loc='upper right')
        plt.xlabel('input point number')
        plt.ylabel('Log ML')
        plt.show()

    def plot_optimization_result(self, label_thetas):
        print('self.GP.AMGP.sim_result')
        print(self.sim_result)
        plt.figure()
        plt.title('title')
        tmp = [[] for i in range(len(self.sim_result[0][1]))]  # Number of hyperpara
        for i in range(len(self.sim_result)):  # Number of optimization steps
            for j in range(len(self.sim_result[0][1])):
                tmp[j].append(self.sim_result[i][1][j])
        for j in range(len(self.sim_result[0][1])):
            plt.plot(tmp[j], markersize=5, linewidth=2, label=label_thetas[j])
        # markersize = 2 * (len(self.sim_result[0]) + 1 - j)
        plt.grid()
        plt.legend(loc='upper right')
        plt.xlabel('input point number')
        plt.ylabel('output error values')
        plt.show()


    def adjust_GPy_ARD_for_solveivp(self, t, X, params):
        """Adjust GPy's standard GP predict function to fit into scipy.solve_ivp"""
        X = np.hstack((X, params['u'].flatten())).reshape((-1,1))
        if params['flag_constrain_GPy_estimate'] == True:
            X = params['Field'].compute_points(X)  # This line enforces the constraints heuristically
        X_new = params['Field'].normalize_points(X, params['Field'].dict_norm_X)
        r1, v1 = params['m1'].predict(X_new.T)
        r2, v2 = params['m2'].predict(X_new.T)
        r3, v3 = params['m3'].predict(X_new.T)
        acc = params['Field'].un_normalize_points(np.array((r1[0], r2[0], r3[0])), params['Field'].dict_norm_Y)
        return np.hstack((X[3:6].flatten(), acc.flatten()))

    def adjust_GPy_multi_for_solveivp(self, t, X, params):
        """Adjust GPy's multioutput GP predict function to fit into scipy.solve_ivp"""
        X_new = np.hstack((X, params['u'].flatten())).reshape((-1,1)).T
        r1, v1 = params['m'].predict(X_new)
        return np.hstack((X[3:6], r1.flatten()))
