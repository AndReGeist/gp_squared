# Copyright 2019 Max Planck Society. All rights reserved.

from class_GP import *

class subclass_AMGP_normalized(class_GP):
    def __init__(self, Field, flag_mean_prior=False):
        super().__init__(Field)
        self.gp_type = 'AMGP'
        self.constraint_A = Field.constraint_A
        self.constraint_b = Field.constraint_b
        self.func_M = Field.func_M
        self.covariance = self.covariance_AMGP

        self.flag_mean_prior = flag_mean_prior
        self.mean_prior = Field.func_ODE
        self.mean = self.mean_transformed

    def mean_transformed(self, X, theta):
        '''
        Computes mean of constrained GP as L(x)*b(x)
        :param X: matrix of NORMALIZED input state vectors
        :return:
        '''
        mean_vec = np.zeros((self.Dout * X.shape[1], 1))
        for i in range(X.shape[1]):
            if self.field_param['flag_normalize_in'] == True:
                X_tmp = (self.dict_norm_X['N_std_inv'] @ X) + self.dict_norm_X['N_mue']
            else:
                X_tmp = X

            if self.flag_mean_prior == False:
                mean_prior = np.zeros((self.Dout * X.shape[1], 1))
            elif self.flag_mean_prior == True:
                mean_prior = np.zeros((self.Dout * X_tmp.shape[1], 1))
                for ii in np.arange(X_tmp.shape[1]):
                    mean_prior[self.Dout* ii: self.Dout * (ii + 1)] = \
                        self.mean_prior(0, X_tmp[:, ii], theta[-len(self.field_param['params']):], 'prior').reshape((self.Dout, 1))

            A1 = self.constraint_A(X_tmp[:, i], theta)
            M1 = self.func_M(X_tmp[:, i], theta)
            MinvAT1 = np.linalg.solve(M1, A1.T)
            AMinvAT1 = np.dot(A1, MinvAT1)

            if self.field_param['flag_normalize_out'] == True:
                if AMinvAT1 < 1e-14:
                    mean_vec[(self.Dout * i):(self.Dout * i + self.Dout)] = \
                    np.einsum('ij,jk', self.dict_norm_Y['N_std'], mean_prior[(self.Dout * i):(self.Dout * (i + 1))])
                else:
                    #MinvATA1 = np.linalg.solve(M1, np.einsum('ji,jk', A1, A1))  # dot(Minv ,dot(A.T, A))
                    MinvATA1 = np.einsum('ij,jk', MinvAT1, A1)  # dot(Minv ,dot(A.T, A))
                    L1 = (1 / AMinvAT1) * MinvAT1
                    T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
                    mean_vec[(self.Dout * i):(self.Dout * (i + 1))] = \
                        (np.einsum('ij,jk', self.dict_norm_Y['N_std'], L1 * self.constraint_b(X_tmp[:, i], theta) +
                         np.einsum('ij,jk', T1, mean_prior[(self.Dout * i):(self.Dout * (i + 1))]) - self.dict_norm_Y['N_mue'])).reshape((self.Dout, 1))
            else:
                if AMinvAT1 < 1e-14:
                    mean_vec[(self.Dout * i):(self.Dout * i + self.Dout)] = mean_prior[(self.Dout * i):(self.Dout * (i + 1))]
                else:
                    mean_vec[(self.Dout * i):(self.Dout * i + self.Dout)] = \
                        np.reshape((1 / AMinvAT1) * MinvAT1 * self.constraint_b(X_tmp[:, i], theta), (self.Dout, 1)) + \
                        np.einsum('ij,jk', T1, mean_prior[(self.Dout * i):(self.Dout * (i + 1))])
        return mean_vec

    def mean_different_theta(self, X, theta):
        '''
        Computes mean of a constrained GP as L(x)*b(x) with a different theta
        :param X: matrix of NORMALIZED input state vectors
        :return:
        '''
        mean_vec = np.zeros((self.Dout * X.shape[1], 1))
        for i in range(X.shape[1]):
            if self.field_param['flag_normalize_in'] == True:
                X_tmp = (self.dict_norm_X['N_std_inv'] @ X) + self.dict_norm_X['N_mue']
            else:
                X_tmp = X

            if self.flag_mean_prior == False:
                mean_prior = np.zeros((self.Dout * X.shape[1], 1))
            elif self.flag_mean_prior == True:
                mean_prior = np.zeros((self.Dout * X_tmp.shape[1], 1))
                for ii in np.arange(X_tmp.shape[1]):
                    mean_prior[self.Dout * ii: self.Dout * (ii + 1)] = \
                        self.mean_prior(0, X_tmp[:, ii], self.theta_different[-len(self.field_param['params']):], 'prior').reshape(
                            (self.Dout, 1))

            A1 = self.constraint_A(X_tmp[:, i], self.theta_different)
            M1 = self.func_M(X_tmp[:, i], self.theta_different)
            # AMinvAT1 = np.einsum('ij,jk', A1, np.einsum('ij,kj', Minv1, A1))**2  # dot(A, dot(Minv, A.T))
            # MinvAT1 = np.einsum('ij,kj', Minv1, A1)  # dot(Minv ,dot(A.T, A))
            MinvAT1 = np.linalg.solve(M1, A1.T)
            AMinvAT1 = np.dot(A1, MinvAT1)

            if self.field_param['flag_normalize_out'] == True:
                if AMinvAT1 < 1e-14:
                    mean_vec[(self.Dout * i):(self.Dout * i + self.Dout)] = \
                        np.einsum('ij,jk', self.dict_norm_Y['N_std'], mean_prior[(self.Dout * i):(self.Dout * (i + 1))])
                else:
                    MinvATA1 = np.linalg.solve(M1, np.einsum('ji,jk', A1, A1))  # dot(Minv ,dot(A.T, A))
                    L1 = (1 / AMinvAT1) * MinvAT1
                    T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
                    mean_vec[(self.Dout * i):(self.Dout * (i + 1))] = \
                        (np.einsum('ij,jk', self.dict_norm_Y['N_std'], L1 * self.constraint_b(X_tmp[:, i], theta) +
                                   np.einsum('ij,jk', T1, mean_prior[(self.Dout * i):(self.Dout * (i + 1))]) -
                                   self.dict_norm_Y['N_mue'])).reshape((self.Dout, 1))
            else:
                if AMinvAT1 < 1e-14:
                    mean_vec[(self.Dout * i):(self.Dout * i + self.Dout)] = mean_prior[
                                                                            (self.Dout * i):(self.Dout * (i + 1))]
                else:
                    mean_vec[(self.Dout * i):(self.Dout * i + self.Dout)] = \
                        np.reshape((1 / AMinvAT1) * MinvAT1 * self.constraint_b(X_tmp[:, i], theta), (self.Dout, 1)) + \
                        np.einsum('ij,jk', T1, mean_prior[(self.Dout * i):(self.Dout * (i + 1))])
        return mean_vec

    def covariance_corr(self, X1, X2, theta):
        # dot(A, B) <-> np.einsum('ij,jk', A, B)
        # dot(A, B.T) <-> np.einsum('ij,kj', A, B)
        # dot(A.T, B) <-> np.einsum('ji,jk', A, B)
        K = self.covariance_same_same_scalar(X1, X2, theta)

        #if self.field_param['flag_normalize_in'] == True:
        X1_tmp = (self.dict_norm_X['N_std_inv'] @ X1) + self.dict_norm_X['N_mue']

        A1 = self.constraint_A(X1_tmp[:, 0], theta)
        Minv1 = np.linalg.inv(self.func_M(X1_tmp, theta))
        AMinvAT1 = np.einsum('ij,jk', A1, np.einsum('ij,kj', Minv1, A1))  # dot(A, dot(Minv, A.T))
        MinvATA1 = np.einsum('ij,jk', Minv1, np.einsum('ji,jk', A1, A1))  # dot(Minv ,dot(A.T, A))

        if self.field_param['flag_normalize_out'] == True:
            if AMinvAT1 > 1e-14:
                T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
                return K * np.einsum('ij,jk', self.dict_norm_Y['N_std'], np.einsum('ij,jk', T1, self.dict_norm_Y['N_std_inv']))
            else:
                return K * np.eye(self.Dout, self.Dout)
        else:
            if AMinvAT1 > 1e-14:
                return K * (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
            else:
                return K * np.eye(self.Dout, self.Dout)

    def covariance_AMGP(self, X1, X2, theta):
        """Standard GP^2 covariance function"""
        # dot(A, B) <-> np.einsum('ij,jk', A, B)
        # dot(A, B.T) <-> np.einsum('ij,kj', A, B)
        # dot(A.T, B) <-> np.einsum('ji,jk', A, B)

        #if self.field_param['flag_normalize_in'] == True:
        X1_tmp = (self.dict_norm_X['N_std_inv'] @ X1) + self.dict_norm_X['N_mue']
        X2_tmp = (self.dict_norm_X['N_std_inv'] @ X2) + self.dict_norm_X['N_mue']

        A1 = self.constraint_A(X1_tmp[:, 0], theta)
        A2 = self.constraint_A(X2_tmp[:, 0], theta)

        M1 = self.func_M(X1_tmp, theta)
        M2 = self.func_M(X2_tmp, theta)
        # AMinvAT1 = np.einsum('ij,jk', A1, np.linalg.solve(M1, A1.T))  # dot(A, dot(Minv, A.T))
        # AMinvAT2 = np.einsum('ij,jk', A2, np.linalg.solve(M2, A2.T))
        # MinvATA1 = np.linalg.solve(M1, np.einsum('ji,jk', A1, A1))  # dot(Minv ,dot(A.T, A))
        # MinvATA2 = np.linalg.solve(M2, np.einsum('ji,jk', A2, A2))

        MinvAT1 = np.linalg.solve(M1, A1.T)  # dot(A, dot(Minv, A.T))
        MinvAT2 = np.linalg.solve(M2, A2.T)

        AMinvAT1 = np.einsum('ij,jk', A1, MinvAT1)  # dot(A, dot(Minv, A.T))
        AMinvAT2 = np.einsum('ij,jk', A2, MinvAT2)
        MinvATA1 = np.einsum('ij,jk', MinvAT1, A1)  # dot(Minv ,dot(A.T, A))
        MinvATA2 = np.einsum('ij,jk', MinvAT2, A2)

        K = self.covariance_same_same_scalar(X1, X2, theta)

        if (AMinvAT1 > 1e-14 and AMinvAT2 > 1e-14):
            T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
            T2 = (np.eye(self.Dout) - ((1 / AMinvAT2) * MinvATA2))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std'], np.einsum('ij,jk', T1,
                                 np.einsum('ij,jk', self.dict_norm_Y['N_var_inv'], np.einsum('ji,jk', T2, self.dict_norm_Y['N_std']))))
        elif AMinvAT1 > 1e-14:
            T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std'], np.einsum('ij,jk', T1, self.dict_norm_Y['N_std_inv']))
        elif AMinvAT2 > 1e-14:
            T2 = (np.eye(self.Dout) - ((1 / AMinvAT2) * MinvATA2))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std_inv'], np.einsum('ji,jk', T2, self.dict_norm_Y['N_std']))
        else:
            return K * np.eye(self.Dout, self.Dout)

    def covariance_AMGP_different_theta(self, X1, X2, theta):
        """Covariance function required to compute the acceleration of  a different constraint configuration"""
        # dot(A, B) <-> np.einsum('ij,jk', A, B)
        # dot(A, B.T) <-> np.einsum('ij,kj', A, B)
        # dot(A.T, B) <-> np.einsum('ji,jk', A, B)

        #if self.field_param['flag_normalize_in'] == True:
        X1_tmp = (self.dict_norm_X['N_std_inv'] @ X1) + self.dict_norm_X['N_mue']
        X2_tmp = (self.dict_norm_X['N_std_inv'] @ X2) + self.dict_norm_X['N_mue']

        A1 = self.constraint_A(X1_tmp[:, 0], self.theta_different)
        A2 = self.constraint_A(X2_tmp[:, 0], self.theta_different)

        M1 = self.func_M(X1_tmp, self.theta_different)
        M2 = self.func_M(X2_tmp, self.theta_different)
        AMinvAT1 = np.einsum('ij,jk', A1, np.linalg.solve(M1, A1.T))  # dot(A, dot(Minv, A.T))
        AMinvAT2 = np.einsum('ij,jk', A2, np.linalg.solve(M2, A2.T))
        MinvATA1 = np.linalg.solve(M1, np.einsum('ji,jk', A1, A1))  # dot(Minv ,dot(A.T, A))
        MinvATA2 = np.linalg.solve(M2, np.einsum('ji,jk', A2, A2))

        K = self.covariance_same_same_scalar(X1, X2, theta)

        if (AMinvAT1 > 1e-14 and AMinvAT2 > 1e-14):
            T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
            T2 = (np.eye(self.Dout) - ((1 / AMinvAT2) * MinvATA2))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std'], np.einsum('ij,jk', T1,
                                 np.einsum('ij,jk', self.dict_norm_Y['N_var_inv'], np.einsum('ji,jk', T2, self.dict_norm_Y['N_std']))))
        elif AMinvAT1 > 1e-14:
            T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std'], np.einsum('ij,jk', T1, self.dict_norm_Y['N_std_inv']))
        elif AMinvAT2 > 1e-14:
            T2 = (np.eye(self.Dout) - ((1 / AMinvAT2) * MinvATA2))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std_inv'], np.einsum('ji,jk', T2, self.dict_norm_Y['N_std']))
        else:
            return K * np.eye(self.Dout, self.Dout)

    def covariance_AMGP_different_theta_corr(self, X1, X2, theta):
        """Covariance function required to compute the unconstraint acceleration"""
        # dot(A, B) <-> np.einsum('ij,jk', A, B)
        # dot(A, B.T) <-> np.einsum('ij,kj', A, B)
        # dot(A.T, B) <-> np.einsum('ji,jk', A, B)

        #if self.field_param['flag_normalize_in'] == True:
        X1_tmp = (self.dict_norm_X['N_std_inv'] @ X1) + self.dict_norm_X['N_mue']
        X2_tmp = (self.dict_norm_X['N_std_inv'] @ X2) + self.dict_norm_X['N_mue']

        A1 = self.constraint_A(X1_tmp[:, 0], theta)
        A2 = self.constraint_A(X2_tmp[:, 0], self.theta_different)

        M1 = self.func_M(X1_tmp, theta)
        M2 = self.func_M(X2_tmp, self.theta_different)
        AMinvAT1 = np.einsum('ij,jk', A1, np.linalg.solve(M1, A1.T))  # dot(A, dot(Minv, A.T))
        AMinvAT2 = np.einsum('ij,jk', A2, np.linalg.solve(M2, A2.T))
        MinvATA1 = np.linalg.solve(M1, np.einsum('ji,jk', A1, A1))  # dot(Minv ,dot(A.T, A))
        MinvATA2 = np.linalg.solve(M2, np.einsum('ji,jk', A2, A2))

        K = self.covariance_same_same_scalar(X1, X2, theta)

        if (AMinvAT1 > 1e-14 and AMinvAT2 > 1e-14):
            T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
            T2 = (np.eye(self.Dout) - ((1 / AMinvAT2) * MinvATA2))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std'], np.einsum('ij,jk', T1,
                                 np.einsum('ij,jk', self.dict_norm_Y['N_var_inv'], np.einsum('ji,jk', T2, self.dict_norm_Y['N_std']))))
        elif AMinvAT1 > 1e-14:
            T1 = (np.eye(self.Dout) - ((1 / AMinvAT1) * MinvATA1))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std'], np.einsum('ij,jk', T1, self.dict_norm_Y['N_std_inv']))
        elif AMinvAT2 > 1e-14:
            T2 = (np.eye(self.Dout) - ((1 / AMinvAT2) * MinvATA2))
            return K * np.einsum('ij,jk', self.dict_norm_Y['N_std_inv'], np.einsum('ji,jk', T2, self.dict_norm_Y['N_std']))
        else:
            return K * np.eye(self.Dout, self.Dout)





