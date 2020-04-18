# Copyright 2019 Max Planck Society. All rights reserved.

from class_ODE import *

class subclass_ODE_duffling_oscillator(class_ODE):
    def __init__(self, flag_control, flag_mean_prior):
        super().__init__(flag_control)
        self.field_param = self.set_field_param(flag_control, flag_mean_prior)

    def set_field_param(self, flag_control, flag_mean_prior):
        if flag_control == False:
            field_param = {'lim_min': [-4, -0.01, -5, -0.01, 0], # x1, x2, x1_dot, x2_dot, t
                                'lim_max': [4, 0.01, 5, 0.01, 5],
                                'lim_num': [10, 1, 10, 1, 10],
                           'ode_flag': 'duffling',
                           'dim_in': 5, 'dim_out': 2,
                           'flag_control': False,  # Changes the ODE for data generation
                           'flag_lengthscales': 'same',  # Either: 'all' or 'same'
                           'flag_signalvar': 'same',   # Either: 'all' or 'same'
                           'flag_normalize_in': True,
                           'flag_normalize_out': True}

            if flag_mean_prior == False:
                field_param['params'] = [1, 0.3, 2*np.pi]  # A0, alpha, omega
            elif flag_mean_prior == True:
                field_param['params'] = [10, 12, 0.1, 0.15, 1, 0.3, 2*np.pi]  # k1, k2, c1, c2, A0, alpha, omega

        elif flag_control == True:
            sys.exit('Control inputs are not available for the closed-loop Duffling oscillator.')

        # Theta, the array of model hyperaparameters, has a predefined structure:
        # theta := LENGTHSCALES -- SIGNAL VARIANCES -- NOISE VARIANCE -- CONSTRAINT PARAMETERS
        if field_param['flag_lengthscales'] == 'all':
            field_param['numb_lengthscales'] = field_param['dim_in']*field_param['dim_out']
        elif field_param['flag_lengthscales'] == 'same':
            field_param['numb_lengthscales'] = field_param['dim_in']
        if field_param['flag_signalvar'] == 'all':
            field_param['numb_signalvar'] = field_param['dim_out']
        if field_param['flag_signalvar'] == 'same':
            field_param['numb_signalvar'] = 1

        assert field_param['lim_num'][1] == 1, 'The second dimension in field_param[lim_num] must be 1 due to constraint removing two DOFs'
        assert field_param['lim_num'][3] == 1, 'The fourth dimension in field_param[lim_num] must be 1 due to constraint removing two DOFs'
        return field_param

    def compute_points(self, tmp_positions):
        """Given N d-dimensional states use implicit constraint equations (position + velocity constraint if system is
        holonomic; or only velocity constraint if system is non-holonomic) to ensure that points lie inside the
        constrained state space"""
        assert tmp_positions.shape[0] == 5, 'arrays have not the right shape.'
        for i in range(tmp_positions.shape[1]):
            tmp_positions[1, i] = self.func_constraint_x2(tmp_positions[:, i], self.field_param['params'])
            tmp_positions[3, i] = self.func_constraint_x2_dot(tmp_positions[:, i], self.field_param['params'])
        self.check_if_states_fulfill_constraint(tmp_positions)
        return tmp_positions

    def func_constraint_x2(self, X, params):
        A0 = params[-3]
        alpha = params[-2]
        omega = params[-1]
        return X[0] - A0*np.math.exp(-alpha*X[4])*np.math.sin(omega*X[4])

    def func_constraint_x2_dot(self, X, params):
        A0 = params[-3]
        alpha = params[-2]
        omega = params[-1]
        return X[2] + A0*alpha*np.math.exp(-alpha*X[4])*np.math.sin(omega*X[4]) \
               - A0 * omega * np.math.exp(-alpha*X[4]) * np.math.cos(omega*X[4])

    def func_M(self, X, theta):
        m1 = 2; m2 = 1
        return np.array([[m1, 0], [0, m2]])

    def constraint_A(self, X, theta):
        return np.array([1, -1]).reshape((1, 2))

    def constraint_b(self, X, theta):
        A0 = theta[-3]
        alpha = theta[-2]
        omega = theta[-1]
        return -A0*np.math.exp(-alpha*X[4]) * ((omega**2)*np.math.sin(omega*X[4])
                                               + 2*omega*alpha*np.math.cos(omega*X[4])
                                               - (alpha**2)*np.math.sin(omega*X[4]))

    def check_if_states_fulfill_constraint(self, X):
        assert (X.shape[0] == 5)
        A0 = self.field_param['params'][-3]
        alpha = self.field_param['params'][-2]
        omega = self.field_param['params'][-1]
        for i in range(X.shape[1]):
            assert np.abs(X[0, i] - X[1, i] - A0*np.math.exp(-alpha*X[4, i])*np.math.sin(omega*X[4, i])) < 1e-14, 'State is not satisfying constraint'
            assert np.abs(X[3, i] - self.func_constraint_x2_dot(X[:, i], self.field_param['params'])) < 1e-10, 'State is not satisfying constraint'

    def func_ODE(self, t, X, params, flag_constrained=None):
        """
        First-order ODE model of particle moving on the surface
        c(x) = L1*X**2 + L2*Y**2 + L3*X*Y + L4*X + L5*Y
        :param X: State vector with
        X[0] := X,
        X[1] := Y,
        X[2] := Z,
        X[3] := X_dot,
        X[4] := Y_dot,
        X[5] := Z_dot,

        X[6] := Fx,
        X[7] := Fy,
        X[8] := Fz

        :return: X_ddot, Y_ddot, Z_ddot
        """
        if X.shape==4:
            X.append(t)
        m1 = 2
        m2 = 1

        k1_nl = 1
        k2_nl = 2
        if flag_constrained == 'prior':
            # [10, 12, 0.1, 0.15, 1, 0.3, 2*np.pi]  # k1, k2, c1, c2, A0, alpha, omega
            k1 = params[-7]
            k2 = params[-6]
            c1 = params[-5]
            c2 = params[-4]
        else:
            k1 = 10; k2 = 12; c1 = 0.1; c2 = 0.15
        K_stiff = np.array([[k1, -k1], [-k1, k1 + k2]])
        C = np.array([[c1, -c1], [-c1, c1 + c2]])

        xp = X[0:2].reshape((2,1))
        xp_dot = X[2:4].reshape((2,1))
        M = self.func_M(X, params)

        F_nonlinear = -1*np.array([[k1_nl*(X[0]-X[1])**3], [k2_nl*(X[1]**3)-k1_nl*(X[0]-X[1])**3]])
        F_linear = -1*(K_stiff@xp + C@xp_dot)

        if flag_constrained == 'prior':
            # Prior model of the linear damping and springs
            a_prior = np.linalg.solve(M, F_linear)
            return a_prior.flatten()

        a = np.linalg.solve(M, F_linear + F_nonlinear)

        if flag_constrained == 'a' or flag_constrained == 'a_bar':
            return a.flatten()

        A = self.constraint_A(X, params)
        b = np.array([[self.constraint_b(X, params)]])

        # Compute constrained acceleration
        M_invAT = np.linalg.solve(M, A.T)  # np.inv(M)@A.T
        if A.shape[0] == 1:
            if (A @ M_invAT) == 0:
                K = 0
            else:
                K = M_invAT * (1 / (A @ M_invAT))
        else:
            K = M_invAT @ np.linalg.pinv(A @ M_invAT)  # Weighted moore-penrose pseudo inverse

        acc = K @ b + (np.eye(2) - K @ A) @ a
        acc_test = a + (m1*m2/(m1+m2))*np.array([[1/m1], [-1/m2]])@(b-a[0,0]+a[1,0])  # Final equation from UKE book, p. 122
        assert np.abs(acc[0,0]-acc_test[0,0] + acc[1,0]-acc_test[1,0]) < 1e-13, 'Computed acceleration violates constraint equation?'
        # Stop simulation if constrained is violated
        assert np.abs(A @ acc - b) < 1e-13, 'Computed acceleration violates constraint equation?'
        return acc.flatten()

    def decorator_changeout_for_solveivp(self, func, number_states, flag_u=None, flag_constrained=None):
        def wrapper(t, X, params):
            X_new = np.hstack((X, t))
            result = func(t, X_new, params['theta'], flag_constrained)
            return np.hstack((X[number_states:2*number_states], result.flatten()))
        return wrapper



