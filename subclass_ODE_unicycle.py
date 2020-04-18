# Copyright 2019 Max Planck Society. All rights reserved.

from class_ODE import *

class subclass_ODE_unicycle(class_ODE):
    def __init__(self, flag_control):
        super().__init__(flag_control)
        self.field_param = self.set_field_param(flag_control)

    def set_field_param(self, flag_control):
        if flag_control == False:
            field_param = {'lim_min': [0, 0, 0, -0.5],  # x_dot, y_dot, theta, theta_dot
                            'lim_max': [1, 1, 2*np.pi, 0.5],
                            'lim_num': [5, 1, 8, 5],
                           'ode_flag': 'unicycle',
                           'dim_in': 4, 'dim_out': 3,
                           'flag_control': False,  # Changes the ODE for data generation
                           'flag_lengthscales': 'same',  # Either: 'all' or 'same'
                           'flag_signalvar': 'same',   # Either: 'all' or 'same'
                           'flag_normalize_in': True,
                           'flag_normalize_out': True,
                           'params': [0.05, 0.02]}  # constraint parameters: R, Ic

        elif flag_control == True:
            field_param = {'lim_min': [0, 0, 0, -0.5, -1, -0.5],
                           'lim_max': [1, 1, 2*np.pi, 0.5, 1, 0.5],
                           'lim_num': [5, 1, 8, 5, 5, 5],
                           'ode_flag': 'unicycle',
                           'dim_in': 6, 'dim_out': 3,
                           'flag_control': True,  # Changes the ODE for data generation
                           'flag_lengthscales': 'same',  # Either: 'all' or 'same'
                           'flag_signalvar': 'same',  # Either: 'all' or 'same'
                           'flag_normalize_in': True,
                           'flag_normalize_out': True,
                           'params': [0.05, 0.02]}

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

        assert field_param['lim_num'][1] == 1, 'The second dimension in field_param[lim_num] must be 1 due to constraint removing one DOF'
        return field_param


    def compute_points(self, tmp_positions):
        """Given N D-dimensional states use implicit constraint equations (position + velocity constraint if system is
        holonomic; or only velocity constraint if system is non-holonomic) to ensure that points lie inside the
        constrained state space"""
        assert (tmp_positions.shape[0] == 4 or tmp_positions.shape[0] == 6), 'arrays have not the right shape.'

        tmp = np.zeros((tmp_positions.shape[0], tmp_positions.shape[1]))
        for i in range(tmp_positions.shape[1]):
            tmp[0, i] = tmp_positions[0, i] * np.math.cos(tmp_positions[2, i])  # x_dot
            tmp[1, i] = tmp_positions[0, i] * np.math.sin(tmp_positions[2, i])  # y_dot
            tmp[2:, i] = tmp_positions[2:, i]  # theta
        self.check_if_states_fulfill_constraint(tmp)
        return tmp

    def func_M(self, X, theta):
        m = 1
        r = theta[-2]
        Ic = theta[-1]
        return np.array([[m, 0, -m * r * np.math.sin(X[2])],
                         [0, m, m * r * np.math.cos(X[2])],
                         [-m * r * np.math.sin(X[2]), m * r * np.math.cos(X[2]), Ic]])

    def constraint_A(self, X, theta):
        return np.array([[np.math.tan(X[2]) * (np.math.cos(X[2]) ** 2)], [-(np.math.cos(X[2]) ** 2)], [0]]).T

    def constraint_b(self, X, theta):
        return -X[0] * X[3]

    def check_if_states_fulfill_constraint(self, X):
        assert (X.shape[0] == 4) or (X.shape[0] == 6)
        for i in range(X.shape[1]):
            assert np.abs(X[1, i] * np.math.cos(X[2, i]) - X[0, i] * np.math.sin(X[2, i])) < 1e-14, \
                'System velocity is not lying on surface'

    def check_if_Y_fulfill_constraint(self, X, Y, theta):
        assert (X.shape[0] == 4) or (X.shape[0] == 6)
        assert (X.shape[1] == Y.shape[1])
        for i in range(X.shape[1]):
            assert (self.constraint_A(X[:, i],theta)@Y[:,[i]] - self.constraint_b(X[:, i],theta)) < 1e-14, \
                'Constraining equation not fulfilled'

    def func_ODE(self, t, X, params, flag_constrained=None):
        """
        First-order ODE model of simple unicycle dynamics
        :param X: State vector with
        X[0] := x_dot,
        X[1] := y_dot,
        X[2] := theta,
        X[3] := theta_dot

        Optional:
        X[4] := F, Force control
        X[5] := T, Torque control
        :param params: System Parameters
        :return:
        acc:
        acc[0] := x_ddot
        acc[1] := y_ddot
        acc[2] := theta_ddot
        """
        m = 1
        r = params[0]
        Ic = params[1]

        if self.field_param['flag_control'] == True:
            Fx = X[4] * np.math.cos(X[2])
            Fy = X[4] * np.math.sin(X[2])
            Tc = X[5]
        else:
            Fx = 0
            Fy = 0
            Tc = 0

        # Unconstrained acceleration of system
        M = np.array([[m, 0, -m * r * np.math.sin(X[2])],
                      [0, m, m * r * np.math.cos(X[2])],
                      [-m * r * np.math.sin(X[2]), m * r * np.math.cos(X[2]), Ic]])
        F = np.array([[Fx + m * r * X[3] ** 2 * np.math.cos(X[2])],
                      [Fy + m * r * X[3] ** 2 * np.math.sin(X[2])],
                      [Tc]])
        a = np.linalg.solve(M, F)  # Unconstrained acceleration

        if flag_constrained == 'a' or flag_constrained == 'prior':
            return a.flatten()

        # Velocity-Quadratic Damping
        a_0 = 0.5  # Damping coefficient (must be positive)
        velocity = np.sqrt(X[0] ** 2 + X[1] ** 2)
        if velocity == 0:
            a_c_ni = 0
        else:
            a_c_ni = -a_0 * ((velocity ** 2) / np.abs(velocity)) * np.array(
                [[X[0]], [X[1]], [0]])  # Non-ideal acceleration
        #a_c_ni = 0; print('a_c_ni set to 0')

        if flag_constrained == 'a_bar':
            return (a + a_c_ni).flatten()

        A = np.array([[np.math.tan(X[2]) * (np.math.cos(X[2]) ** 2), -(np.math.cos(X[2]) ** 2), 0]])
        b = np.array([[-X[0] * X[3]]])
        M_invAT = np.linalg.solve(M, A.T)
        if A.shape[0] == 1:
            if (A @ M_invAT) == 0:
                K = 0
            else:
                K = M_invAT * (1 / (A @ M_invAT))
        else:
            K = M_invAT @ np.linalg.pinv(A @ M_invAT)  # Weighted moore-penrose pseudo inverse

        # Stop simulation if constrained is violated
        acc = (K @ b + (np.eye(3) - K @ A) @ (a + a_c_ni)).flatten()
        assert np.abs(A @ acc - b) < 1e-14, 'Computed acceleration violates constraint equation.'
        return acc

    def decorator_changeout_for_solveivp(self, func, number_states, flag_u=True, flag_constrained=None):
        def wrapper(t, X, params):
            if flag_u==True:
                X_new = np.array([X[3], X[4], X[2], X[5], params['u'][0], params['u'][1]])
            else:
                X_new = np.array([X[3], X[4], X[2], X[5]])
            result = func(t, X_new, params['theta'], flag_constrained)
            return np.hstack((X[number_states:2*number_states], result.flatten()))
        return wrapper