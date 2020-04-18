# Copyright 2019 Max Planck Society. All rights reserved.

from class_ODE import *

class subclass_ODE_mass_on_surface(class_ODE):
    def __init__(self, flag_control):
        super().__init__(flag_control)
        self.field_param = self.set_field_param(flag_control)

    def set_field_param(self, flag_control):
        if flag_control == False:
            field_param = {'lim_min': [-2, -2, -0.01, -0.5, -0.5, -0.5],
                                'lim_max': [2, 2, 0.01, 0.5, 0.5, 0.5],  # x, y, x_dot, y_dot, z_dot
                                'lim_num': [10, 10, 1, 5, 5, 1],
                           'ode_flag': 'mass_surface',
                           'dim_in': 6, 'dim_out': 3,
                           'flag_control': False,  # Changes the ODE for data generation
                           'flag_lengthscales': 'same',  # Either: 'all' or 'same'
                           'flag_signalvar': 'same',   # Either: 'all' or 'same'
                           'flag_normalize_in': True,
                           'flag_normalize_out': True,
                           'params': [0.08, 0.05, 0.05, 0, 0, 0.1, 3]}

        elif flag_control == True:
            field_param = {'lim_min': [-2, -2, -0.01, -0.5, -0.5, -0.5, -5, -5, -5],
                                'lim_max': [2, 2, 0.01, 0.5, 0.5, 0.5, 5, 5, 5],
                                'lim_num': [3, 3, 1, 3, 3, 1, 3, 3, 3],
                           'ode_flag': 'mass_surface',
                           'dim_in': 9, 'dim_out': 3,
                           'flag_control': True,  # Changes the ODE for data generation
                           'flag_lengthscales': 'same',  # Either: 'all' or 'same'
                           'flag_signalvar': 'same',  # Either: 'all' or 'same'
                           'flag_normalize_in': True,
                           'flag_normalize_out': True,
                           'params': [0.08, 0.05, 0.05, 0, 0, 0.1, 3]}  # constraint parameters

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

        assert field_param['lim_num'][2] == 1, 'The third dimension in field_param[lim_num] must be 1 due to constraint removing one DOF'
        assert field_param['lim_num'][5] == 1, 'The sixth dimension in field_param[lim_num] must be 1 due to constraint removing one DOF'

        return field_param


    def compute_points(self, tmp_positions):
        """Given N D-dimensional states use implicit constraint equations (position + velocity constraint if system is
        holonomic; or only velocity constraint if system is non-holonomic) to ensure that points lie inside the
        constrained state space"""
        assert (tmp_positions.shape[0] == 6 or tmp_positions.shape[0] == 9), 'arrays have not the right shape.'
        params = self.field_param['params']
        for i in range(tmp_positions.shape[1]):
            tmp_positions[2,i] = self.func_z(tmp_positions[:, i], params)
            tmp_positions[5,i] = self.func_z_dot(tmp_positions[:, i], params)

            # Assert velocity constraint
            x = tmp_positions[0,i]; y = tmp_positions[1,i]; z = tmp_positions[2,i];
            xdot = tmp_positions[3,i]; ydot = tmp_positions[4,i]; zdot = tmp_positions[5,i]
            assert np.abs(2*params[0]*x*xdot + 2*params[1]*y*ydot + params[2]*xdot*y + params[2]*x*ydot \
               + params[3]*xdot + params[4]*ydot - zdot - params[5]*params[6]*xdot*np.math.sin(params[6]*x)) < 1e-14, \
                'The computed point does not fulfill the velocity constraint?'
        return tmp_positions

    def func_z(self, X, params):
        return params[0]*X[0]**2 + params[1]*X[1]**2 + params[2]*X[0]*X[1] + params[3]*X[0] + params[4]*X[1] + params[5] * np.math.cos(params[6] * X[0])

    def func_z_dot(self, X, params):
        dz_dx = 2 * params[0] * X[0] + params[2] * X[1] + params[3] - params[5]*params[6]*np.math.sin(params[6]*X[0])
        dz_dy = 2*params[1]*X[1] + params[2]*X[0] + params[4]
        return dz_dx*X[3] + dz_dy*X[4]

    def func_M(self, X, theta):
        m = 3
        return np.array([[m, 0, 0], [0, m, 0], [0, 0, m]])

    def constraint_A(self, X, theta):
        L0 = theta[-7]
        L1 = theta[-6]
        L2 = theta[-5]
        L3 = theta[-4]
        L4 = theta[-3]
        L5 = theta[-2]
        L6 = theta[-1]

        a1 = 2 * L0 * X[0] + L2 * X[1] + L3 - L5 * L6 * np.math.sin(L6 * X[0])
        a2 = 2 * L1 * X[1] + L2 * X[0] + L4
        return np.array([a1, a2, -1]).reshape((1, 3))

    def constraint_b(self, X, theta):
        L0 = theta[-7]
        L1 = theta[-6]
        L2 = theta[-5]
        L5 = theta[-2]
        L6 = theta[-1]
        return -2 * L0 * (X[3] ** 2) - 2 * L1 * (X[4] ** 2) - 2 * L2 * X[3] * X[4] + L5*(
                    L6 ** 2) * (X[3] ** 2) * np.math.cos(L6 * X[0])

    def check_if_states_fulfill_constraint(self, X):
        assert (X.shape[0] == 6) or (X.shape[0] == 9)
        for i in range(X.shape[1]):
            assert np.abs(self.func_z(X[:,i], self.field_param['params']) - X[2,i]) < 1e-14, 'State position is not lying on surface'
            assert np.abs(self.func_z_dot(X[:,i], self.field_param['params']) - X[5,i]) < 1e-14, 'State velocity is not tangential to surface'
    
    def compute_energy(self, X_array, m=3):
        assert (X_array.shape[0] == 6 or X_array.shape[0] == 9), 'array has not the right shape.'
        Energy_kin = 0.5 * m * (X_array[3, :] ** 2 + X_array[4, :] ** 2 + X_array[5, :] ** 2)
        Energy_pot = m * 9.81 * (X_array[2, :]) * np.ones(X_array.shape[1])
        Energy_tot = Energy_kin + Energy_pot
        return Energy_kin, Energy_pot, Energy_tot
        
    def func_ODE(self, t, X, p, flag_constrained=None):
        """
        First-order ODE model of particle moving on the surface
        c(x) = L1*X**2 + L2*Y**2 + L3*X*Y + L4*X + L5*Y
        If flag_constrained='a' returns the unconstrained acceleration
        If flag_constrained='a_bar' returns the unconstrained acceleration with nonidealities
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
        g = -9.81
        m = 3

        if self.field_param['flag_control'] == True:
            Fx = X[6]
            Fy = X[7]
            Fz = X[8]
        else:
            Fx = 0
            Fy = 0
            Fz = 0

        M = np.array([[m, 0, 0],
                      [0, m, 0],
                      [0, 0, m]])
        F = np.array([[Fx],
                      [Fy],
                      [m * g + Fz]])
        a = np.linalg.solve(M, F)

        if flag_constrained == 'a' or flag_constrained == 'prior':
            return a.flatten()

        # Damping from Udwadia2002 - On the foundations of analytical mechanics
        a_0 = 0.2
        velocity = np.sqrt(X[3] ** 2 + X[4] ** 2 + X[5] ** 2)
        if velocity == 0:
            a_c_ni = 0
        else:
            a_c_ni = -a_0 * ((velocity ** 2) / np.abs(velocity)) * np.array(
                [[X[3]], [X[4]], [X[5]]])  # Non-ideal acceleration
        #a_c_ni = 0; print('a_c_ni set to 0')

        if flag_constrained == 'a_bar':
            return (a + a_c_ni).flatten()

        A = np.array([[2 * p[0] * X[0] + p[2] * X[1] + p[3] - p[5]*p[6]*np.math.sin(p[6]*X[0]),
                       2 * p[1] * X[1] + p[2] * X[0] + p[4],
                       -1]])
        b = np.array([[-2 * p[0] * (X[3] ** 2) - 2 * p[1] * (X[4] ** 2) - 2 * p[2] * X[3] * X[4] + p[5]*(p[6]**2)*(X[3]**2)*np.math.cos(p[6]*X[0])]])

        # Compute constrained acceleration
        M_invAT = np.linalg.solve(M, A.T)  # np.inv(M)@A.T
        if A.shape[0] == 1:
            if (A @ M_invAT) == 0:
                K = 0
            else:
                K = M_invAT * (1 / (A @ M_invAT))
        else:
            K = M_invAT @ np.linalg.pinv(A @ M_invAT)  # Weighted moore-penrose pseudo inverse

        acc = K @ b + (np.eye(3) - K @ A) @ (a + a_c_ni)
        # Stop simulation if constrained is violated
        assert np.abs(A @ acc - b) < 1e-14, 'Computed acceleration violates constraint equation?'
        return acc.flatten()  # return np.vstack((acc)).flatten()

    def decorator_changeout_for_solveivp(self, func, number_states, flag_u=True, flag_constrained=None):
        def wrapper(t, X, params):
            if flag_u==True:
                X_new = np.hstack((X, params['u'].flatten()))
            else:
                X_new = X.flatten()
            result = func(t, X_new, params['theta'], flag_constrained)
            return np.hstack((X[number_states:2*number_states], result.flatten()))
        return wrapper


