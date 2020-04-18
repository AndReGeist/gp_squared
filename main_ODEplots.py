from subclass_ODE_mass_on_surface import *
from subclass_ODE_unicycle import *
from subclass_ODE_duffling_oscillator import *

# SYSTEM-settings
#flag_ode = 'mass_surface'
#flag_ode = 'unicycle'
flag_ode = 'duffling'

flag_constrained = None
#flag_constrained = 'a'

if flag_ode == 'unicycle':
    flag_control = True
    Field = subclass_ODE_unicycle(flag_control)
    x0 = 0; y0 = 0; theta0 = 0 * (np.pi / 180);
    dx0 = 0.1; dy0 = 0.; dtheta0 = -0.5
    X0_tmp = [dx0, dy0, theta0, dtheta0]
    u = np.array([0, 0.01])
    X0_tmp = Field.compute_points(np.array(X0_tmp).reshape(-1, 1)).flatten()  # the initial condition
    X0 = np.array([x0, y0, X0_tmp[2], X0_tmp[0], X0_tmp[1], X0_tmp[3]])
elif flag_ode == 'mass_surface':
    flag_control = True
    Field = subclass_ODE_mass_on_surface(flag_control)
    x0 = 0.4; y0 = 0.8; z0 = 0
    dx0 = 0.8; dy0 = 1; dz0 = 0
    X0 = [x0, y0, z0, dx0, dy0, dz0]
    u = np.array([0, 0, 0])
    X0 = Field.compute_points(np.array(X0).reshape(-1, 1)).flatten()  # the initial condition

elif flag_ode == 'duffling':
    flag_control = False
    Field = subclass_ODE_duffling_oscillator(flag_control, flag_mean_prior=False)
    #x0 = 0.4; y0 = 0.8
    #dx0 = 2 * np.pi + 1; dy0 = 1
    x0 = 1; y0 = 1
    dx0 = 2*np.pi+2; dy0 = 2
    X0 = [x0, y0, dx0, dy0, 0]
    u = None
    X0 = Field.compute_points(np.array(X0).reshape(-1, 1)).flatten()  # the initial condition
    X0 = X0[:-1]

params_phys = {'theta': Field.field_param['params'], 'u': u}
decorated_ODE = Field.decorator_changeout_for_solveivp(Field.func_ODE, Field.field_param['dim_out'],
                                                       flag_u=flag_control, flag_constrained=flag_constrained)

t_steps = np.linspace(0, 6, 100)
XXy_phys = Field.compute_trajectory(t_steps, np.copy(X0), decorated_ODE, params_phys)

dXX_phys = np.zeros((Field.field_param['dim_out'], XXy_phys.shape[1]))  # Acceleration at solution
for i in range(XXy_phys.shape[1]):
    dXX_phys[:, i] = decorated_ODE(t_steps[i], XXy_phys[:, i], params_phys)[Field.field_param['dim_out']:]

# plt.title('positions')
# fig = plt.figure(1)
# plt.plot(t_steps, XXy_phys[0, :], color='blue', linestyle='-', label=r'$q_1$')
# plt.plot(t_steps, XXy_phys[1, :], color='green', linestyle='-', label=r'$q_2$')
# plt.plot(t_steps, XXy_phys[1, :]-XXy_phys[0, :], linestyle='--', color='black', label=r'$q_2-q_1$')
# plt.xlabel(r't [s]')
# plt.ylabel(r'positions [m]')
# plt.xlim(np.min(t_steps), np.max(t_steps))
# plt.legend(loc='lower right')
# plt.subplots_adjust(left=0.15, bottom=0.2, right=0.97, top=0.95, wspace=0.3, hspace=0.1)
# plt.grid()

fig = plt.figure(2)
plt.plot(t_steps, dXX_phys[0, :], color='blue', linestyle='-', label=r'$\ddot{q}_1$')
plt.plot(t_steps, dXX_phys[1, :], color='green', linestyle='-', label=r'$\ddot{q}_2$')
if Field.field_param['dim_out'] > 2:
    plt.plot(t_steps, dXX_phys[2, :], color='lightcoral', linestyle='-', label=r'$\ddot{\theta}$')
    plt.ylabel(r'accelerations [m/s$^2$, rad/s$^2$]')
else:
    plt.ylabel(r'accelerations [m/s$^2$]')
plt.xlabel(r't [s]')
plt.xlim(np.min(t_steps), np.max(t_steps))
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.97, top=0.95, wspace=0.3, hspace=0.1)
plt.grid()

#fig = plt.figure(2)
#plt.plot(t_steps, XXy_phys[1, :]-XXy_phys[0, :], color='green', linestyle='--', label='y')
#plt.plot(t_steps, XXy_phys[2, :], color='blue', linestyle='--', label='z')
#plt.rcParams.update({'font.size': 12})  # increase the font size
#plt.xlabel('time')
#plt.ylabel('positions')
#plt.legend()
#plt.grid()

# 2D trajectory plot
if flag_ode is not 'duffling':
    fig = plt.figure(3)
    #plt.title('Trajectory with velocity')
    plt.xlabel(r"$q_1$ [m]")
    plt.ylabel(r"$q_2$ [m]")
    plt.plot(XXy_phys[0, :], XXy_phys[1, :], color='blue', label='position')
    plt.quiver(XXy_phys[0, :], XXy_phys[1, :], XXy_phys[3, :], XXy_phys[4, :], scale=3,width=0.005, color='green', label='velocity')
    #plt.legend(loc='lower right')
    #plt.xlim(0.1, 1.2)
    #plt.ylim(-0.05, 1.05)
    plt.axis('equal')
    plt.gca().set_aspect('equal')  # , adjustable='box'
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.95)
    plt.grid()

plt.show()

