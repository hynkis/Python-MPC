import osqp
import numpy as np
import scipy as sp
import scipy.sparse as sparse

import matplotlib.pyplot as plt

"""
"""

# Simulate in closed loop
nsim = 1500
# Prediction horizon
N = 100

plt_tic = []
plt_u = []
plt_del_u = []
plt_x_1 = []
plt_x_2 = []
plt_x_3 = []
plt_x_4 = []
plt_s = []  # for slack

# Initial and reference states
#x0_sys = np.array([0., 0., 0., 3])
x0 = np.array([0., 0., 5*np.pi/180, 3., 0.]) # state and past control input
xr = np.array([0., 0., 0., 0.]) # (nx_sys, 1)

# ============== initialize =======================
# Discrete time model of the vehicle lateral dynamics
Ad_sys = sparse.csc_matrix([
        [0.960,	-0.019,	0.,	0.],
        [0.00469,	0.961,	0.,	0.],
        [0.,	0.0196,	1.,	0.],
        [0.163,	0.,	0.166,	1.]
])

Bd_sys = sparse.csc_matrix([
        [0.020575],
        [0.115],
        [0.001157],
        [0.00182]])

[nx_sys, nu_sys] = Bd_sys.shape

# Augmentation for Incremental Control
Aug_A_sys = sparse.hstack([Ad_sys, Bd_sys])
Aug_A_increment = sparse.hstack([sparse.csr_matrix((nu_sys, nx_sys)), sparse.eye(nu_sys)])
Ad_tilda = sparse.vstack([Aug_A_sys, Aug_A_increment])

Bd_tilda = sparse.vstack([Bd_sys, sparse.eye(nu_sys)])

[nx, nu] = Bd_tilda.shape # (nx_sys+1, nu_sys)

# Constraints
del_umin = np.array([-0.5*np.pi/180.]) # del_u / tic (not del_u / sec) => del_u/sec = del_u/tic * tic/sec = del_u/tic * 20(Hz)
del_umax = np.array([0.5*np.pi/180.])
xmin_tilda = np.array([-np.pi, -0.5*np.pi, -15*np.pi/180, -10., -30*np.pi/180]) # (x_min, u_min)
xmax_tilda = np.array([ np.pi, 0.5*np.pi, 15*np.pi/180, 10., 30*np.pi/180])     # (x_max, u_max)

# Objective function
# C_tilda = [I, 0]
# Q_tilda = C_tilda.T * Q * C_tilta : (nx+1, nx) * (nx, nx) * (nx, nx+1) => (nx+1, nx+1)
C_tilda = sparse.hstack([sparse.eye(nx_sys), np.zeros([nx_sys, nu])])
Q = sparse.diags([5., 5., 10., 10.]) # weight matrix for state
Q_tilda = C_tilda.transpose() * Q * C_tilda
Q_C_tilda = Q * C_tilda
QN = Q_tilda

R = 10*sparse.eye(nu)                  # weight matrix for control input

W_tilda = sparse.diags([10., 10., 10., 10., 0.])   # weight matrix for slack variable sx. 0 for u_k-1
WN = W_tilda

# Cast MPC problem to a QP:
#   x = (x(0),x(1),...,x(N), u(0),...,u(N-1), sx(0),sx(1),...,sx(N))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q_tilda),       # Q x (N+1) on diagonal
                       QN,
                       sparse.kron(sparse.eye(N), R),             # R X (N) on diagonal
                       sparse.kron(sparse.eye(N), W_tilda),       # W x (N+1) on diagonal
                       WN,
                      ]).tocsc()      
# - linear objective
Q_C_tilda_trans = Q_C_tilda.transpose()
q = np.hstack([np.kron(np.ones(N), -Q_C_tilda_trans.dot(xr)), -Q_C_tilda_trans.dot(xr),  # (nx+1, nx) * (nx, 1) '-xr.T*Q' x (N+1) horizontally
               np.zeros(N*nu),                                # '[0,..,0]' x (N)   horizontally
               np.kron(np.ones(N+1), np.zeros(nx))            # '[0,..,0]' x (N+1) horizontally
               ])

# - Equality constraint (linear dynamics) : lower bound and upper bound
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad_tilda)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd_tilda)

S = sparse.kron(sparse.eye(N+1), 0*sparse.eye(nx))

Aeq = sparse.hstack([Ax, Bu, S])        # A-equality matrix
leq = np.hstack([-x0, np.zeros(N*nx)])  # equality constraints
ueq = leq                               # equality constraints (same with lower bound)

# - Inequality constraints (input and state constraints) : only upper bound
#weight_slack = sparse.diags([1., 1., 1., 1.]) # larger value, more softer
weight_slack_tilda = sparse.diags([1., 1., 1., 1., 0.]) # larger value, more softer

Sineq = sparse.vstack([sparse.kron(sparse.eye(N+1), weight_slack_tilda), sparse.csr_matrix((N*nu,(N+1)*nx))])
Aineq = sparse.hstack([sparse.eye((N+1)*nx + N*nu), Sineq])

lineq = np.hstack([np.kron(np.ones(N+1), xmin_tilda), np.kron(np.ones(N), del_umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax_tilda), np.kron(np.ones(N), del_umax)])

# - OSQP constraints
A = sparse.vstack([Aeq, Aineq]).tocsc()
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

for i in range(nsim):

    # Dynamic reference states
    if i <= 100:
      xr = np.array([0., 0., 0., 0.])
    else:
      xr = np.array([0., 0., 0., 0.])

    # Discrete time model of the vehicle lateral dynamics
    Ad_sys = sparse.csc_matrix([
        [0.960,	-0.019,	0.,	0.],
        [0.00469,	0.961,	0.,	0.],
        [0.,	0.0196,	1.,	0.],
        [0.163,	0.,	0.166,	1.]
    ])

    Bd_sys = sparse.csc_matrix([
            [0.020575],
            [0.115],
            [0.001157],
            [0.00182]])

    [nx_sys, nu_sys] = Bd_sys.shape

    # Augmentation for Incremental Control
    Aug_A_sys = sparse.hstack([Ad_sys, Bd_sys])
    Aug_A_increment = sparse.hstack([sparse.csr_matrix((nu_sys, nx_sys)), sparse.eye(nu_sys)])
    Ad_tilda = sparse.vstack([Aug_A_sys, Aug_A_increment])

    Bd_tilda = sparse.vstack([Bd_sys, sparse.eye(nu_sys)])

    [nx, nu] = Bd_tilda.shape # (nx_sys+1, nu_sys)

    # Constraints

    if i <= 400:
      del_umin = np.array([-0.5*np.pi/180.])
      del_umax = np.array([0.5*np.pi/180.])
      xmin_tilda = np.array([-np.pi, -0.5*np.pi, -15*np.pi/180, -10., -30*np.pi/180]) # (x_min, u_min)
      xmax_tilda = np.array([ np.pi, 0.5*np.pi, 15*np.pi/180, 10., 30*np.pi/180])     # (x_max, u_max)
    elif i <= 900:
      del_umin = np.array([-0.5*np.pi/180.])
      del_umax = np.array([0.5*np.pi/180.])
      xmin_tilda = np.array([-np.pi, -0.5*np.pi, -15*np.pi/180, 2., -30*np.pi/180]) # (x_min, u_min)
      xmax_tilda = np.array([ np.pi, 0.5*np.pi, 15*np.pi/180, 10., 30*np.pi/180])     # (x_max, u_max)
    else:
      del_umin = np.array([-0.5*np.pi/180.])
      del_umax = np.array([0.5*np.pi/180.])
      xmin_tilda = np.array([-np.pi, -0.5*np.pi, -15*np.pi/180, -10., -30*np.pi/180]) # (x_min, u_min)
      xmax_tilda = np.array([ np.pi, 0.5*np.pi, 15*np.pi/180, 10., 30*np.pi/180])     # (x_max, u_max)

    # Objective function
    # C_tilda = [I, 0]
    # Q_tilda = C_tilda.T * Q * C_tilta : (nx+1, nx) * (nx, nx) * (nx, nx+1) => (nx+1, nx+1)
    C_tilda = sparse.hstack([sparse.eye(nx_sys), np.zeros([nx_sys, nu])])
    Q = sparse.diags([5., 5., 10., 10.]) # weight matrix for state
    Q_tilda = C_tilda.transpose() * Q * C_tilda
    Q_C_tilda = Q * C_tilda
    QN = Q_tilda

    R = 10*sparse.eye(nu)                  # weight matrix for control input

    W_tilda = sparse.diags([10., 10., 10., 10., 0.])   # weight matrix for slack variable sx. 0 for u_k-1
    WN = W_tilda

    

    # Cast MPC problem to a QP:
    #   x = (x(0),x(1),...,x(N), u(0),...,u(N-1), sx(0),sx(1),...,sx(N))
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q_tilda),       # Q x (N+1) on diagonal
                          QN,
                          sparse.kron(sparse.eye(N), R),             # R X (N) on diagonal
                          sparse.kron(sparse.eye(N), W_tilda),       # W x (N+1) on diagonal
                          WN,
                          ]).tocsc()      
    # - linear objective
    Q_C_tilda_trans = Q_C_tilda.transpose()
    q_new = np.hstack([np.kron(np.ones(N), -Q_C_tilda_trans.dot(xr)), -Q_C_tilda_trans.dot(xr),  # (nx+1, nx) * (nx, 1) '-xr.T*Q' x (N+1) horizontally
                       np.zeros(N*nu),                                # '[0,..,0]' x (N)   horizontally
                       np.kron(np.ones(N+1), np.zeros(nx))            # '[0,..,0]' x (N+1) horizontally
                      ])

    # - Equality constraint (linear dynamics) : lower bound and upper bound
    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad_tilda)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd_tilda)

    S = sparse.kron(sparse.eye(N+1), 0*sparse.eye(nx))

    Aeq = sparse.hstack([Ax, Bu, S])        # A-equality matrix
    leq = np.hstack([-x0, np.zeros(N*nx)])  # equality constraints
    ueq = leq                               # equality constraints (same with lower bound)

    # - Inequality constraints (input and state constraints) : only upper bound
    #weight_slack = sparse.diags([1., 1., 1., 1.]) # larger value, more softer
    weight_slack_tilda = sparse.diags([1., 1., 1., 1., 0.]) # larger value, more softer

    Sineq = sparse.vstack([sparse.kron(sparse.eye(N+1), weight_slack_tilda), sparse.csr_matrix((N*nu,(N+1)*nx))])
    Aineq = sparse.hstack([sparse.eye((N+1)*nx + N*nu), Sineq])

    lineq = np.hstack([np.kron(np.ones(N+1), xmin_tilda), np.kron(np.ones(N), del_umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax_tilda), np.kron(np.ones(N), del_umax)])

    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq]).tocsc()
    l_new = np.hstack([leq, lineq])
    u_new = np.hstack([ueq, uineq])

    # Create an OSQP object
    # prob = osqp.OSQP()


    # Update workspace
    #prob.update(Ax=A, Ax_idx=np.array([0,0]), len(A) # update constraint matrix. 좀더 참고 필요 : https://osqp.org/docs/interfaces/python.html#python-interface
    prob.update(q=q_new, l=l_new, u=u_new) # update constraint limits

    print("nsim :", i)
    # Initial values
    plt_x_1.append(x0[0])
    plt_x_2.append(x0[1])
    plt_x_3.append(x0[2])
    plt_x_4.append(x0[3])
    plt_tic.append(i)

    # Solve
    res = prob.solve()
    #res1 = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    del_ctrl = res.x[(N+1)*nx : (N+1)*nx + 1] # indexing u0
    x0 = Ad_tilda.dot(x0) + Bd_tilda.dot(del_ctrl) # x0 : (x1, x2, ... xn, uk-1)

    slack = res.x[-(N+1)*nx:][3] # slack for error y


    plt_u.append(x0[-1])
    plt_del_u.append(del_ctrl)
    plt_s.append(slack)

    # Update initial state
    l_new[:nx] = -x0
    u_new[:nx] = -x0
    prob.update(l=l_new, u=u_new) # update initial state, control

# Plot result

fig = plt.figure()

ax1 = fig.add_subplot(2, 4, 1)
ax2 = fig.add_subplot(2, 4, 2) 
ax3 = fig.add_subplot(2, 4, 3)
ax4 = fig.add_subplot(2, 4, 4)
ax5 = fig.add_subplot(2, 4, 5) 
ax6 = fig.add_subplot(2, 4, 6)
ax7 = fig.add_subplot(2, 4, 7)

ax1.set_title("x_1 side_slip")
ax2.set_title("x_2 yaw_rate")
ax3.set_title("x_3 error yaw")
ax4.set_title("x_4 error y")
ax5.set_title("x_5 control")
ax6.set_title("del_u")
ax7.set_title("slack")

ax1.plot(plt_tic, plt_x_1)
ax2.plot(plt_tic, plt_x_2)
ax3.plot(plt_tic, plt_x_3)
ax4.plot(plt_tic, plt_x_4)

ax5.plot(plt_tic, plt_u)
ax6.plot(plt_tic, plt_del_u)
ax7.plot(plt_tic, plt_s)

plt.show()

