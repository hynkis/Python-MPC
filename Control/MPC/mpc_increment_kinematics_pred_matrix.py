"""
Date : 2019.08.31
Author : Hyunki Seong

Python MPC
    - based on Kinematics model
    - using predictive linearized matrix
    - Incremental MPC
"""

import osqp
import numpy as np
import scipy as sp
import scipy.sparse as sparse

import matplotlib.pyplot as plt
import math
import time
import sys
sys.path.append("../../Vehicle_Dynamics/")
try:
    import vehicle_models
    from visualization_vehicle import plot_car
except:
    raise

def nearest_point(path_x, path_y, x, y, look_ind=0):
    min_d = np.inf
    min_ind = -1

    for i in reversed(range(len(path_x))):
        d = np.sqrt( (path_x[i]-x)**2 + (path_y[i]-y)**2 )
        if d < min_d:
            min_d = d
            min_ind = i
    
    min_ind = min_ind + look_ind

    return min_ind

def reference_search(path_x, path_y, pred_state, dt, N):
    """
    Find reference for MPC
        States  : [x; y; v; yaw]
        Actions : [steer; accel]
    """
    x_ref = np.zeros((4, N+1)) # nx : 4
    u_ref = np.zeros((2, N+1)) # nu : 2

    cumul_d = 0
    # ind = nearest_point(path_x, path_y, pred_state[0,0], pred_state[1,0])
    ind = nearest_point(path_x, path_y, pred_state[0,0], pred_state[1,0], look_ind=1) # look ahead 1 index.
    path_d = np.sqrt( (path_x[ind+1]-path_x[ind])**2 + (path_y[ind+1]-path_y[ind])**2 )

    # Reference points from x0 to xN
    for i in range(N+1):
        # Calculate Reference points
        cumul_d = cumul_d + abs(pred_state[2,i])*dt # vx == x[2]

        if cumul_d < path_d:
            x_ref[0, i] = path_x[ind]
            x_ref[1, i] = path_y[ind]
            x_ref[2, i] = 10.0
            x_ref[3, i] = np.deg2rad(0)

            u_ref[0, i] = 0.0 # steer operational point should be 0.
            u_ref[1, i] = 0.0

        else:
            # go forward until cumul_d < path_d
            while(cumul_d >= path_d):
                ind = ind + 1
                path_d = path_d + np.sqrt( (path_x[ind+1]-path_x[ind])**2 + (path_y[ind+1]-path_y[ind])**2 )

            x_ref[0, i] = path_x[ind]
            x_ref[1, i] = path_y[ind]
            x_ref[2, i] = 10.0
            x_ref[3, i] = np.deg2rad(0)

            u_ref[0, i] = 0.0 # steer operational point should be 0.
            u_ref[1, i] = 0.0

    return x_ref, u_ref

def reference_search_(path_x, path_y, path_yaw, pred_state, dt, N):
    """
    Find reference for MPC
        States  : [x; y; v; yaw]
        Actions : [steer; accel]
    """
    x_ref = np.zeros((4, N+1)) # nx : 4
    u_ref = np.zeros((2, N+1)) # nu : 2

    cumul_d = 0
    # ind = nearest_point(path_x, path_y, pred_state[0,0], pred_state[1,0])
    ind = nearest_point(path_x, path_y, pred_state[0,0], pred_state[1,0], look_ind=1) # look ahead 1 index.
    path_d = np.sqrt( (path_x[ind+1]-path_x[ind])**2 + (path_y[ind+1]-path_y[ind])**2 )

    # Reference points from x0 to xN
    for i in range(N+1):
        # Calculate Reference points
        cumul_d = cumul_d + abs(pred_state[2,i])*dt # vx == x[2]

        if cumul_d < path_d:
            x_ref[0, i] = path_x[ind]
            x_ref[1, i] = path_y[ind]
            x_ref[2, i] = 10.0
            x_ref[3, i] = path_yaw[ind]

            u_ref[0, i] = 0.0 # steer operational point should be 0.
            u_ref[1, i] = 0.0

        else:
            # go forward until cumul_d < path_d
            while(cumul_d >= path_d):
                ind = ind + 1
                path_d = path_d + np.sqrt( (path_x[ind+1]-path_x[ind])**2 + (path_y[ind+1]-path_y[ind])**2 )

            x_ref[0, i] = path_x[ind]
            x_ref[1, i] = path_y[ind]
            x_ref[2, i] = 10.0
            x_ref[3, i] = path_yaw[ind]

            u_ref[0, i] = 0.0 # steer operational point should be 0.
            u_ref[1, i] = 0.0

    return x_ref, u_ref

def check_goal(state, goal, goal_dist, stop_speed):
    # check goal
    dx = state[0] - goal[0]
    dy = state[1] - goal[1]
    d = math.sqrt(dx**2 + dy**2)

    if (d <= goal_dist):
        isgoal = True
    else:
        isgoal = False

    if (state[2] <= stop_speed):
        isstop = True
    else:
        isstop = False

    if isgoal and isstop:
        return True

    return False

def mpc_increment(Ad_list, Bd_list, gd_list, x_tilda_vec, Xr, pred_x_tilda, pred_del_u, Q, QN, R, N, xmin_tilda, xmax_tilda, del_umin, del_umax):
    """
    Incremental MPC
    x_tilda_vec : [nx+nu, nx+nu]
    Xr          : [nx, N+1]
    Q           : [nx, nx]
    R           : [nu, nu]
    """
    # ========== Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1)) ==========
    nx = Ad_list[0].shape[0]
    nu = Bd_list[0].shape[1]

    # Cast MPC problem to a QP:
    #   x = (x(0),x(1),...,x(N), u(0),...,u(N-1))

    # Objective function
    # C_tilda = [I, 0]
    # Q_tilda = C_tilda.T * Q * C_tilta : (nx+nu, nx) * (nx, nx) * (nx, nx+nu) => (nx+nu, nx+nu)
    C_tilda = sparse.hstack([sparse.eye(nx), np.zeros([nx, nu])]) # (nx, nx+nu)
    Q_tilda = C_tilda.transpose() * Q * C_tilda
    Q_tilda_N = C_tilda.transpose() * QN * C_tilda

    # - quadratic objective (P)
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q_tilda),       # Q x (N+1) on diagonal
                        Q_tilda_N,
                        sparse.kron(sparse.eye(N), R),             # R X (N) on diagonal
                        ]).tocsc()      

    # - linear objective (q)
    Q_C_tilda = Q * C_tilda
    QN_C_tilda = QN * C_tilda

    Q_C_tilda_trans = Q_C_tilda.transpose()
    QN_C_tilda_trans = QN_C_tilda.transpose()

    q = -Q_C_tilda_trans.dot(Xr[:,0])                                   # index 0
    for ii in range(N-1):
        q = np.hstack([q, -Q_C_tilda_trans.dot(Xr[:,ii+1])])             # index 1 ~ N-1
    q = np.hstack([q, -QN_C_tilda_trans.dot(Xr[:,N]), np.zeros(N*nu)])   # index N

    # Augmentation for Incremental Control
    Ad_sys = Ad_list[0]
    Bd_sys = Bd_list[0]
    Aug_A_sys = np.hstack([Ad_sys, Bd_sys])
    Aug_A_increment = sparse.hstack([sparse.csr_matrix((nu, nx)), sparse.eye(nu)])
    Ad_tilda = sparse.vstack([Aug_A_sys, Aug_A_increment])
    Bd_tilda = sparse.vstack([Bd_sys, sparse.eye(nu)])

    Ax_Ad = sparse.csc_matrix(Ad_tilda)
    Ax_diag = sparse.kron(sparse.eye(N+1),-sparse.eye(nx+nu))
    Bu_Bd = sparse.csc_matrix(Bd_tilda)

    for i in range(N-1):
        Ad_sys = Ad_list[i+1]
        Bd_sys = Bd_list[i+1]
        Aug_A_sys = np.hstack([Ad_sys, Bd_sys])
        Aug_A_increment = sparse.hstack([sparse.csr_matrix((nu, nx)), sparse.eye(nu)])
        Ad_tilda = sparse.vstack([Aug_A_sys, Aug_A_increment])
        Bd_tilda = sparse.vstack([Bd_sys, sparse.eye(nu)])

        Ax_Ad = sparse.block_diag([Ax_Ad, Ad_tilda])
        Bu_Bd = sparse.block_diag([Bu_Bd, Bd_tilda])

    Ax_Ad_top = sparse.kron(np.ones(N+1), np.zeros((nx+nu, nx+nu)))
    Ax_Ad_side = sparse.kron(np.ones((N,1)), np.zeros((nx+nu, nx+nu)))
    Ax = Ax_diag + sparse.vstack([Ax_Ad_top, sparse.hstack([Ax_Ad, Ax_Ad_side])])
    Bu_Bd_top = sparse.kron(np.ones(N), np.zeros((nx+nu, nu)))
    Bu = sparse.vstack([Bu_Bd_top, Bu_Bd])
    Aeq = sparse.hstack([Ax, Bu])

    # - Equality constraint (linear dynamics) : lower bound and upper bound
    leq = -x_tilda_vec # later ueq == leq
    for i in range(N):
        gd_tilda = np.vstack([gd_list[i], np.zeros((nu,1))]) # gd_tilda for augmented system
        gd_tilda = np.squeeze(gd_tilda, axis=1) # from (N,1) to (N,)
        leq = np.hstack([leq, -gd_tilda])
    # leq = np.hstack([-x_tilda_vec, np.zeros(N*nx)])
    ueq = leq

    # Original Code
    # ----- input and state constraints -----
    Aineq = sparse.eye((N+1)*(nx+nu) + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin_tilda), np.kron(np.ones(N), del_umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax_tilda), np.kron(np.ones(N), del_umax)])

    # ----- OSQP constraints -----
    A = sparse.vstack([Aeq, Aineq]).tocsc()
    lb = np.hstack([leq, lineq])
    ub = np.hstack([ueq, uineq])

    # ==========Create an OSQP object and Setup workspace ==========
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=True, polish=False, warm_start=False) # verbose: print output.

    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        print('OSQP did not solve the problem!')
        # raise ValueError('OSQP did not solve the problem!')
        plt.pause(0.5)
        # continue

    # Predictive States and Actions
    sol_state = res.x[:-N*nu]
    sol_action = res.x[-N*nu:]

    for ii in range((N+1)*(nx+nu)):
        if ii % (nx+nu) == 0:
            pred_x_tilda[0,ii//(nx+nu)] = sol_state[ii] # X
        elif ii % (nx+nu) == 1:
            pred_x_tilda[1,ii//(nx+nu)] = sol_state[ii] # Y
        elif ii % (nx+nu) == 2:
            pred_x_tilda[2,ii//(nx+nu)] = sol_state[ii] # Vx
        elif ii % (nx+nu) == 3:
            pred_x_tilda[3,ii//(nx+nu)] = sol_state[ii] # Yaw
        elif ii % (nx+nu) == 4:
            pred_x_tilda[4,ii//(nx+nu)] = sol_state[ii] # Steer
        else: # ii % (nx+nu) == 5:
            pred_x_tilda[5,ii//(nx+nu)] = sol_state[ii] # accel_track

    for jj in range((N)*nu):
        if jj % nu == 0:
            pred_del_u[0,jj//nu] = sol_action[jj]
        else: # jj % nu == 1
            pred_del_u[1,jj//nu] = sol_action[jj]
    pred_del_u[:,-1] = pred_del_u[:,-2] # append last control

    return pred_x_tilda, pred_del_u

def main():
    # ===== Vehicle parameters =====
    l_f = 1.25
    l_r = 1.40

    dt = 0.02

    m=1300
    width = 1.78
    length = 4.25
    turning_circle=10.4
    C_d = 0.34
    A_f = 2.0
    C_roll = 0.015

    vehicle = vehicle_models.Vehicle_Kinematics(l_f=l_f, l_r=l_r, dt = dt) # Vehicle Kinematic Model

    # ========== MPC parameters ==========
    N = 40 # Prediction horizon

    # ========== Constraints ==========
    del_umin = np.array([-np.deg2rad(0.5), -0.5]) # del_u / tic (not del_u / sec) => del_u/sec = del_u/tic * tic/sec = del_u/tic * 20(Hz)
    del_umax = np.array([ np.deg2rad(0.5),  0.5])
    xmin_tilda = np.array([-np.inf,-np.inf, -100., -2*np.pi, -np.deg2rad(15), -3.])     # (x_min, u_min)
    xmax_tilda = np.array([ np.inf, np.inf,  100.,  2*np.pi,  np.deg2rad(15),  1.])     # (x_max, u_max)

    # ========== Objective function ==========
    # MPC weight matrix
    Q = sparse.diags([50.0, 50.0, 10.0, 50.0])         # weight matrix for state
    # QN = Q
    QN = sparse.diags([1000.0, 1000.0, 100.0, 1000.0])   # weight matrix for terminal state
    R = sparse.diags([100, 100])                      # weight matrix for control input
    # R_before = 10*sparse.eye(nu)                    # weight matrix for control input

    # ========== Initialization ==========
    # Path
    path_x = np.linspace(-10, 100, int(100/0.5))
    # path_y = np.ones(int(int(100/0.5)))*5
    path_y = np.linspace(-10, 100, int(100/0.5)) * 0.0 + 0.0

    # Initial state
    # States  : [x; y; v; yaw]
    # Actions : [steer; accel]
    x0 = np.array([[0.0],
                [0.0],
                [20.0],
                [np.deg2rad(0)]])    #  [X; Y; V; Yaw]
    u0 = np.array([[0*math.pi/180],
                [0.01]])               #  [steer; accel]

    del_u0 = np.array([[0*math.pi/180],
                       [0.0]])            # [del steer; del accel]

    nx = x0.shape[0]
    nu = u0.shape[0]

    # for Incremental MPC
    x0_tilda = np.vstack([x0, u0])
    
    x_tilda = np.copy(x0_tilda)
    x_tilda_vec = np.squeeze(x_tilda, axis=1)

    # ========== Initialize Predictive States and Controls ==========
    # u_noise = np.zeros((nu, 1))
    # mu_steer = 0.0
    # sigma_steer = np.deg2rad(1)
    # mu_accel = 0.0
    # sigma_accel = 0.1

    x_tilda_k = np.copy(x0_tilda)
    del_u_k = np.copy(del_u0)

    pred_del_u = np.zeros((nu, N+1))
    for i in range(N):
        pred_del_u[:,i] = del_u_k.T
    pred_del_u[:,-1] = pred_del_u[:,-2]

    pred_x_tilda = np.zeros((nx+nu, N+1))
    pred_x_tilda[:,0] = x_tilda_k.T
    for i in range(0, N):
        x_k = x_tilda_k[:-nu] # decompose to xk, uk
        u_k = x_tilda_k[-nu:]
        Ad, Bd, gd = vehicle.get_kinematics_model(x_k, u_k)
        x_k1 = np.matmul(Ad, x_k) + np.matmul(Bd, u_k) + gd # calc next state
        u_k1 = u_k + np.expand_dims(pred_del_u[:,i], axis=1) # calc next action
        x_tilda_k1 = np.vstack([x_k1, u_k1]) # compose x tilda
        pred_x_tilda[:,i+1] = x_tilda_k1.T
        x_tilda_k = x_tilda_k1


    # ========== Simulation Setup ==========
    sim_time = 1000
    plt_tic = np.linspace(0, sim_time, sim_time)
    plt_states = np.zeros((nx, sim_time))
    plt_actions = np.zeros((nu, sim_time))
    # plt_pred_final_states = np.zeros((nx, sim_time))

    for i in range(sim_time):
        # Scenario : Double Lane Change
        if i > 0.05*sim_time:
            # Path
            path_x = np.linspace(-10, 100, int(100/0.5))
            path_y = np.linspace(-10, 100, int(100/0.5)) * 0.0 + 4
            # path_x = np.ones(int(int(100/0.5)))*5
            # path_y = np.ones(int(int(100/0.5)))*5

        if i > 0.15*sim_time:
            # Path
            path_x = np.linspace(-10, 100, int(100/0.5))
            # path_y = np.ones(int(int(100/0.5)))*5
            path_y = np.linspace(-10, 100, int(100/0.5)) * 0.0 + 0

        tic = time.time()
        print("===================== sim_time :", i, "=====================")
        # Discrete time model of the vehicle lateral dynamics

        # Reference states
        Xr, _ = reference_search(path_x, path_y, pred_x_tilda[:-nu,:], dt, N)
        plt.plot(pred_x_tilda[0,:], pred_x_tilda[1,:], "y")

        # Discrete time model of the vehicle lateral dynamics
        Ad_list, Bd_list, gd_list = [], [], []
        for ii in range(N):
            Ad, Bd, gd = vehicle.get_kinematics_model(pred_x_tilda[:-nu,ii], pred_x_tilda[-nu:,ii])
            Ad_list.append(Ad)
            Bd_list.append(Bd)
            gd_list.append(gd)

        # Solve MPC
        pred_x_tilda, pred_del_u = mpc_increment(Ad_list, Bd_list, gd_list, x_tilda_vec, Xr, pred_x_tilda, pred_del_u, Q, QN, R, N, xmin_tilda, xmax_tilda, del_umin, del_umax)
        toc = time.time()

        # Plot
        print("Current   x :", x_tilda[0], "y :", x_tilda[1], "v :", x_tilda[2], "yaw :", x_tilda[3])
        print("------------------------------------------------------------")
        print("Reference x :", Xr[0,0], "y :", Xr[1,0], "v :", Xr[2,0], "yaw :", Xr[3,0])
        print("------------------------------------------------------------")
        print("steer :", x_tilda[4], "accel :", x_tilda[5])

        print("Process time :", toc - tic)

        # Save current state
        plt_states[:,i] = x_tilda[:-nu].T
        plt_actions[:,i] = x_tilda[-nu:].T
        # plt_pred_final_states[:,i] = pred_x_tilda[:-nu, -1].T

        # Plot current state
        plt.cla()
        plt.plot(path_x, path_y, "y", label="Path")
        plt.plot(Xr[0,:], Xr[1,:], "g", label="Local Reference")
        plt.plot(plt_states[0,:i+1], plt_states[1,:i+1], "-b", label="Drived") # plot from 0 to i
        plt.grid(True)
        plt.axis("equal")
        plt.plot(x_tilda[0], x_tilda[1], "*g", label="Initial")
        plot_car(x_tilda[0], x_tilda[1], x_tilda[3], steer=x_tilda[4]) # plotting w.r.t. rear axle.
        plt.plot(pred_x_tilda[0,:], pred_x_tilda[1,:], "r", label="Predictive States")
        # plt.plot(plt_pred_final_states[0,:], plt_pred_final_states[1,:], "*r", label="Predictive Final States")
        
        plt.pause(0.0001)
        
        # Update States
        u_past = x_tilda[-nu:]
        u = u_past + np.expand_dims(pred_del_u[:,0], axis=1)
        x_next = np.matmul(Ad_list[0], x_tilda[:-nu]) + np.matmul(Bd_list[0], u) + gd_list[0]

        x = x_next
        x_tilda = np.vstack([x, u])
        
        x_tilda_vec = np.squeeze(x_tilda, axis=1)

        # Update Predictive States and Actions
        temp_pred_x_tilda = pred_x_tilda
        temp_pred_del_u = pred_del_u
        # index 0
        pred_x_tilda[:,0] = x_tilda.T                   
        pred_del_u[:,0] = temp_pred_del_u[:,1] # before u.T
        # index 1 ~ N-2
        for ii in range(0, N-2):
            pred_x_tilda[:,ii+1] = temp_pred_x_tilda[:,ii+2]
            pred_del_u[:,ii+1] = temp_pred_del_u[:,ii+2]
        # index N-1
        pred_x_tilda[:,-2] = temp_pred_x_tilda[:,N]
        pred_del_u[:,-2] = temp_pred_del_u[:,N-1]

        # index N
        # append last state using last A, B matrix and last pred state
        # append last control with last pred control
        last_state = np.expand_dims(temp_pred_x_tilda[:-nu,N], axis=1) # from (N,) to (N,1)
        last_control = np.expand_dims(temp_pred_x_tilda[-nu:,N-1], axis=1)
        pred_x_tilda[:-nu,-1] = np.transpose(vehicle.update_kinematics_model(last_state, last_control))
        pred_x_tilda[-nu:,-1] = pred_x_tilda[-nu:,-2]
        pred_del_u[:,-1] = pred_del_u[:,-2]

        # if check_goal(x, xr, goal_dist=1, stop_speed=5):
        #     print("Goal")
        #     break
    
if __name__ == "__main__":
    main()

