"""
Date : 2019.08.22
Author : Hyunki Seong

Python MPC
    - based on Kinematics model
    - using predictive linearized matrix
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

def mpc(Ad_mat, Bd_mat, gd_mat, x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax):
    # ========== Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1)) ==========
    nx = Ad_mat.shape[0]
    nu = Bd_mat.shape[1]

    Ad = sparse.csc_matrix(Ad_mat)
    Bd = sparse.csc_matrix(Bd_mat)
    gd = np.squeeze(gd_mat, axis=1) # from (N,1) to (N,)

    # ----- quadratic objective -----
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)]).tocsc()

    # ----- linear objective -----
    # xr_vec = np.squeeze(xr, axis=1)
    # q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
    #               np.zeros(N*nu)])
    # q = np.hstack([np.kron(np.ones(N), -Q.dot(xr_vec)), -QN.dot(xr_vec),
    #               np.zeros(N*nu)])

    q = -Q.dot(Xr[:,0])                                    # index 0
    for ii in range(N-1):
        q = np.hstack([q, -Q.dot(Xr[:,ii+1])])             # index 1 ~ N-1
    q = np.hstack([q, -QN.dot(Xr[:,-1]), np.zeros(N*nu)])  # index N

    # ----- linear dynamics -----
    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])

    # leq = np.hstack([-x0, np.zeros(N*nx)])
    # leq = np.hstack([-x0, np.kron(np.ones(N), -gd)])

    leq = np.hstack([-x_vec, np.kron(np.ones(N), -gd)])
    ueq = leq

    # ----- input and state constraints -----
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    # ----- OSQP constraints -----
    A = sparse.vstack([Aeq, Aineq]).tocsc()
    lb = np.hstack([leq, lineq])
    ub = np.hstack([ueq, uineq])

    # ==========Create an OSQP object and Setup workspace ==========
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=False, warm_start=True) # verbose: print output.

    # Solve
    res = prob.solve()

    return res

def mpc_(Ad_mat, Bd_mat, gd_mat, x_vec, Xr, Q, QN, R, N, lb_x, ub_x, lb_y, ub_y, umin, umax):
    # ========== Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1)) ==========
    nx = Ad_mat.shape[0]
    nu = Bd_mat.shape[1]

    Ad = sparse.csc_matrix(Ad_mat)
    Bd = sparse.csc_matrix(Bd_mat)
    gd = np.squeeze(gd_mat, axis=1) # from (N,1) to (N,)

    # ----- quadratic objective -----
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)]).tocsc()

    # ----- linear objective -----
    # xr_vec = np.squeeze(xr, axis=1)
    # q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
    #               np.zeros(N*nu)])
    # q = np.hstack([np.kron(np.ones(N), -Q.dot(xr_vec)), -QN.dot(xr_vec),
    #               np.zeros(N*nu)])

    q = -Q.dot(Xr[:,0])                                    # index 0
    for ii in range(N-1):
        q = np.hstack([q, -Q.dot(Xr[:,ii+1])])             # index 1 ~ N-1
    q = np.hstack([q, -QN.dot(Xr[:,-1]), np.zeros(N*nu)])  # index N

    # ----- linear dynamics -----
    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])

    # leq = np.hstack([-x0, np.zeros(N*nx)])
    # leq = np.hstack([-x0, np.kron(np.ones(N), -gd)])

    leq = np.hstack([-x_vec, np.kron(np.ones(N), -gd)])
    ueq = leq

    # ----- input and state constraints -----
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = []
    uineq = []
    # lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    # uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

    for i in range(len(lb_x)):
        xmin = [lb_x[i], lb_y[i], -10, -np.pi]
        xmax = [ub_x[i], ub_y[i],  10,  np.pi]
        lineq = np.hstack([lineq, xmin])
        uineq = np.hstack([uineq, xmax])
    lineq = np.hstack([lineq, np.kron(np.ones(N), umin)])
    uineq = np.hstack([uineq, np.kron(np.ones(N), umax)])

    # ----- OSQP constraints -----
    A = sparse.vstack([Aeq, Aineq]).tocsc()
    lb = np.hstack([leq, lineq])
    ub = np.hstack([ueq, uineq])

    # ==========Create an OSQP object and Setup workspace ==========
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=False, warm_start=True) # verbose: print output.

    # Solve
    res = prob.solve()

    return res

def mpc__(Ad_list, Bd_list, gd_list, x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax):
    """
    Initialize Nonlinear Dynamics with Shooting Method

    """
    # ========== Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1)) ==========
    nx = Ad_list[0].shape[0]
    nu = Bd_list[0].shape[1]

    # ----- quadratic objective -----
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)]).tocsc()

    # ----- linear objective -----
    # xr_vec = np.squeeze(xr, axis=1)
    # q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
    #               np.zeros(N*nu)])
    # q = np.hstack([np.kron(np.ones(N), -Q.dot(xr_vec)), -QN.dot(xr_vec),
    #               np.zeros(N*nu)])

    q = -Q.dot(Xr[:,0])                                    # index 0
    for ii in range(N-1):
        q = np.hstack([q, -Q.dot(Xr[:,ii+1])])             # index 1 ~ N-1
    q = np.hstack([q, -QN.dot(Xr[:,-1]), np.zeros(N*nu)])  # index N

    # ----- linear dynamics -----
    Ax_Ad = sparse.csc_matrix(Ad_list[0])
    Ax_diag = sparse.kron(sparse.eye(N+1),-sparse.eye(nx))
    Bu_Bd = sparse.csc_matrix(Bd_list[0])
    
    for i in range(N-1):
        Ad = sparse.csc_matrix(Ad_list[i+1])
        Bd = sparse.csc_matrix(Bd_list[i+1])
        Ax_Ad = sparse.block_diag([Ax_Ad, Ad])
        Bu_Bd = sparse.block_diag([Bu_Bd, Bd])

    Ax_Ad_top = sparse.kron(np.ones(N+1), np.zeros((nx,nx)))
    Ax_Ad_side = sparse.kron(np.ones((N,1)), np.zeros((nx,nx)))
    Ax = Ax_diag + sparse.vstack([Ax_Ad_top, sparse.hstack([Ax_Ad, Ax_Ad_side])])
    Bu_Bd_top = sparse.kron(np.ones(N), np.zeros((nx,nu)))
    Bu = sparse.vstack([Bu_Bd_top, Bu_Bd])
    Aeq = sparse.hstack([Ax, Bu])

    leq = -x_vec # later ueq == leq
    for i in range(N):
        gd = np.squeeze(gd_list[i], axis=1) # from (N,1) to (N,)
        leq = np.hstack([leq, -gd])
    ueq = leq

    # Original Code
    # Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad_list[0])
    # Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd_list[0])
    # Aeq = sparse.hstack([Ax, Bu])
    # gd = np.squeeze(gd_list[0], axis=1) # from (N,1) to (N,)
    # leq = np.hstack([-x_vec, np.kron(np.ones(N), -gd)])
    # ueq = leq

    # ----- input and state constraints -----
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    # lineq = []
    # uineq = []

    # for i in range(len(lb_x)):
    #     xmin = [lb_x[i], lb_y[i], -10, -np.pi]
    #     xmax = [ub_x[i], ub_y[i],  10,  np.pi]
    #     lineq = np.hstack([lineq, xmin])
    #     uineq = np.hstack([uineq, xmax])
    # lineq = np.hstack([lineq, np.kron(np.ones(N), umin)])
    # uineq = np.hstack([uineq, np.kron(np.ones(N), umax)])

    # ----- OSQP constraints -----
    A = sparse.vstack([Aeq, Aineq]).tocsc()
    lb = np.hstack([leq, lineq])
    ub = np.hstack([ueq, uineq])

    # ==========Create an OSQP object and Setup workspace ==========
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=False, warm_start=True) # verbose: print output.

    # Solve
    res = prob.solve()

    return res


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
    N = 100 # Prediction horizon

    # ========== Initialization ==========
    # Path
    path_x = np.linspace(-10, 100, int(100/0.5))
    # path_y = np.ones(int(100/0.5))*5
    path_y = np.linspace(-10, 100, int(100/0.5)) * 0.0 - 5.0

    # Initial state
    # States  : [x; y; v; yaw]
    # Actions : [steer; accel]
    x = np.array([[0.0],
                [0.0],
                [20.0],
                [np.deg2rad(0)]])    #  [X; Y; V; Yaw]
    u = np.array([[0*math.pi/180],
                [0.01]])               #  [steer; accel]

    x0 = x
    u0 = u
    x_vec = np.squeeze(x, axis=1) # (N,) shape for QP solver, NOT (N,1).

    nx = x.shape[0]
    nu = u.shape[0]

    # ========== Initialize Predictive States and Controls ==========
    u_noise = np.zeros((nu, 1))
    mu_steer = 0.0
    sigma_steer = np.deg2rad(1)
    mu_accel = 0.0
    sigma_accel = 0.1

    pred_u = np.zeros((nu, N+1))
    for i in range(N):
        # u_noise[0] = np.random.normal(mu_steer, sigma_steer, 1)
        # u_noise[1] = np.random.normal(mu_accel, sigma_accel, 1)
        # pred_u[:,i] = np.transpose(u0 + u_noise)
        pred_u[:,i] = np.transpose(u0)
    pred_u[:,-1] = pred_u[:,-2] # append last pred_u for N+1

    pred_x = np.zeros((nx, N+1))
    pred_x[:,0] = x0.T
    x_k = np.copy(x0)  # if x_k = x0, x0 would be changed by update kinematic model
    for i in range(0, N):
        x_k1 = vehicle.update_kinematics_model(x_k, pred_u[:,i])
        pred_x[:,i+1] = x_k1.T
        x_k = x_k1
        # pred_x[:,i+1] = x0.T



    # ========== Constraints ==========
    umin = np.array([-np.deg2rad(15), -3.]) # u : [steer, accel]
    umax = np.array([ np.deg2rad(15),  1.])
    xmin = np.array([-np.inf,-np.inf, -100., -2*np.pi]) #  [X; Y; vel_x; Yaw]
    xmax = np.array([ np.inf, np.inf,  100.,  2*np.pi])

    # ========== Objective function ==========
    # MPC weight matrix
    Q = sparse.diags([10.0, 10.0, 100.0, 10.0])         # weight matrix for state
    # QN = Q
    QN = sparse.diags([100.0, 100.0, 1000.0, 100.0])   # weight matrix for terminal state
    R = sparse.diags([1000, 100])                      # weight matrix for control input
    # R_before = 10*sparse.eye(nu)                    # weight matrix for control input

    # ========== Simulation Setup ==========
    sim_time = 1000
    plt_tic = np.linspace(0, sim_time, sim_time)
    plt_states = np.zeros((nx, sim_time))
    plt_actions = np.zeros((nu, sim_time))

    for i in range(sim_time):
        tic = time.time()
        print("===================== sim_time :", i, "=====================")
        
        # Discrete time model of the vehicle lateral dynamics

        # Reference states
        Xr, _ = reference_search(path_x, path_y, pred_x, dt, N)
        plt.plot(pred_x[0,:], pred_x[1,:], "y")

        print("x_vec :", x_vec)
        print("pred_x[:,0] :", pred_x[:,0])

        # Discrete time model of the vehicle lateral dynamics
        Ad_list, Bd_list, gd_list = [], [], []
        for ii in range(N):
            Ad, Bd, gd = vehicle.get_kinematics_model(pred_x[:,ii], pred_u[:,ii])
            Ad_list.append(Ad)
            Bd_list.append(Bd)
            gd_list.append(gd)

        # ========== Constraints ==========
        umin = np.array([-np.deg2rad(15), -3.]) # u : [steer, accel]
        umax = np.array([ np.deg2rad(15),  1.])
        xmin = np.array([-np.inf,-np.inf, -100., -2*np.pi]) #  [X; Y; vel_x; Yaw]
        xmax = np.array([ np.inf, np.inf,  100.,  2*np.pi])

        # Solve MPC
        # res = mpc(Ad_mat, Bd_mat, gd_mat, x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax)
        # if i <= -1:
        #     res = mpc(Ad_list[0], Bd_list[0], gd_list[0], x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax)
        # else:
        #     res = mpc__(Ad_list, Bd_list, gd_list, x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax)
        res = mpc__(Ad_list, Bd_list, gd_list, x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax)

        # Check solver status
        if res.info.status != 'solved':
            print('OSQP did not solve the problem!')
            # raise ValueError('OSQP did not solve the problem!')
            plt.pause(5.0)
            continue

        # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]
        toc = time.time()
        print("ctrl :", ctrl)

        # Predictive States and Actions
        sol_state = res.x[:-N*nu]
        sol_action = res.x[-N*nu:]
        
        for ii in range((N+1)*nx):
            if ii % nx == 0:
                pred_x[0,ii//nx] = sol_state[ii]
            elif ii % nx == 1:
                pred_x[1,ii//nx] = sol_state[ii]
            elif ii % nx == 2:
                pred_x[2,ii//nx] = sol_state[ii]
            else: # ii % 4 == 3:
                pred_x[3,ii//nx] = sol_state[ii]

        for jj in range((N)*nu):
            if jj % nu == 0:
                pred_u[0,jj//nu] = sol_action[jj]
            else: # jj % nu == 1
                pred_u[1,jj//nu] = sol_action[jj]
        pred_u[:,-1] = pred_u[:,-2] # append last control

        # Plot
        print("Current   x :", x[0], "y :", x[1], "v :", x[2], "yaw :", x[3])
        print("------------------------------------------------------------")
        # print("Reference x :", xr[0], "y :", xr[1], "v :", xr[2], "yaw :", xr[3])
        print("Reference x :", Xr[0,0], "y :", Xr[1,0], "v :", Xr[2,0], "yaw :", Xr[3,0])
        print("------------------------------------------------------------")
        print("steer :", u[0], "accel :", u[1])

        print("Process time :", toc - tic)

        plt.cla()
        plt.plot(plt_states[0,:i], plt_states[1,:i], "-b", label="Drived") # plot from 0 to i
        plt.grid(True)
        plt.axis("equal")
        plt.plot(x0[0], x0[1], "*g", label="Initial")
        plot_car(x[0], x[1], x[3], steer=u[0]) # plotting w.r.t. rear axle.
        plt.plot(pred_x[0,:], pred_x[1,:], "r")
        plt.plot(path_x, path_y, label="Path")
        plt.plot(Xr[0,:], Xr[1,:], "g")
        plt.pause(0.0001)
        
        # Update States
        u = np.expand_dims(ctrl, axis=1) # from (N,) to (N,1)
        x_next = np.matmul(Ad_list[0], x) + np.matmul(Bd_list[0], u) + gd_list[0]

        plt_states[:,i] = x.T
        plt_actions[:,i] = u.T

        x = x_next
        x_vec = np.squeeze(x, axis=1) # (N,) shape for QP solver, NOT (N,1).

        # Update Predictive States and Actions
        temp_pred_x = pred_x
        temp_pred_u = pred_u
        # index 0
        pred_x[:,0] = x_next.T                   
        pred_u[:,0] = temp_pred_u[:,1] # before u.T
        # index 1 ~ N-2
        for ii in range(0, N-2):
            pred_x[:,ii+1] = temp_pred_x[:,ii+2]
            pred_u[:,ii+1] = temp_pred_u[:,ii+2]
        # index N-1
        pred_x[:,-2] = temp_pred_x[:,N]
        pred_u[:,-2] = temp_pred_u[:,N-1]

        # index N
        # append last state using last A, B matrix and last pred state
        # append last control with last pred control
        last_state = np.expand_dims(temp_pred_x[:,N], axis=1) # from (N,) to (N,1)
        last_control = np.expand_dims(temp_pred_u[:,N-1], axis=1)
        pred_x[:,-1] = np.transpose(vehicle.update_kinematics_model(last_state, last_control))
        pred_u[:,-1] = pred_u[:,N-1]


        # if check_goal(x, xr, goal_dist=1, stop_speed=5):
        #     print("Goal")
        #     break
    
if __name__ == "__main__":
    main()


"""

    

# Plot result
# Figure 1
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,3,1)
ax2 = fig1.add_subplot(2,3,2)
ax3 = fig1.add_subplot(2,3,3)
ax4 = fig1.add_subplot(2,3,4)
ax5 = fig1.add_subplot(2,3,5)
ax6 = fig1.add_subplot(2,3,6)

ax1.set_title("X")
ax2.set_title("Y")
ax3.set_title("Yaw")
ax4.set_title("Vel_x")
ax5.set_title("Vel_y")
ax6.set_title("Yaw_rate")

ax1.plot(plt_tic, plt_x_1)
ax2.plot(plt_tic, plt_x_2)
ax3.plot(plt_tic, plt_x_3)
ax4.plot(plt_tic, plt_x_4)
ax5.plot(plt_tic, plt_x_5)
ax6.plot(plt_tic, plt_x_6)

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()

# Figure 2
fig2 = plt.figure()
ax1 = fig2.add_subplot(1,3,1)
ax2 = fig2.add_subplot(1,3,2)
ax3 = fig2.add_subplot(1,3,3)

ax1.set_title("Steer")
ax2.set_title("Accel")
ax3.set_title("X-Y plot")

ax1.plot(plt_tic, plt_u_1)
ax2.plot(plt_tic, plt_u_2)
ax3.plot(plt_x_1, plt_x_2)


ax1.grid()
ax2.grid()
ax3.grid()
ax3.axis('square')

plt.show()

"""