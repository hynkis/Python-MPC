import math
import time
import numpy as np
import matplotlib.pyplot as plt

import vehicle_dynamics
from visualization_vehicle import plot_car


def main():
    # ===== Vehicle parameters =====
    l_f = 1.25
    l_r = 1.40

    dt = 0.02

    m=1300
    l_f=l_f
    l_r=l_r
    width = 1.78
    length = 4.25
    turning_circle=10.4
    C_d = 0.34
    A_f = 2.0
    C_roll = 0.015

    vehicle = vehicle_dynamics.Vehicle_Dynamics(m=m,
                                                l_f=l_f,
                                                l_r=l_r,
                                                width = width,
                                                length = length,
                                                turning_circle=turning_circle,
                                                C_d = C_d,
                                                A_f = A_f,
                                                C_roll = C_roll,
                                                dt = dt)

    # ===== Initial state =====
    # States  : [x; y; v; yaw]
    # Actions : [steer; accel]
    x = np.array([[0.0],
                  [0.0],
                  [0.0],
                  [np.deg2rad(0)]])    #  [X; Y; V; Yaw]
    u = np.array([[0*math.pi/180],
                  [0.0]])               #  [steer; accel]

    # Reference state
    xr = np.array([[10.0],
                   [10.0],
                   [10.0],
                   [np.deg2rad(90)]])  #  [X; Y; vel_x; Yaw]

    # ===== Simulation Setup =====
    nx = x.shape[0]
    nu = u.shape[0]

    sim_time = 1000
    sim_tic = np.linspace(0, sim_time, sim_time)
    states_plot = np.zeros((nx, sim_time))
    actions_plot = np.zeros((nu, sim_time))

    states_nonlinear_plot = np.zeros((nx, sim_time))
    states_nonlinear_plot[:,0] = x.T

    # Ad, Bd, gd = vehicle.get_kinematics_model(x, u)

    for i in range(sim_time):
        print("===================== nsim :", i, "=====================")

        # timestamp
        tic = time.time()

        if i >= 0.5*sim_time:
            xr = np.array([[10.0],
                           [10.0],
                           [10.0],
                           [np.deg2rad(90)]])  #  [X; Y; vel_x; Yaw]

        # ===== Get System matrices =====
        # using current state and past action
        Ad, Bd, gd = vehicle.get_kinematics_model(x, u)
        
        # ===== PID Control ===== #
        # p_steer = 5.0
        # error_yaw = vehicle_dynamics.normalize_angle(xr[2] - x[2])
        
        # # Steer control
        # u[0] = p_steer * error_yaw
        # if u[0] >= np.deg2rad(15):
        #     u[0] = np.deg2rad(15)
        # if u[0] <= -np.deg2rad(15):
        #     u[0] = -np.deg2rad(15)

        # # Speed control
        # p_accel = 10.0
        # error_vx = xr[3] - x[3]
        # u[1] = p_accel * error_vx
        # if u[1] >= 1:
        #     u[1] = 1
        # if u[1] <= -3:
        #     u[1] = -3

        u[0] = np.deg2rad(10)
        u[1] = 1

        # Plot
        print("x :", x[0], "y :", x[1], "v :", x[2], "yaw :", x[3])
        print("--------------------------------------------------")
        print("steer :", u[0], "accel :", u[1])

        plt.cla()
        plt.plot(states_plot[0,:], states_plot[1,:], "-b", label="Drived")
        plt.plot(states_nonlinear_plot[0,:], states_nonlinear_plot[1,:], "--r", label="Non linear")
        plt.grid(True)
        plt.axis("equal")
        plot_car(x[0], x[1], x[3], steer=u[0]) # plotting w.r.t. rear axle.
        # plt.plot(x_pred, y_pred, "r")
        # plt.plot(X_pred_last, Y_pred_last, ".r")
        plt.pause(0.0001)

        # Update States
        x_next = np.matmul(Ad, x) + np.matmul(Bd, u) + gd
        x_next_nonlinear = vehicle.update_kinematic_model(x, u)
        

        x_next[3] = vehicle_dynamics.normalize_angle(x_next[3])
        print("x_next :", x_next)
        states_plot[:,i]  = x.T
        actions_plot[:,i] = u.T
        states_nonlinear_plot[:,i] = x_next_nonlinear.T

        x = x_next



if __name__ == "__main__":
    main()