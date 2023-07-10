import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm

from visualization_vehicle import plot_car

import time


# Initial state
x = np.array([[ 0.],
              [ 0.],
              [ 0.],
              [ 0.],
              [ 0.],
              [ 0.]])  #  [X; Y; Yaw; vel_x; vel_y; Yaw_rate]

u = np.array([[0*math.pi/180],
              [0.0]]) #  [steer; traction_accel]

dt = 0.02 # 50Hz

# num of state, action
nx = 6
nu = 2

l_f = 1.25
l_r = 1.40

def normalize_angle(angle):
    while angle > math.pi:
        angle = angle - 2*math.pi
    while angle < -math.pi:
        angle = angle + 2*math.pi
    
    return angle

def vehicle_dynamics(x, u):
    """
    Front wheel vehicle.
    I30 parameters
    reference:
    https://www.car.info/en-se/hyundai/i30/specs
    https://www.hyundai.news/eu/press-kits/new-generation-hyundai-i30-technical-specifications/
    
    """
    # ===== Model parameters ===== #

    m = 1300  # i30

    width = 1.78
    length = 4.25

    l_f = 1.25
    l_r = 1.40
    wheelbase = l_f + l_r # 2.650
    turning_circle = 10.4 # min radius
    max_steer = math.atan(wheelbase/turning_circle)


    Iz = 1/12 * m * (width**2 + length**2)

    # Iw = 1.8  # wheel inertia
    # rw = 0.3  # wheel radius

    roh = 1.23 # density of air       [kg/m3]
    C_d = 0.34 # drag coefficient
    A_f = 2.0  # vehicle frontal area [m2]
    C_roll = 0.015 # rolling resistance coefficient

    # Pacejka lateral tire model params
    Fz_f = 9.81 * (m * l_r/wheelbase) * 0.001 # vertical force at front axle. [kN]

    a_lat_f = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
    C_lat_f = 1.30
    D_lat_f = a_lat_f[0]*Fz_f**2 + a_lat_f[1]*Fz_f
    BCD_lat_f = a_lat_f[2]*math.sin(a_lat_f[3]*math.atan2(a_lat_f[4]*Fz_f, 1)) # before, atan
    B_lat_f = BCD_lat_f/(C_lat_f*D_lat_f)
    E_lat_f = a_lat_f[5]*Fz_f**2 + a_lat_f[6]*Fz_f + a_lat_f[7]

    Fz_r = 9.81 * (m * l_f/wheelbase) * 0.001 # vertical force at rear axle. [kN]

    a_lat_r = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
    C_lat_r = 1.30
    D_lat_r = a_lat_r[0]*Fz_r**2 + a_lat_r[1]*Fz_r
    BCD_lat_r = a_lat_r[2]*math.sin(a_lat_r[3]*math.atan2(a_lat_r[4]*Fz_r, 1)) # berore, atan
    B_lat_r = BCD_lat_r/(C_lat_r*D_lat_r)
    E_lat_r = a_lat_r[5]*Fz_r**2 + a_lat_r[6]*Fz_r + a_lat_r[7]

    # ===== Discretize Dynamics model ===== #
    """
    Inputs:
        x : states,  [X; Y; Yaw; vel_x; vel_y; Yaw_rate]
        u : actions, [steer; traction_accel]

    Outputs:
        x_k1 : next_states
    """

    # normalize angle
    # x[2] = normalize_angle(x[2])

    # Avoiding zero denominator (for slip angle, expm in discretization procedure)
    if x[3] >=0 and x[3] < 0.5:
        x[4] = 0.    # v_y
        x[5] = 0.    # yaw_rate
        u[0] = 0.    # steer
        if x[3] < 0.3:
            x[3] = 0.3  # v_x

    if x[3] > -0.5 and x[3] < 0:
        x[3] = -0.5
        x[4] = 0.
        x[5] = 0.
        u[0] = 0.
        if x[3] > -0.3:
            x[3] = -0.3

    # States
    yaw =      x[2][0]  # [0] for scalar data
    v_x =      x[3][0]
    v_y =      x[4][0]
    yaw_rate = x[5][0]

    steer =       u[0][0]
    accel_track = u[1][0]

    # Dynamics model
    # Slip angle [deg]
    alpha_f = np.rad2deg(-math.atan2( l_f*yaw_rate + v_y,v_x) + steer)
    alpha_r = np.rad2deg(-math.atan2(-l_r*yaw_rate + v_y,v_x))

    # Lateral force (front & rear)
    Fy_f = D_lat_f * math.sin(C_lat_f * math.atan2(B_lat_f * alpha_f, 1)) # before was atan
    Fy_r = D_lat_r * math.sin(C_lat_r * math.atan2(B_lat_r * alpha_r, 1)) # before was atan

    # Longitudinal force

    # for forward driving.
    R_roll = C_roll * m * 9.81         # rolling resistance. [N] f*(Fzf+Fzr) = f*(mg)
    F_aero = 0.5*roh*C_d*A_f*v_x**2    # aero dynamics drag. [N] 0.5*rho*cd*A.
    Fx_f = m*accel_track - F_aero - R_roll

    # # for backward driving.
    # if v_x < 0:
    #     Fx_f = m*accel_track + F_aero + R_roll

    if abs(v_x) < 0.01:  # No rolling resistance when stopping.
        R_roll = 0

    # Next state
    f = np.array([[v_x*math.cos(yaw) - v_y*math.sin(yaw)],
                  [v_y*math.cos(yaw) + v_x*math.sin(yaw)],
                  [yaw_rate],
                  [1./m*(Fx_f*math.cos(steer) - Fy_f*math.sin(steer) + m*v_y*yaw_rate)],
                  [1./m*(Fx_f*math.sin(steer) + Fy_r + Fy_f*math.cos(steer) - m*v_x*yaw_rate)],
                  [1./Iz*(Fx_f*l_f*math.sin(steer) + Fy_f*l_f*math.cos(steer)- Fy_r*l_r)]])

    # Derivatives of the force laws

    # derivatives of Fx_f
    # for forward driving.
    dFxf_dvx = -roh*C_d*A_f*v_x
    dFxf_daccel = m

    # # for backward driving.
    # if v_x < 0:
    #     dFxf_dvx = 0

    # derivatives of Fy_f
    # before was atan
    dFyf_dvx       =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan2(B_lat_f*alpha_f, 1)))/(1+B_lat_f**2 * alpha_f**2) \
                        *(l_f*yaw_rate + v_y)/((l_f*yaw_rate + v_y)**2+v_x**2)

    dFyf_dvy       =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan2(B_lat_f*alpha_f, 1)))/(1+B_lat_f**2 * alpha_f**2) \
                        *(-v_x/((l_f*yaw_rate + v_y)**2+v_x**2))

    dFyf_dyaw_rate =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan2(B_lat_f*alpha_f, 1)))/(1+B_lat_f**2 * alpha_f**2) \
                        *(-l_f*v_x)/((l_f*yaw_rate + v_y)**2+v_x**2)

    dFyf_dsteer    =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan2(B_lat_f*alpha_f, 1)))/(1+B_lat_f**2 * alpha_f**2)

    # derivatives of Fy_r
    # before was atan
    dFyr_dvx       =    (B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan2(B_lat_r*alpha_r, 1)))/(1+B_lat_r**2*alpha_r**2) \
                        *(-l_r*yaw_rate + v_y)/((-l_r*yaw_rate + v_y)**2+v_x**2)

    dFyr_dvy       =    (B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan2(B_lat_r*alpha_r, 1)))/(1+B_lat_r**2*alpha_r**2) \
                        *(-v_x)/((-l_r*yaw_rate + v_y)**2+v_x**2)

    dFyr_dyaw_rate =    (B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan2(B_lat_r*alpha_r, 1)))/(1+B_lat_r**2*alpha_r**2) \
                        *(l_r*v_x)/((-l_r*yaw_rate + v_y)**2+v_x**2)

    # f1 = v_x*math.cos(yaw) - v_y*math.sin(yaw) 
    df1_dyaw = -v_x*math.sin(yaw) - v_y*math.cos(yaw)
    df1_dvx  = math.cos(yaw)
    df1_dvy  = -math.sin(yaw)

    # f2 = v_y*math.cos(yaw) + v_x*math.sin(yaw)
    df2_dyaw = -v_y*math.sin(yaw) + v_x*math.cos(yaw)
    df2_dvx  = math.sin(yaw)
    df2_dvy  = math.cos(yaw)

    # f3 = yaw_rate
    df3_dyaw_rate = 1.

    # f4 = 1./m*(Fx_f*math.cos(steer) - Fy_f*math.sin(steer) + m*v_y*yaw_rate)
    df4_dvx        = 1/m*(dFxf_dvx * math.cos(steer) - dFyf_dvx * math.sin(steer))
    df4_dvy        = 1/m*(                         - dFyf_dvy * math.sin(steer)       + m*yaw_rate)
    df4_dyaw_rate  = 1/m*(                         - dFyf_dyaw_rate * math.sin(steer) + m*v_y)

    df4_daccel     = 1/m*(dFxf_daccel * math.cos(steer))
    df4_dsteer     = 1/m*(-Fx_f * math.sin(steer) - dFyf_dsteer * math.sin(steer) - Fy_f * math.cos(steer))

    # f5 = 1./m*(Fx_f*math.sin(steer) + Fy_r + Fy_f*math.cos(steer) - m*v_x*yaw_rate)
    df5_dvx        = 1/m*(dFxf_dvx*math.sin(steer) + dFyr_dvx         + dFyf_dvx * math.cos(steer) - m*yaw_rate)
    df5_dvy        = 1/m*(                           dFyr_dvy         + dFyf_dvy * math.cos(steer))   
    df5_dyaw_rate  = 1/m*(                           dFyr_dyaw_rate   + dFyf_dyaw_rate * math.cos(steer) - m*v_x)

    df5_daccel     = 1/m*(dFxf_daccel * math.sin(steer))
    df5_dsteer     = 1/m*(Fx_f * math.cos(steer)                      + dFyf_dsteer * math.cos(steer) - Fy_f * math.sin(steer))

    # f6 = 1./Iz*(Fx_f*l_f*math.sin(steer) + Fy_f*l_f*math.cos(steer)- Fy_r*l_r)
    df6_dvx        = 1/Iz*(dFxf_dvx*l_f*math.sin(steer) + dFyf_dvx * l_f * math.cos(steer)       - dFyr_dvx * l_r)
    df6_dvy        = 1/Iz*(                               dFyf_dvy * l_f * math.cos(steer)       - dFyr_dvy * l_r)
    df6_dyaw_rate  = 1/Iz*(                               dFyf_dyaw_rate * l_f * math.cos(steer) - dFyr_dyaw_rate * l_r)

    df6_daccel     = 1/Iz*(dFxf_daccel*l_f*math.sin(steer))
    df6_dsteer     = 1/Iz*(Fx_f*l_f*math.cos(steer)     + dFyf_dsteer*l_f*math.cos(steer) - Fy_f*l_f*math.sin(steer))

    #  Jacobians.
    Ac=np.array([[0., 0., df1_dyaw, df1_dvx, df1_dvy, 0.           ],
                 [0., 0., df2_dyaw, df2_dvx, df2_dvy, 0.           ],
                 [0., 0., 0.,       0.,      0.,      df3_dyaw_rate],
                 [0., 0., 0.,       df4_dvx, df4_dvy, df4_dyaw_rate],
                 [0., 0., 0.,       df5_dvx, df5_dvy, df5_dyaw_rate],
                 [0., 0., 0.,       df6_dvx, df6_dvy, df6_dyaw_rate]])

    Bc=np.array([[0.,            0.,       ],
                 [0.,            0.,       ],
                 [0.,            0.,       ],
                 [df4_daccel,    df4_dsteer],
                 [df5_daccel,    df5_dsteer],
                 [df6_daccel,    df6_dsteer]])

    gc = f - np.matmul(Ac, x) - np.matmul(Bc, u)

    Bc_aug = np.concatenate([Bc, gc], axis=1)

    # Discretize
    # see report for proof of following method

    # expm_matrix = np.concatenate([np.array([[Ac, Bc_aug]]),
    #                               np.zeros([su+1,sx+su+1])])
    expm_matrix_up = np.concatenate([Ac, Bc_aug], axis=1)
    expm_matrix_down = np.zeros([nu+1,nx+nu+1]) # size of Bc_aug is nu+1
    expm_matrix = np.concatenate([expm_matrix_up, expm_matrix_down])

    tmp = expm(expm_matrix * dt)

    Ad = np.zeros([nx,nx])  # shape: (nx, nx)
    Bd = np.zeros([nx,nu])  # shape: (nu, nu)
    gd = np.zeros([nx,1])
    Ad[0:nx,0:nx] = tmp[0:nx,  0:nx]
    Bd[0:nx,0:nu] = tmp[0:nx, nx:nx+nu]
    gd[0:nx]      = tmp[0:nx, nx+nu:] # sx+su: could not broadcast input array from shape (6) into shape (6,1)

    # =========================


    # Result
    # X_k1 = Ad * x_k + Bd * u_k + gd

    x_k = x
    u_k = u
    x_k1 = np.matmul(Ad, x_k) + np.matmul(Bd, u_k) + gd    # shape: (nx+1, 1)

    # normalize angle
    x_k1[2] = normalize_angle(x_k1[2])

    # # No backward driving
    # if x_k1[3] <= 0:
    #     x_k1[3] = 0

    # # Avoiding zero denominator (for slip angle, expm in discretization procedure)
    # if x_k1[3] >=0 and x_k1[3] < 0.5:
    #     x_k1[4] = 0.    # v_y
    #     x_k1[5] = 0.    # yaw_rate
    #     u[0] = 0.    # steer
    #     if x_k1[3] < 0.3:
    #         x_k1[3] = 0.3  # v_x

    # if x_k1[3] > -0.5 and x_k1[3] < 0:
    #     x_k1[3] = -0.5
    #     x_k1[4] = 0.
    #     x_k1[5] = 0.
    #     u[0] = 0.
    #     if x_k1[3] > -0.3:
    #         x_k1[3] = -0.3

    return x_k1, alpha_f, alpha_r

# Simulation
sim_time = 1000  # time. [sec]
XX = np.zeros([nx, sim_time])
XX_front_axle = np.zeros([2, sim_time])
XX_rear_axle = np.zeros([2, sim_time])

XX_alpha = np.zeros([2, sim_time]) # front, rear

UU = np.zeros([nu, sim_time])
TT = np.linspace(0, sim_time*dt, sim_time)

for i in range(sim_time):

    # timestamp
    tic = time.time()

    # change control input
    if i >= 0 and i < 500:
        u[1] = 1.0

    if i >= 500 and i < 800:
        u[0] = u[0] + 5*dt * math.pi/180
        if u[0] >= 5*math.pi/180:
            u[0] = 5*math.pi/180
        
        u[1] = 2.0

    # if i >= 600 and i < 700:
    #     u[0] = u[0] - 25*dt * math.pi/180
    #     if u[0] <= -10*math.pi/180:
    #         u[0] = -10*math.pi/180
    #     u[1] = 2.0
        
        # u[0] = u[0] + 10*dt * math.pi/180 # 10 deg per sec
        # if u[0] >= 20 * math.pi/180:
        #     u[0] = 20 * math.pi/180
        # u[1] = 2.0

    if i >= 800:
        u[0] = 5*math.pi/180
        u[1] = 0.0

    XX[:,i] = x.T
    TT[i] = dt * i
    UU[:,i] = u.T

    x_next, alpha_f, alpha_r = vehicle_dynamics(x, u)

    XX_alpha[0,i] = alpha_f
    XX_alpha[1,i] = alpha_r

    x = x_next
    

    toc = time.time()
    print("process time :", toc - tic)

print("final position")
print("x, y, yaw :", XX[0,-1], XX[1,-1], XX[2,-1])

for i in range(sim_time):
    XX_front_axle[0,i] = XX[0,i] + l_f * math.cos(XX[2,i])
    XX_front_axle[1,i] = XX[1,i] + l_f * math.sin(XX[2,i])
    XX_rear_axle[0,i] = XX[0,i] - l_r * math.cos(XX[2,i])
    XX_rear_axle[1,i] = XX[1,i] - l_r * math.sin(XX[2,i])

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

ax1.plot(TT, XX[0,:])
ax2.plot(TT, XX[1,:])
ax3.plot(TT, XX[2,:])
ax4.plot(TT, XX[3,:])
ax5.plot(TT, XX[4,:])
ax6.plot(TT, XX[5,:])

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

ax1.plot(TT, UU[0,:])
ax2.plot(TT, UU[1,:])
ax3.plot(XX[0,:], XX[1,:])
ax3.plot(XX_front_axle[0,:], XX_front_axle[1,:], color='red')
ax3.plot(XX_rear_axle[0,:], XX_rear_axle[1,:], color='blue')

ax1.grid()
ax2.grid()
ax3.grid()
ax3.axis('square')

plot_car(XX_rear_axle[0,-1], XX_rear_axle[1,-1], XX[2,-1], UU[0,-1]) # vehicle at final state

# Figure 3
fig3 = plt.figure()
ax1 = fig3.add_subplot(1,2,1)
ax2 = fig3.add_subplot(1,2,2)

ax1.set_title("Front Slip angle")
ax2.set_title("Rear Slip angle")

ax1.plot(TT, XX_alpha[0,:])
ax2.plot(TT, XX_alpha[1,:])

ax1.grid()
ax2.grid()


plt.show()