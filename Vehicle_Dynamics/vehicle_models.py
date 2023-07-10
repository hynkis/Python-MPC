import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm

from visualization_vehicle import plot_car, plot_car_force

import time

def normalize_angle(angle):
    
    while angle > math.pi:
        print("===== normailized! angle :", angle)
        angle = angle - 2*math.pi
        
    while angle < -math.pi:
        print("===== normailized! angle :", angle)
        angle = angle + 2*math.pi
    
    return angle

class Vehicle_Dynamics(object):
    """
    Vehicle Dynamics Model.
    """

    def __init__(self, m=1300, l_f=1.25, l_r=1.40, width = 1.78, length = 4.25, turning_circle=10.4, C_d = 0.34, A_f = 2.0, C_roll = 0.015, dt = 0.02):
        """
        Vehicle Dynamics Parameters
        """
        self.m = m                                                # mass. [kg]
        self.l_f = l_f                                            # from front axle to c.g. [m]
        self.l_r = l_r                                            # from rear axle to c.g. [m]
        self.wheelbase = l_f + l_r                                # wheelbase [m]  
        self.width = width                                        # width [m]
        self.length = length                                      # length [m]
        self.turning_circle = turning_circle                      # turning_circle. min radius. [m]
        self.max_steer = math.atan(self.wheelbase / turning_circle)  # maximum steer angle. [rad]
        self.Iz = 1/12 * m * (width**2 + length**2)               # moment of inertia. [kg m]

        # Iw = 1.8  # wheel inertia
        # rw = 0.3  # wheel radius
        self.C_d = C_d                                            # drag coefficient
        self.A_f = A_f                                            # vehicle frontal area [m2]
        self.C_roll = C_roll                                      # rolling resistance coefficient
        self.roh = 1.23                                           # density of air       [kg/m3]

        self.dt = dt                                              # sampling rate. for discritization. [sec]
        self.nx = 6
        self.nu = 2

    def get_dynamics_model(self, x, u):
        """
        ===== Discretize Linearized Dynamics model =====

        Inputs:
            x : states,  [X; Y; Yaw; vel_x; vel_y; Yaw_rate]
            u : actions, [steer; traction_accel]

        Outputs:
            Ad :      Linearized Discrete A matrix.
            Bd :      Linearized Discrete B matrix.
            gd :      Linearized Discrete vector for accurate next state. 
            x_k1 :    next_states.
            alpha_f : Slip angle front.
            alpha_r : Slip angle rear.

        Front wheel vehicle.
        I30 parameters
        reference:
        https://www.car.info/en-se/hyundai/i30/specs
        https://www.hyundai.news/eu/press-kits/new-generation-hyundai-i30-technical-specifications/
        
        """
        if x.ndim < 2:
            x = np.expand_dims(x, axis=1) # shape of x should be (N,1), not (N,)
        if u.ndim < 2:
            u = np.expand_dims(u, axis=1)

        # ===== Model parameters ===== #

        # num of state, action
        nx = self.nx
        nu = self.nu

        m = self.m  # i30

        width = self.width
        length = self.length

        l_f = self.l_f
        l_r = self.l_r
        wheelbase = self.wheelbase
        turning_circle = self.turning_circle
        max_steer = self.max_steer

        Iz = self.Iz

        # Iw = 1.8  # wheel inertia
        # rw = 0.3  # wheel radius

        roh = self.roh               # density of air       [kg/m3]
        C_d = self.C_d               # drag coefficient
        A_f = self.A_f               # vehicle frontal area [m2]
        C_roll = self.C_roll         # rolling resistance coefficient

        dt = self.dt                 # sampling time.       [sec]

        
        """
        Pacejka lateral tire model params
        
        """
        Fz_f = 9.81 * (m * l_r/wheelbase) * 0.001 # vertical force at front axle. [kN]

        a_lat_f = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
        C_lat_f = 1.30
        D_lat_f = a_lat_f[0]*Fz_f**2 + a_lat_f[1]*Fz_f
        BCD_lat_f = a_lat_f[2]*math.sin(a_lat_f[3]*math.atan(a_lat_f[4]*Fz_f)) # before, atan
        # B_lat_f = BCD_lat_f/(C_lat_f*D_lat_f)
        B_lat_f = BCD_lat_f/(C_lat_f*D_lat_f) * 180/np.pi  # for radian sideslip angle.
        E_lat_f = a_lat_f[5]*Fz_f**2 + a_lat_f[6]*Fz_f + a_lat_f[7]

        Fz_r = 9.81 * (m * l_f/wheelbase) * 0.001 # vertical force at rear axle. [kN]

        a_lat_r = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
        C_lat_r = 1.30
        D_lat_r = a_lat_r[0]*Fz_r**2 + a_lat_r[1]*Fz_r
        BCD_lat_r = a_lat_r[2]*math.sin(a_lat_r[3]*math.atan(a_lat_r[4]*Fz_r)) # berore, atan
        # B_lat_r = BCD_lat_r/(C_lat_r*D_lat_r)
        B_lat_r = BCD_lat_r/(C_lat_r*D_lat_r) * 180/np.pi  # for radian sideslip angle
        E_lat_r = a_lat_r[5]*Fz_r**2 + a_lat_r[6]*Fz_r + a_lat_r[7]

        """
        ===== Discretize Linearized Dynamics model =====
        """

        # normalize angle
        # x[2] = normalize_angle(x[2])

        # Avoiding zero denominator (for slip angle, expm in discretization procedure)
        # before 19.07.31, 0.5 m/s
        if x[3,0] >=0 and x[3,0] < 0.5:
            # x[3] = 0.0
            x[4,0] = 0.       # v_y
            x[5,0] = 0.       # yaw_rate
            u[0,0] = 0.       # steer
            if x[3,0] < 0.3:
                x[3,0] = 0.3  # v_x
            print("Avoiding zero denominator")

        if x[3,0] > -0.5 and x[3,0] < 0:
            # x[3] = 0.
            x[4,0] = 0.
            x[5,0] = 0.
            u[0,0] = 0.
            if x[3,0] > -0.3:
                x[3,0] = -0.3
            print("Avoiding zero denominator")

        # States
        yaw =         x[2,0]  # [0] for scalar data
        v_x =         x[3,0]
        v_y =         x[4,0]
        yaw_rate =    x[5,0]

        steer =       u[0,0]
        accel_track = u[1,0]

        # Dynamics model
        # Slip angle [deg]
        # TODO: use radian unit
        # alpha_f = np.rad2deg(-math.atan2( l_f*yaw_rate + v_y,v_x) + steer)
        # alpha_r = np.rad2deg(-math.atan2(-l_r*yaw_rate + v_y,v_x))
        
        alpha_f = -math.atan2( l_f*yaw_rate + v_y,v_x) + steer
        alpha_r = -math.atan2(-l_r*yaw_rate + v_y,v_x)

        # Lateral force (front & rear)
        # Fy_f = D_lat_f * math.sin(C_lat_f * math.atan2(B_lat_f * alpha_f, 1)) # before was atan
        # Fy_r = D_lat_r * math.sin(C_lat_r * math.atan2(B_lat_r * alpha_r, 1)) # before was atan
        Fy_f = D_lat_f * math.sin(C_lat_f * math.atan(B_lat_f * alpha_f))
        Fy_r = D_lat_r * math.sin(C_lat_r * math.atan(B_lat_r * alpha_r))

        # Longitudinal force

        # for both forward and backward driving.
        R_roll = C_roll * m * 9.81 * np.sign(v_x)               # rolling resistance. [N] f*(Fzf+Fzr) = f*(mg)
        F_aero = 0.5*roh*C_d*A_f*v_x**2 * np.sign(v_x)          # aero dynamics drag. [N] 0.5*rho*cd*A.
        Fx_f = m*accel_track - F_aero - R_roll

        # X dot
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
        dFyf_dvx       =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan(B_lat_f*alpha_f)))/(1+B_lat_f**2 * alpha_f**2) \
                            *(l_f*yaw_rate + v_y)/((l_f*yaw_rate + v_y)**2+v_x**2)

        dFyf_dvy       =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan(B_lat_f*alpha_f)))/(1+B_lat_f**2 * alpha_f**2) \
                            *(-v_x/((l_f*yaw_rate + v_y)**2+v_x**2))

        dFyf_dyaw_rate =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan(B_lat_f*alpha_f)))/(1+B_lat_f**2 * alpha_f**2) \
                            *(-l_f*v_x)/((l_f*yaw_rate + v_y)**2+v_x**2)

        dFyf_dsteer    =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan(B_lat_f*alpha_f)))/(1+B_lat_f**2 * alpha_f**2)

        # derivatives of Fy_r
        # before was atan
        dFyr_dvx       =    (B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan(B_lat_r*alpha_r)))/(1+B_lat_r**2*alpha_r**2) \
                            *(-l_r*yaw_rate + v_y)/((-l_r*yaw_rate + v_y)**2+v_x**2)

        dFyr_dvy       =    (B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan(B_lat_r*alpha_r)))/(1+B_lat_r**2*alpha_r**2) \
                            *(-v_x)/((-l_r*yaw_rate + v_y)**2+v_x**2)

        dFyr_dyaw_rate =    (B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan(B_lat_r*alpha_r)))/(1+B_lat_r**2*alpha_r**2) \
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

        # Bc=np.array([[0.,            0.,       ],
        #              [0.,            0.,       ],
        #              [0.,            0.,       ],
        #              [df4_daccel,    df4_dsteer],
        #              [df5_daccel,    df5_dsteer],
        #              [df6_daccel,    df6_dsteer]])

        Bc=np.array([[0.,            0.,       ],
                     [0.,            0.,       ],
                     [0.,            0.,       ],
                     [df4_dsteer,    df4_daccel],
                     [df5_dsteer,    df5_daccel],
                     [df6_dsteer,    df6_daccel]])

        gc = f - np.matmul(Ac, x) - np.matmul(Bc, u)
        
        # Discretize
        # # -- Exponential Matrix
        # Bc_aug = np.concatenate([Bc, gc], axis=1)
        # # expm_matrix = np.concatenate([np.array([[Ac, Bc_aug]]),
        # #                               np.zeros([su+1,sx+su+1])])
        # expm_matrix_up = np.concatenate([Ac, Bc_aug], axis=1)
        # expm_matrix_down = np.zeros([nu+1,nx+nu+1]) # size of Bc_aug is nu+1
        # expm_matrix = np.concatenate([expm_matrix_up, expm_matrix_down])

        # tmp = expm(expm_matrix * dt)

        # Ad = np.zeros([nx,nx])  # shape: (nx, nx)
        # Bd = np.zeros([nx,nu])  # shape: (nu, nu)
        # gd = np.zeros([nx,1])
        # Ad[0:nx,0:nx] = tmp[0:nx,  0:nx]
        # Bd[0:nx,0:nu] = tmp[0:nx, nx:nx+nu]
        # gd[0:nx]      = tmp[0:nx, nx+nu:] # sx+su: could not broadcast input array from shape (6) into shape (6,1)
        # #  following to avoid numerical errors
        # Ad[-1,-1] = 1.
        # Bd[-1,-1] = self.dt

        # -- Forward Euler Method (Faster and similar with Exponential Matrix method. 19.08.01)
        Ad = np.zeros((nx, nx))
        for i in range(nx):
            Ad[i, i] = 1
        Ad = Ad + Ac * self.dt # Ad = I + dt*Ac

        Bd = Bc * self.dt      # Bd = dt*Bc
        gd = gc * self.dt      # gd = (f(x0,u0) - Acx0 - Bcu0) * dt


        # =========================


        # Result
        # X_k1 = Ad * x_k + Bd * u_k + gd

        # x_k = x
        # u_k = u
        # x_k1 = np.matmul(Ad, x_k) + np.matmul(Bd, u_k) + gd    # shape: (nx+1, 1)

        # normalize angle
        # x_k1[2] = normalize_angle(x_k1[2])

        return Ad, Bd, gd
        # return Ad, Bd, gd, alpha_f, alpha_r

    def update_dynamics_model(self, x, u):
        """
        ===== Discretize Linearized Dynamics model =====

        Inputs:
            x : states,  [X; Y; Yaw; vel_x; vel_y; Yaw_rate]
            u : actions, [steer; traction_accel]

        Outputs:
            x_next  : next_states.
            alpha_f : side slip on front axle
            alpha_r : side slip on rear axle
        """
        if x.ndim < 2:
            x = np.expand_dims(x, axis=1) # shape of x should be (N,1), not (N,)
        if u.ndim < 2:
            u = np.expand_dims(u, axis=1)
        # ===== Model parameters ===== #

        # num of state, action
        nx = self.nx
        nu = self.nu

        m = self.m  # i30

        width = self.width
        length = self.length

        l_f = self.l_f
        l_r = self.l_r
        wheelbase = self.wheelbase
        turning_circle = self.turning_circle
        max_steer = self.max_steer

        Iz = self.Iz

        # Iw = 1.8  # wheel inertia
        # rw = 0.3  # wheel radius

        roh = self.roh               # density of air       [kg/m3]
        C_d = self.C_d               # drag coefficient
        A_f = self.A_f               # vehicle frontal area [m2]
        C_roll = self.C_roll         # rolling resistance coefficient

        dt = self.dt                 # sampling time.       [sec]

        
        """
        Pacejka lateral tire model params
        
        """
        Fz_f = 9.81 * (m * l_r/wheelbase) * 0.001 # vertical force at front axle. [kN]

        a_lat_f = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
        C_lat_f = 1.30
        D_lat_f = a_lat_f[0]*Fz_f**2 + a_lat_f[1]*Fz_f
        BCD_lat_f = a_lat_f[2]*math.sin(a_lat_f[3]*math.atan(a_lat_f[4]*Fz_f)) # before, atan
        # B_lat_f = BCD_lat_f/(C_lat_f*D_lat_f)
        B_lat_f = BCD_lat_f/(C_lat_f*D_lat_f) * 180/np.pi  # for radian sideslip angle
        E_lat_f = a_lat_f[5]*Fz_f**2 + a_lat_f[6]*Fz_f + a_lat_f[7]

        Fz_r = 9.81 * (m * l_f/wheelbase) * 0.001 # vertical force at rear axle. [kN]

        a_lat_r = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
        C_lat_r = 1.30
        D_lat_r = a_lat_r[0]*Fz_r**2 + a_lat_r[1]*Fz_r
        BCD_lat_r = a_lat_r[2]*math.sin(a_lat_r[3]*math.atan(a_lat_r[4]*Fz_r)) # berore, atan
        # B_lat_r = BCD_lat_r/(C_lat_r*D_lat_r)
        B_lat_r = BCD_lat_r/(C_lat_r*D_lat_r) * 180/np.pi   # for radian sideslip angle
        E_lat_r = a_lat_r[5]*Fz_r**2 + a_lat_r[6]*Fz_r + a_lat_r[7]

        """
        ===== Discretize Linearized Dynamics model =====
        """

        # normalize angle
        # x[2] = normalize_angle(x[2])

        # Avoiding zero denominator (for slip angle, expm in discretization procedure)
        # before 19.07.31, 0.5 m/s
        if x[3,0] >=0 and x[3,0] < 0.5:
            # x[3] = 0.0
            x[4,0] = 0.       # v_y
            x[5,0] = 0.       # yaw_rate
            u[0,0] = 0.       # steer
            if x[3,0] < 0.3:
                x[3,0] = 0.3  # v_x
            print("Avoiding zero denominator")

        if x[3,0] > -0.5 and x[3,0] < 0:
            # x[3] = 0.
            x[4,0] = 0.
            x[5,0] = 0.
            u[0,0] = 0.
            if x[3,0] > -0.3:
                x[3,0] = -0.3
            print("Avoiding zero denominator")

        # States
        yaw =         x[2,0]  # [0] for scalar data
        v_x =         x[3,0]
        v_y =         x[4,0]
        yaw_rate =    x[5,0]

        steer =       u[0,0]
        accel_track = u[1,0]

        # Dynamics model
        # Slip angle [deg]
        # alpha_f = np.rad2deg(-math.atan2( l_f*yaw_rate + v_y,v_x) + steer)
        # alpha_r = np.rad2deg(-math.atan2(-l_r*yaw_rate + v_y,v_x))

        alpha_f = -math.atan2( l_f*yaw_rate + v_y,v_x) + steer
        alpha_r = -math.atan2(-l_r*yaw_rate + v_y,v_x)

        # Lateral force (front & rear)
        # Fy_f = D_lat_f * math.sin(C_lat_f * math.atan2(B_lat_f * alpha_f, 1)) # before was atan
        # Fy_r = D_lat_r * math.sin(C_lat_r * math.atan2(B_lat_r * alpha_r, 1)) # before was atan
        Fy_f = D_lat_f * math.sin(C_lat_f * math.atan(B_lat_f * alpha_f))
        Fy_r = D_lat_r * math.sin(C_lat_r * math.atan(B_lat_r * alpha_r))

        # Longitudinal force

        # for both forward and backward driving.
        R_roll = C_roll * m * 9.81 * np.sign(v_x)               # rolling resistance. [N] f*(Fzf+Fzr) = f*(mg)
        F_aero = 0.5*roh*C_d*A_f*v_x**2 * np.sign(v_x)          # aero dynamics drag. [N] 0.5*rho*cd*A.
        Fx_f = m*accel_track - F_aero - R_roll

        # Next state
        x_dot = np.array([[v_x*math.cos(yaw) - v_y*math.sin(yaw)],
                            [v_y*math.cos(yaw) + v_x*math.sin(yaw)],
                            [yaw_rate],
                            [1./m*(Fx_f*math.cos(steer) - Fy_f*math.sin(steer) + m*v_y*yaw_rate)],
                            [1./m*(Fx_f*math.sin(steer) + Fy_r + Fy_f*math.cos(steer) - m*v_x*yaw_rate)],
                            [1./Iz*(Fx_f*l_f*math.sin(steer) + Fy_f*l_f*math.cos(steer)- Fy_r*l_r)]])

        x_next = x + x_dot * dt

        return x_next, alpha_f, alpha_r

    def get_rear_wheel_dynamics_model(self, x, u):

        if x.ndim < 2:
            x = np.expand_dims(x, axis=1) # shape of x should be (N,1), not (N,)
        if u.ndim < 2:
            u = np.expand_dims(u, axis=1)

        # ===== Model parameters ===== #
        # num of state, action
        nx = self.nx
        nu = self.nu

        m = self.m  # i30

        width = self.width
        length = self.length

        l_f = self.l_f
        l_r = self.l_r
        wheelbase = self.wheelbase
        turning_circle = self.turning_circle
        max_steer = self.max_steer

        Iz = self.Iz

        # Iw = 1.8  # wheel inertia
        # rw = 0.3  # wheel radius

        roh = self.roh               # density of air       [kg/m3]
        C_d = self.C_d               # drag coefficient
        A_f = self.A_f               # vehicle frontal area [m2]
        C_roll = self.C_roll         # rolling resistance coefficient

        dt = self.dt                 # sampling time.       [sec]

        """
        Pacejka lateral tire model params
        
        """
        Fz_f = 9.81 * (m * l_r/wheelbase) * 0.001 # vertical force at front axle. [kN]

        a_lat_f = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
        C_lat_f = 1.30
        D_lat_f = a_lat_f[0]*Fz_f**2 + a_lat_f[1]*Fz_f
        BCD_lat_f = a_lat_f[2]*math.sin(a_lat_f[3]*math.atan(a_lat_f[4]*Fz_f)) # before, atan
        B_lat_f = BCD_lat_f/(C_lat_f*D_lat_f)
        E_lat_f = a_lat_f[5]*Fz_f**2 + a_lat_f[6]*Fz_f + a_lat_f[7]

        Fz_r = 9.81 * (m * l_f/wheelbase) * 0.001 # vertical force at rear axle. [kN]

        a_lat_r = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
        C_lat_r = 1.30
        D_lat_r = a_lat_r[0]*Fz_r**2 + a_lat_r[1]*Fz_r
        BCD_lat_r = a_lat_r[2]*math.sin(a_lat_r[3]*math.atan(a_lat_r[4]*Fz_r)) # berore, atan
        B_lat_r = BCD_lat_r/(C_lat_r*D_lat_r)
        E_lat_r = a_lat_r[5]*Fz_r**2 + a_lat_r[6]*Fz_r + a_lat_r[7]

        # MPCC
        Cm1 = 17303.
        Cm2 = 175.

        """
        ===== Discretize Linearized Dynamics model =====
        """

        # normalize angle
        # x[2] = normalize_angle(x[2])

        # Avoiding zero denominator (for slip angle, expm in discretization procedure)
        # before 19.07.31, 0.5 m/s
        if x[3,0] >=0 and x[3,0] < 0.5:
            # x[3] = 0.0
            x[4,0] = 0.       # v_y
            x[5,0] = 0.       # yaw_rate
            u[0,0] = 0.       # steer
            if x[3,0] < 0.3:
                x[3,0] = 0.3  # v_x
            print("Avoiding zero denominator")

        if x[3,0] > -0.5 and x[3,0] < 0:
            # x[3] = 0.
            x[4,0] = 0.
            x[5,0] = 0.
            u[0,0] = 0.
            if x[3,0] > -0.3:
                x[3,0] = -0.3
            print("Avoiding zero denominator")

        # States
        yaw =         x[2,0]  # [0] for scalar data
        v_x =         x[3,0]
        v_y =         x[4,0]
        yaw_rate =    x[5,0]

        steer         =       u[0,0]
        accel_track   =       u[1,0]

        # Dynamics model
        # Slip angle [deg]
        # alpha_f = np.rad2deg(-math.atan2( l_f*yaw_rate + v_y,v_x) + steer)
        # alpha_r = np.rad2deg(-math.atan2(-l_r*yaw_rate + v_y,v_x))

        alpha_f = np.rad2deg(-math.atan2( l_f*yaw_rate + v_y,v_x) + steer)
        alpha_r = np.rad2deg(-math.atan2(-l_r*yaw_rate + v_y,v_x))

        # Lateral force (front & rear)
        # Fy_f = D_lat_f * math.sin(C_lat_f * math.atan2(B_lat_f * alpha_f, 1)) # before was atan
        # Fy_r = D_lat_r * math.sin(C_lat_r * math.atan2(B_lat_r * alpha_r, 1)) # before was atan
        Fy_f = D_lat_f * math.sin(C_lat_f * math.atan(B_lat_f * alpha_f))
        Fy_r = D_lat_r * math.sin(C_lat_r * math.atan(B_lat_r * alpha_r))

        R_roll = C_roll * m * 9.81 * np.sign(v_x)               # rolling resistance. [N] f*(Fzf+Fzr) = f*(mg)
        F_aero = 0.5*roh*C_d*A_f*v_x**2 * np.sign(v_x)          # aero dynamics drag. [N] 0.5*rho*cd*A.

        # Fx_f = (Cm1 * D-Cm2 * D * v_x) - R_roll - F_aero
        Fx_f = m * accel_track - R_roll - F_aero

        f = np.array([[v_x*math.cos(yaw) - v_y*math.sin(yaw)],
                    [v_y*math.cos(yaw) + v_x*math.sin(yaw)],
                    [yaw_rate],
                    [1./m*(Fx_f - Fy_f*math.sin(steer) + m*v_y*yaw_rate)],
                    [1./m*(Fy_r + Fy_f*math.cos(steer) - m*v_x*yaw_rate)],
                    [1./Iz*(Fy_f*l_f*math.cos(steer)- Fy_r*l_r)]])
        # Derivatives of the force laws
        #  Fx_f
        dFrx_dvx = - roh*C_d*A_f*v_x
        dFrx_daccel_track  = m
        #  Fy_r
        dFry_dvx = ((B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan(B_lat_r*alpha_r)))/(1+B_lat_r**2*alpha_r**2)) \
                *(-(l_r*yaw_rate - v_y)/((-l_r*yaw_rate + v_y)**2+v_x**2))

        dFry_dvy = ((B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan(B_lat_r*alpha_r)))/(1+B_lat_r**2*alpha_r**2)) \
                *((-v_x)/((-l_r*yaw_rate + v_y)**2+v_x**2))

        dFry_dyaw_rate = ((B_lat_r*C_lat_r*D_lat_r*math.cos(C_lat_r*math.atan(B_lat_r*alpha_r)))/(1+B_lat_r**2*alpha_r**2)) \
                *((l_r*v_x)/((-l_r*yaw_rate + v_y)**2+v_x**2))
        #  Fy_f 

        dFfy_dvx =     (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan(B_lat_f*alpha_f)))/(1+B_lat_f**2*alpha_f**2) \
                    *((l_f*yaw_rate + v_y)/((l_f*yaw_rate + v_y)**2+v_x**2))


        dFfy_dvy =     (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan(B_lat_f*alpha_f)))/(1+B_lat_f**2*alpha_f**2) \
                    *(-v_x/((l_f*yaw_rate + v_y)**2+v_x**2))

        dFfy_dyaw_rate =    (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan(B_lat_f*alpha_f)))/(1+B_lat_f**2*alpha_f**2) \
                    *((-l_f*v_x)/((l_f*yaw_rate + v_y)**2+v_x**2))

        dFfy_dsteer =  (B_lat_f*C_lat_f*D_lat_f*math.cos(C_lat_f*math.atan(B_lat_f*alpha_f)))/(1+B_lat_f**2 * alpha_f**2) 

        #  f1 = v_x*math.cos(yaw) - v_y*math.sin(yaw)
        df1_dyaw = -v_x*math.sin(yaw) - v_y*math.cos(yaw)
        df1_dvx  = math.cos(yaw)
        df1_dvy  = -math.sin(yaw)

        #  f2 = v_y*math.cos(yaw) + v_x*math.sin(yaw)
        df2_dyaw = -v_y*math.sin(yaw) + v_x*math.cos(yaw)
        df2_dvx  = math.sin(yaw)
        df2_dvy  = math.cos(yaw)

        #  f3 = yaw_rate
        df3_dyaw_rate = 1.

        #  f4 = 1/m*(Fx_f - Fy_f*math.sin(steer) + m*v_y*yaw_rate)
        df4_dvx     = 1/m*(dFrx_dvx - dFfy_dvx * math.sin(steer))
        df4_dvy     = 1/m*(           - dFfy_dvy*math.sin(steer)     + m*yaw_rate)
        df4_dyaw_rate  = 1/m*(           - dFfy_dyaw_rate*math.sin(steer) + m*v_y)
        df4_daccel_track      = 1/m*     dFrx_daccel_track
        df4_dsteer  = 1/m*(           - dFfy_dsteer*math.sin(steer)  - Fy_f*math.cos(steer))


        #  f5 = 1/m*(Fy_r + Fy_f*math.cos(steer) - m*v_x*yaw_rate)
        df5_dvx     = 1/m*(dFry_dvx     + dFfy_dvx*math.cos(steer)     - m*yaw_rate)
        df5_dvy     = 1/m*(dFry_dvy     + dFfy_dvy*math.cos(steer))   
        df5_dyaw_rate  = 1/m*(dFry_dyaw_rate + dFfy_dyaw_rate*math.cos(steer) - m*v_x)
        df5_dsteer  = 1/m*(               dFfy_dsteer*math.cos(steer)  - Fy_f*math.sin(steer))

        #  f6 = 1/Iz*(Fy_f*l_f*math.cos(steer)- Fy_r*l_r)
        df6_dvx     = 1/Iz*(dFfy_dvx*l_f*math.cos(steer)    - dFry_dvx*l_r)
        df6_dvy     = 1/Iz*(dFfy_dvy*l_f*math.cos(steer)    - dFry_dvy*l_r)
        df6_dyaw_rate  = 1/Iz*(dFfy_dyaw_rate*l_f*math.cos(steer) - dFry_dyaw_rate*l_r)
        df6_dsteer  = 1/Iz*(dFfy_dsteer*l_f*math.cos(steer)  - Fy_f*l_f*math.sin(steer))

        #  Jacobians
        Ac=np.array([[0., 0., df1_dyaw, df1_dvx, df1_dvy, 0.        ],
                    [0., 0., df2_dyaw, df2_dvx, df2_dvy, 0.        ],
                    [0., 0., 0.,       0.,      0.,      df3_dyaw_rate],
                    [0., 0., 0.,       df4_dvx, df4_dvy, df4_dyaw_rate],
                    [0., 0., 0.,       df5_dvx, df5_dvy, df5_dyaw_rate],
                    [0., 0., 0.,       df6_dvx, df6_dvy, df6_dyaw_rate]])

        # Bc=np.array([[0.,     0.,       ],
        #             [0.,     0.,       ],
        #             [0.,     0.,       ],
        #             [df4_daccel_track, df4_dsteer],
        #             [0.,     df5_dsteer],
        #             [0.,     df6_dsteer]])
        Bc=np.array([[0.,         0.,       ],
                     [0.,         0.,       ],
                     [0.,         0.,       ],
                     [df4_dsteer, df4_daccel_track    ],
                     [df5_dsteer, 0.        ],
                     [df6_dsteer, 0.        ]])

        gc = f - np.matmul(Ac, x) - np.matmul(Bc, u)

        # -- Forward Euler Method (Faster and similar with Exponential Matrix method. 19.08.01)
        Ad = np.zeros((nx, nx))
        for i in range(nx):
            Ad[i, i] = 1
        Ad = Ad + Ac * self.dt # Ad = I + dt*Ac

        Bd = Bc * self.dt      # Bd = dt*Bc
        gd = gc * self.dt      # gd = (f(x0,u0) - Acx0 - Bcu0) * dt

        return Ad, Bd, gd

    def update_rear_wheel_dynamics_model(self, x, u):

        if x.ndim < 2:
            x = np.expand_dims(x, axis=1) # shape of x should be (N,1), not (N,)
        if u.ndim < 2:
            u = np.expand_dims(u, axis=1)
        # ===== Model parameters ===== #

        # num of state, action
        nx = self.nx
        nu = self.nu

        m = self.m  # i30

        width = self.width
        length = self.length

        l_f = self.l_f
        l_r = self.l_r
        wheelbase = self.wheelbase
        turning_circle = self.turning_circle
        max_steer = self.max_steer

        Iz = self.Iz

        # Iw = 1.8  # wheel inertia
        # rw = 0.3  # wheel radius

        roh = self.roh               # density of air       [kg/m3]
        C_d = self.C_d               # drag coefficient
        A_f = self.A_f               # vehicle frontal area [m2]
        C_roll = self.C_roll         # rolling resistance coefficient

        dt = self.dt                 # sampling time.       [sec]

        """
        Pacejka lateral tire model params
        
        """
        Fz_f = 9.81 * (m * l_r/wheelbase) * 0.001 # vertical force at front axle. [kN]

        a_lat_f = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
        C_lat_f = 1.30
        D_lat_f = a_lat_f[0]*Fz_f**2 + a_lat_f[1]*Fz_f
        BCD_lat_f = a_lat_f[2]*math.sin(a_lat_f[3]*math.atan(a_lat_f[4]*Fz_f)) # before, atan
        B_lat_f = BCD_lat_f/(C_lat_f*D_lat_f)
        E_lat_f = a_lat_f[5]*Fz_f**2 + a_lat_f[6]*Fz_f + a_lat_f[7]

        Fz_r = 9.81 * (m * l_f/wheelbase) * 0.001 # vertical force at rear axle. [kN]

        a_lat_r = [-22.1, 1011, 1078, 1.82, 0.208, 0.000, -0.354, 0.707] # for Fy
        C_lat_r = 1.30
        D_lat_r = a_lat_r[0]*Fz_r**2 + a_lat_r[1]*Fz_r
        BCD_lat_r = a_lat_r[2]*math.sin(a_lat_r[3]*math.atan(a_lat_r[4]*Fz_r)) # berore, atan
        B_lat_r = BCD_lat_r/(C_lat_r*D_lat_r)
        E_lat_r = a_lat_r[5]*Fz_r**2 + a_lat_r[6]*Fz_r + a_lat_r[7]

        # MPCC
        Cm1 = 17303.
        Cm2 = 175.

        """
        ===== Discretize Linearized Dynamics model =====
        """

        # normalize angle
        # x[2] = normalize_angle(x[2])

        # Avoiding zero denominator (for slip angle, expm in discretization procedure)
        # before 19.07.31, 0.5 m/s
        if x[3,0] >=0 and x[3,0] < 0.5:
            # x[3] = 0.0
            x[4,0] = 0.       # v_y
            x[5,0] = 0.       # yaw_rate
            u[0,0] = 0.       # steer
            if x[3,0] < 0.3:
                x[3,0] = 0.3  # v_x
            print("Avoiding zero denominator")

        if x[3,0] > -0.5 and x[3,0] < 0:
            # x[3] = 0.
            x[4,0] = 0.
            x[5,0] = 0.
            u[0,0] = 0.
            if x[3,0] > -0.3:
                x[3,0] = -0.3
            print("Avoiding zero denominator")

        # States
        yaw =         x[2,0]  # [0] for scalar data
        v_x =         x[3,0]
        v_y =         x[4,0]
        yaw_rate =    x[5,0]

        steer         =       u[0,0]
        accel_track   =       u[1,0]

        # Dynamics model
        # Slip angle [deg]
        # alpha_f = np.rad2deg(-math.atan2( l_f*yaw_rate + v_y,v_x) + steer)
        # alpha_r = np.rad2deg(-math.atan2(-l_r*yaw_rate + v_y,v_x))

        alpha_f = np.rad2deg(-math.atan2( l_f*yaw_rate + v_y,v_x) + steer)
        alpha_r = np.rad2deg(-math.atan2(-l_r*yaw_rate + v_y,v_x))

        # Lateral force (front & rear)
        # Fy_f = D_lat_f * math.sin(C_lat_f * math.atan2(B_lat_f * alpha_f, 1)) # before was atan
        # Fy_r = D_lat_r * math.sin(C_lat_r * math.atan2(B_lat_r * alpha_r, 1)) # before was atan
        Fy_f = D_lat_f * math.sin(C_lat_f * math.atan(B_lat_f * alpha_f))
        Fy_r = D_lat_r * math.sin(C_lat_r * math.atan(B_lat_r * alpha_r))

        R_roll = C_roll * m * 9.81 * np.sign(v_x)               # rolling resistance. [N] f*(Fzf+Fzr) = f*(mg)
        F_aero = 0.5*roh*C_d*A_f*v_x**2 * np.sign(v_x)          # aero dynamics drag. [N] 0.5*rho*cd*A.

        # Fx_f = (Cm1 * D-Cm2 * D * v_x) - R_roll - F_aero
        Fx_f = m * accel_track - R_roll - F_aero

        x_dot = np.array([[v_x*math.cos(yaw) - v_y*math.sin(yaw)],
                            [v_y*math.cos(yaw) + v_x*math.sin(yaw)],
                            [yaw_rate],
                            [1./m*(Fx_f - Fy_f*math.sin(steer) + m*v_y*yaw_rate)],
                            [1./m*(Fy_r + Fy_f*math.cos(steer) - m*v_x*yaw_rate)],
                            [1./Iz*(Fy_f*l_f*math.cos(steer)- Fy_r*l_r)]])

        x_next = x + x_dot * dt

        return x_next, alpha_f, alpha_r

class Vehicle_Kinematics(object):
    def __init__(self, l_f=1.25, l_r=1.40, dt = 0.02):
        self.l_f = l_f
        self.l_r = l_r
        self.wheelbase = l_f + l_r
        self.dt = dt

    def get_kinematics_model(self, x, u):
        """
        States  : [x; y; v; yaw]
        Actions : [steer; accel]
        """
        nx = 4
        nu = 2

        A = np.zeros((nx, nx))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.dt * math.cos(x[3])
        A[0, 3] = - self.dt * x[2] * math.sin(x[3])
        A[1, 2] = self.dt * math.sin(x[3])
        A[1, 3] = self.dt * x[2] * math.cos(x[3])
        A[3, 2] = self.dt * math.tan(u[0]) / self.wheelbase

        B = np.zeros((nx, nu))
        B[2, 1] = self.dt
        B[3, 0] = self.dt * x[2] / (self.wheelbase * math.cos(u[0]) ** 2)

        C = np.zeros((nx,1))
        C[0] = self.dt * x[2] * math.sin(x[3]) * x[3]
        C[1] = - self.dt * x[2] * math.cos(x[3]) * x[3]
        C[3] = - self.dt * x[2] * u[0] / (self.wheelbase * math.cos(u[0]) ** 2)

        return A, B, C


    def update_kinematics_model(self, x, u):
        """
        Update Kinematic Model.
            States  : [x; y; v; yaw]
            Actions : [steer; accel]

            x_next   = x   + v * math.cos(state.yaw) * dt
            y_next   = y   + v * math.sin(state.yaw) * dt
            v_next   = v   + a * dt
            yaw_next = yaw + v / wheelbase * math.tan(steer) * dt
        """
        x[0] = x[0] + x[2] * math.cos(x[3]) * self.dt
        x[1] = x[1] + x[2] * math.sin(x[3]) * self.dt
        x[2] = x[2] + u[1] * self.dt
        x[3] = x[3] + x[2] / self.wheelbase * math.tan(u[0]) * self.dt
        

        return x


def main():
    # Simulation

    # Initial state
    x0 = np.array([[ 0.],
                  [ 0.],
                  [ np.deg2rad(0)],
                  [ 5.0],
                  [ 0.],
                  [ 0.]])  #  [X; Y; Yaw; vel_x; vel_y; Yaw_rate]

    u0 = np.array([[0*math.pi/180],
                   [0.0]]) #  [steer; traction_accel]

    # Reference state
    xr = np.array([[ 0.],
                   [ 0.],
                   [ np.deg2rad(0)],
                   [ 25.0],
                   [ 0.],
                   [ 0.]])  #  [X; Y; Yaw; vel_x; vel_y; Yaw_rate]

    
    # num of state, action
    nx = 6
    nu = 2

    # Vehicle parameters
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

    sim_time = 500  # time. [sec]

    XX = np.zeros([nx, sim_time])
    XX_front_axle = np.zeros([2, sim_time])
    XX_rear_axle = np.zeros([2, sim_time])

    XX_alpha = np.zeros([2, sim_time]) # front, rear

    UU = np.zeros([nu, sim_time])
    TT = np.linspace(0, sim_time*dt, sim_time)

    vehicle = Vehicle_Dynamics(m=m, l_f=l_f, l_r=l_r, width = width, length = length,
                                turning_circle=turning_circle,
                                C_d = C_d, A_f = A_f, C_roll = C_roll, dt = dt)

    for i in range(sim_time):
        print("===================== nsim :", i, "=====================")

        # timestamp
        tic = time.time()

        if i >= 0.1*sim_time:
            xr = np.array([[ 0.],
                            [ 0.],
                            [ np.deg2rad(10)],
                            [ 15.0],
                            [ 0.],
                            [ 0.]])  #  [X; Y; Yaw; vel_x; vel_y; Yaw_rate]
        if i >= 0.5*sim_time:
            xr = np.array([[ 0.],
                            [ 0.],
                            [ np.deg2rad(100)],
                            [ 15.0],
                            [ 0.],
                            [ 0.]])  #  [X; Y; Yaw; vel_x; vel_y; Yaw_rate]

        # u[0] += np.deg2rad(0.1)
        # if u[0] >= np.deg2rad(5):
        #     u[0] = np.deg2rad(5)
        
        # ===== PID Control ===== #
        p_steer = 5.0
        error_yaw = normalize_angle(xr[2,0] - x0[2,0])
        
        # Steer control
        u0[0,0] = p_steer * error_yaw
        if u0[0,0] >= np.deg2rad(15):
            u0[0,0] = np.deg2rad(15)
        if u0[0,0] <= -np.deg2rad(15):
            u0[0,0] = -np.deg2rad(15)

        # Speed control
        p_accel = 2.0
        error_vx = xr[3,0] - x0[3,0]
        u0[1,0] = p_accel * error_vx
        if u0[1,0] >= 1:
            u0[1,0] = 1
        if u0[1,0] <= -3:
            u0[1,0] = -3


        XX[:,i] = x0.T   # list for plot.
        TT[i] = dt * i
        UU[:,i] = u0.T

        # normalize angle (Better not to normalize.)
        # x_k1[2] = normalize_angle(x_k1[2])
        # print("Normalized in loop. x_k1[2] :", x_k1[2])

        # State Prediction
        N = 50
        pred_x = np.zeros((nx, N+1))
        pred_x[:,0] = x0.T

        x = x0

        for ii in range(0, N):
            # x_next, _, _ = vehicle.update_rear_wheel_dynamics_model(x, u0)
            x_next, _, _ = vehicle.update_dynamics_model(x, u0)
            pred_x[:,ii+1] = x_next.T
            x = x_next
        
        # print
        print("x :", x0[0,0], "y :", x0[1,0], "yaw :", x0[2,0], "Vx :", x0[3,0], "Vy :", x0[4,0], "yawRate :", x0[5,0])
        print("    ------------------------------------------------------------------------------    ")
        print("steer :", u0[0,0], "accel :", u0[1,0])

        # print("Error yaw :", np.rad2deg(error_yaw))

        plt.cla()
        plt.plot(XX[0,:i+1], XX[1,:i+1], "-b", label="Driven Path")
        plt.grid(True)
        plt.axis("equal")
        plot_car(x0[0,0], x0[1,0], x0[2,0], steer=u0[0,0])
        plt.plot(pred_x[0,:], pred_x[1,:], "r")
        plt.pause(0.0001)

        # x1, alpha_f, alpha_r = vehicle.update_rear_wheel_dynamics_model(x0, u0)
        x1, alpha_f, alpha_r = vehicle.update_dynamics_model(x0, u0)
        XX_alpha[0,i] = alpha_f
        XX_alpha[1,i] = alpha_r
        x0 = x1 # update t+1 state

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

if __name__ == "__main__":
    main()