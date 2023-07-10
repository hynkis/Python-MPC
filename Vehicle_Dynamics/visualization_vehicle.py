import matplotlib.pyplot as plt
import numpy as np
import math

# Vehicle parameters
# I30 based.

OVERHANG_FRONT = 0.9   # [m]
OVERHANG_REAR = 0.8    # [m]
WHEELBASE = 2.60       # [m]

WHEEL_LEN = 0.3        # [m]
WHEEL_WIDTH = 0.185    # [m]
TRACK = 1.50           # [m] length btw wheels.

WIDTH = 1.9            # [m]
LENGTH = OVERHANG_FRONT + OVERHANG_REAR + WHEELBASE  # 4.3, [m]

TURNING_CIRCLE = 5.3   # [m], min radius. 10.4, 5.3

# i30 parameter etc
Steering_Gear_Ratio	= 13.4

MAX_STEER = math.atan(WHEELBASE/TURNING_CIRCLE)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)                    # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6                           # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6                          # minimum speed [m/s]
MAX_ACCEL = 1.0                                  # maximum accel [m/ss]

def plot_car(x, y, yaw, steer, cabcolor='-r', truckcolor='-k'):

    outline = np.array([[-OVERHANG_REAR, (LENGTH - OVERHANG_REAR), (LENGTH - OVERHANG_REAR), -OVERHANG_REAR, -OVERHANG_REAR],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])
    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TRACK/2, -WHEEL_WIDTH - TRACK/2, WHEEL_WIDTH - TRACK/2, WHEEL_WIDTH - TRACK/2, -WHEEL_WIDTH - TRACK/2]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WHEELBASE
    fl_wheel[0, :] += WHEELBASE

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def plot_car_force(x, y, yaw, steer, Fx, cabcolor='-r', truckcolor='-k'):

    outline = np.array([[-OVERHANG_REAR, (LENGTH - OVERHANG_REAR), (LENGTH - OVERHANG_REAR), -OVERHANG_REAR, -OVERHANG_REAR],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])
    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TRACK/2, -WHEEL_WIDTH - TRACK/2, WHEEL_WIDTH - TRACK/2, WHEEL_WIDTH - TRACK/2, -WHEEL_WIDTH - TRACK/2]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WHEELBASE
    fl_wheel[0, :] += WHEELBASE

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

    # Plot Force
    plt.arrow(np.squeeze(x + WHEELBASE*math.cos(yaw)), np.squeeze(y + WHEELBASE*math.sin(yaw)), 0.005*Fx*math.cos(steer+yaw), 0.005*Fx*math.sin(steer+yaw), head_width=0.05, head_length=0.05, fc='g', ec='g')


def main():
    x = 0.0
    y = 0.0
    yaw = np.deg2rad(0)
    steer = np.deg2rad(0)

    print("Max steer :", np.rad2deg(MAX_STEER))

    plot_car(x, y, yaw, MAX_STEER)
    plt.axis('square')
    plt.show()

if __name__ == "__main__":
    main()