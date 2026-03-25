import numpy as np
from numpy import sin, cos
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=4, suppress=True)

l1, l2, l3 = 1.0, 0.5, 1.0

def T(q1, q2):

    T0 = np.array(
        (
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, -1, 0, l1),
            (0, 0, 0, 1),
        )
    )

    T1 = np.array(
        ((sin(q1), cos(q1), 0, l2 * sin(q1)),
        (-cos(q1), sin(q1), 0, l2 * (-cos(q1))),
        (0, 0, 1, 0),
        (0, 0, 0, 1),)
    )

    T2 = np.array(
        ((cos(q2), -sin(q2), 0, l3 * cos(q2)),
        (sin(q2), cos(q2), 0, l3 * sin(q2)),
        (0, 0, 1, 0),
        (0, 0, 0, 1),)
    )

    return T0 @ T1 @ T2

def T_test(q1, q2):

    T0 = np.array(
        (
            (1, 0, 0, l1),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )

    T1 = np.array(
        ((cos(q1), -sin(q1), 0, l2 * cos(q1)),
        (sin(q1), cos(q1), 0, l2 * sin(q1)),
        (0, 0, 1, 0),
        (0, 0, 0, 1),)
    )

    T2 = np.array(
        ((cos(q2), -sin(q2), 0, l3 * cos(q2)),
        (sin(q2), cos(q2), 0, l3 * sin(q2)),
        (0, 0, 1, 0),
        (0, 0, 0, 1),)
    )

    return T0 @ T1 @ T2


q1, q2 = 0.0, -np.pi/2.0
T_val = T(q1,q2)

def Rx(theta):
    return np.array(
        ((1, 0, 0, 0),
        (0, cos(theta), -sin(theta), 0),
        (0, sin(theta), cos(theta), 0),
        (0, 0, 0, 1),)
    )

def Rz(theta):
    return np.array(
        ((cos(theta), -sin(theta), 0, 0),
        (sin(theta), cos(theta), 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),)
    )

# Rx = np.array(
#     ((1, 0, 0, 0),
#     (0, cos(-np.pi/2), -sin(-np.pi/2), 0),
#     (0, sin(-np.pi/2), cos(-np.pi/2), 0),
#     (0, 0, 0, 1),)
# )
# Rz = np.array(
#     ((cos(-np.pi/2), -sin(-np.pi/2), 0, 0),
#     (sin(-np.pi/2), cos(-np.pi/2), 0, 0),
#     (0, 0, 1, 0),
#     (0, 0, 0, 1),)
# )


T_t = Rx(-np.pi/2) @ Rz(-np.pi/2) @ T_test(q1,q2)

print(T_val)
print(T_t)

ee_pos = T_val[:3, 3]

print(ee_pos)