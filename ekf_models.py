import numpy as np
import utility_functions as uf


# EKF process & measurement models

# Process function to propagate state
def fx(x, u, dt):
    x = x.copy()

    p = x[0:3]
    v = x[3:6]
    q = x[6:10]
    ba = x[10:13]
    bg = x[13:16]

    omega = u[0:3] - bg
    acc_b = u[3:6] - ba

    wx, wy, wz = omega
    Omega = np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])

    q = uf.normalize_quat(q + 0.5 * Omega @ q * dt)

    Cbn = uf.quat_to_dcm(q)
    g_n = np.array([0.0, 0.0, 9.81])
    acc_n = Cbn @ acc_b

    v = v + (acc_n + g_n) * dt
    p = p + v * dt

    x[0:3] = p
    x[3:6] = v
    x[6:10] = q

    return x

# Measurement function to extract Position and Velocity
def hx(imu):
    return np.hstack((imu[0:3], imu[3:6]))

def H_jacobian(x):
    H = np.zeros((6, 16))
    H[0:3, 0:3] = np.eye(3)
    H[3:6, 3:6] = np.eye(3)
    return H

# State Jacobian of the process model
def compute_F_2(x, imu, dt, acc_bias_static): 
    F = np.eye(16)
    
    q = x[6:10]
    q0, q1, q2, q3 = q
    
    # Current estimated bias
    current_acc_bias = x[10:13]
    
    # Acceleration in body frame (measured - bias)
    ax_b, ay_b, az_b = imu[3:6] - current_acc_bias
    
    # 1. Position depends on Velocity
    F[0:3, 3:6] = np.eye(3) * dt
    
    # 2. Velocity depends on Attitude (Quaternion) - 3x4 Matrix
    # We need d(Cbn * a_b) / dq
    # This block is derived from the partial derivative of the rotation matrix
    
    # Row 1 (Velocity X)
    F[3, 6] = 2 * (q2*az_b - q3*ay_b) * dt
    F[3, 7] = 2 * (q2*ay_b + q3*az_b) * dt
    F[3, 8] = 2 * (-2*q2*ax_b + q1*ay_b + q0*az_b) * dt
    F[3, 9] = 2 * (-2*q3*ax_b - q0*ay_b + q1*az_b) * dt

    # Row 2 (Velocity Y)
    F[4, 6] = 2 * (-q1*az_b + q3*ax_b) * dt
    F[4, 7] = 2 * (q1*ax_b - 2*q2*ay_b - q0*az_b) * dt
    F[4, 8] = 2 * (q0*ax_b + q1*ay_b) * dt
    F[4, 9] = 2 * (q0*ax_b - 2*q3*ay_b + q2*az_b) * dt

    # Row 3 (Velocity Z)    
    F[5, 6] = 2 * (q1*ay_b - q2*ax_b) * dt
    # Simplified: 2*(q0*ay + q3*ax - 2*q1*az)
    F[5, 7] = 2 * (q0*ay_b + q3*ax_b - 2*q1*az_b) * dt
    F[5, 8] = 2 * (-q0*ax_b + q3*ay_b - 2*q2*az_b) * dt
    F[5, 9] = 2 * (q1*ax_b + q2*ay_b) * dt

    # 3. Velocity depends on Accel Bias (Correct in your original code)
    Cbn = uf.quat_to_dcm(q)
    F[3:6, 10:13] = -Cbn * dt
    
    # 4. Attitude depends on Gyro Bias (Correct in your original code)
    Xi = uf.quat_kinematic_matrix(q)
    F[6:10, 13:16] = -Xi * dt
  
    return F