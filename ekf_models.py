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
def compute_F(x, imu, dt, acc_bias):
    F = np.eye(16)
    q = x[6:10]
    acc_bias = x[10:13]
    acc_b = imu[3:6] - acc_bias  
    
    # 1. Position depends on Velocity
    F[0:3, 3:6] = np.eye(3) * dt
    
    # 2. Velocity depends on Attitude (The "Tilt-to-Accel" coupling)
    # We use a skew-symmetric matrix of the rotated acceleration
    Cbn = uf.quat_to_dcm(q)
    acc_n = Cbn @ acc_b
    # Skew-symmetric matrix of acc_n helps map orientation errors to velocity errors
    S = np.array([[0, -acc_n[2], acc_n[1]],
                  [acc_n[2], 0, -acc_n[0]],
                  [-acc_n[1], acc_n[0], 0]])
    
    # This maps the 3-degree-of-freedom orientation error into velocity
    F[3:6, 6:9] = -S * dt 
    
    # 3. Velocity depends on Accel Bias
    F[3:6, 10:13] = -Cbn * dt
    
    # 4. Attitude depends on Gyro Bias
    Xi = uf.quat_kinematic_matrix(q)
    F[6:10, 13:16] = -Xi * dt
  
    return F