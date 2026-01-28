# EKF GPS-IMU Fusion

A high-performance **16-state Extended Kalman Filter (EKF)** implementation for sensor fusion. This system integrates high-frequency Inertial Measurement Unit (IMU) data with absolute GPS positioning to provide a robust, smoothed navigation solution.

---

## State Vector Definition

The filter maintains a 16-element state vector $x$:

| Component      | Description              | Frame                       |
| :------------- | :----------------------- | :-------------------------- |
| **Position**   | $p_n, p_e, p_d$          | Local NED (North-East-Down) |
| **Velocity**   | $v_n, v_e, v_d$          | Navigation Frame            |
| **Attitude**   | $q_0, q_1, q_2, q_3$     | Unit Quaternion             |
| **Accel Bias** | $b_{ax}, b_{ay}, b_{az}$ | Body Frame                  |
| **Gyro Bias**  | $b_{gx}, b_{gy}, b_{gz}$ | Body Frame                  |

---

## Core Features

- **Nonlinear Kinematics:** Uses first-order quaternion integration to avoid Euler angle singularities (Gimbal Lock).
- **Coordinate Projection:** Converts Geodetic LLA (Lat, Lon, Alt) to a local NED tangent plane using a Flat-Earth model.
- **Real-time NMEA Stream:** Generates standardized `$GPRMC` strings with automatic checksum calculation and NaN protection.
- **Bias Tracking:** Dynamically estimates sensor null-shifts to improve dead-reckoning accuracy.

---

## Mathematical Framework

### 1. Prediction (IMU Integration)

The state propagates using raw IMU control inputs $u = [\omega, a]$:
$$x_{k+1} = f(x_k, u_k, \Delta t)$$
Gravity is rotated into the navigation frame and compensated for in the velocity update.

### 2. Correction (GPS Update)

When a GPS fix is detected, the filter performs a measurement update:

- **Measurement $z$:** $[p_x, p_y, p_z, v_x, v_y, v_z]^T$
- **Jacobian $H$:** Maps the 16D state to the 6D measurement space.
