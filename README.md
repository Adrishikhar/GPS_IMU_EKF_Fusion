# EKF GPS-IMU Fusion

A high-performance **16-state Extended Kalman Filter (EKF)** implementation for sensor fusion. This system integrates high-frequency Inertial Measurement Unit (IMU) data with absolute GPS positioning to provide a robust, smoothed navigation solution.

---

## 🛰️ State Vector Definition

The filter maintains a 16-element state vector $x$:

| Component      | Description              | Frame                       |
| :------------- | :----------------------- | :-------------------------- |
| **Position**   | $p_n, p_e, p_d$          | Local NED (North-East-Down) |
| **Velocity**   | $v_n, v_e, v_d$          | Navigation Frame            |
| **Attitude**   | $q_0, q_1, q_2, q_3$     | Unit Quaternion             |
| **Accel Bias** | $b_{ax}, b_{ay}, b_{az}$ | Body Frame                  |
| **Gyro Bias**  | $b_{gx}, b_{gy}, b_{gz}$ | Body Frame                  |

---

## 📂 Project Structure

| File                       | Description                                                                                                                  |
| :------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **`fusion.py`**            | Main entry point. Manages the EKF loop, data synchronization, initialization from first GPS fix, and performance plotting.   |
| **`ekf_models.py`**        | Core EKF math: Nonlinear state transition ($f$), measurement model ($h$), and the full Jacobians ($F, H$).                   |
| **`utility_functions.py`** | Helper functions for quaternion algebra, coordinate projections (LLA to NED), NMEA string generation, and RMSE calculations. |
| **`imu_gps_log.mat`**      | Raw sensor dataset containing synchronized IMU and GPS logs.                                                                 |
| **`requirements.txt`**     | Dependency list (NumPy, SciPy, Matplotlib, FilterPy).                                                                        |

## 📐 Mathematical Framework

### 1. Prediction (IMU Integration)

The state propagates using the basic Euler integration method.

### 2. Correction (GPS Update)

When a GPS fix is detected, the filter performs a measurement update:

- **Measurement $z$:** $[p_n, p_e, p_d, v_n, v_e, v_d]^T$
- **Innovation:** $y = z - h(x)$
- **Update:** The EKF uses the 6D measurement to correct the 16D state via the Kalman Gain.

---

## 🚀 Getting Started

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt

   ```

2. **Run File**

```bash
 python fusion.py

```
