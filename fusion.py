import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from filterpy.kalman import ExtendedKalmanFilter
import utility_functions as uf
import ekf_models as model

# ------------------- Data Preprocessing-----------------------

mat = loadmat('imu_gps_log.mat')

dataLog = mat['dataLog']     # shape: (N, 12)
timeLog = mat['timeLog'].squeeze()

ax = dataLog[:, 0] * 9.81
ay = dataLog[:, 1] * 9.81
az = dataLog[:, 2] * 9.81

gx = np.deg2rad(dataLog[:, 3])
gy = np.deg2rad(dataLog[:, 4])
gz = np.deg2rad(dataLog[:, 5])

gps_lat = dataLog[:, 6]
gps_lon = dataLog[:, 7]
gps_alt = dataLog[:, 8]

gps_fix = dataLog[:, 9].astype(bool)

gps_vx = dataLog[:, 10]
gps_vy = dataLog[:, 11]

N = len(timeLog)


# Convert GPS LLA --> local NED (flat-earth)

R_E = 6378137.0

lat0 = gps_lat[gps_fix][0]
lon0 = gps_lon[gps_fix][0]
alt0 = gps_alt[gps_fix][0]

lat = gps_lat
lon = gps_lon

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

gps_pos = np.zeros((N, 3))
gps_pos[:, 0] = (lat - lat0) * deg2rad * R_E                                # North
gps_pos[:, 1] = (lon - lon0) * deg2rad * R_E * np.cos(lat0 * deg2rad)       # East
gps_pos[:, 2] = -(gps_alt - alt0)                                           # Down

gps_vel = np.zeros((N, 3))
gps_vel[:, 0] = gps_vx
gps_vel[:, 1] = gps_vy
gps_vel[:, 2] = 0.0

imu = np.column_stack((gx, gy, gz, ax, ay, az))



#------------------- EKF initialization -----------------------

ekf = ExtendedKalmanFilter(dim_x=16, dim_z=6)

acc_bias, gyro_bias = uf.remove_imu_bias(dataLog)

ekf.x = np.zeros(16)
ekf.x[6] = 1.0
ekf.x[10:13] = acc_bias
ekf.x[13:16] = gyro_bias


ekf.P = np.diag([
    3, 3, 3,          # Position
    1,1,1,            # Velocity
    0.1,0.1,0.1,0.1,  # Attitude Quaternion
    0.01,0.01,0.01,   # Accel Bias
    0.001,0.001,0.001 # Gyro Bias
])
ekf.Q = np.eye(16) * 1e-3
ekf.R = np.diag([5,5,5, 0.5,0.5,0.5])


est_pos = np.zeros((N, 3))
ms_to_knots = 1.94384


# Start from first GPS fix, otherwise intial estimate is poor
first_fix_idx = np.where(gps_fix == True)[0][0]

ekf.x[0:3] = gps_pos[first_fix_idx]

print("--- Starting NMEA Stream ---")

# ------------------- Run EKF --------------------
for k in range(first_fix_idx + 1, N):
    if k == 0:
        dt = 0.01
    else:
        dt = timeLog[k] - timeLog[k-1]

    if dt > 1.0: dt = 0.01

    # Update State 
    ekf.x = model.fx(ekf.x, imu[k], dt)

    # Update Jacobian 
    ekf.F = model.compute_F(ekf.x, imu[k], dt, acc_bias)
    
    # Predict Step
    ekf.predict()

    # NMEA String Generation
    vx, vy = gps_vel[k, 0], gps_vel[k, 1]
    speed_ms = np.sqrt(vx**2 + vy**2)
    speed_knots = speed_ms * ms_to_knots
    course = np.rad2deg(np.arctan2(vy, vx)) % 360

    nmea_string = uf.generate_nmea_gprmc
    print(timeLog[k]*1e-6, nmea_string) 

    # Update Step (GPS)
    if gps_fix[k]:
        z = np.hstack((gps_pos[k], gps_vel[k]))
        ekf.update(z, HJacobian=model.H_jacobian, Hx=model.hx)

    est_pos[k] = ekf.x[0:3]


# Convert estimated NED --> Lat/Lon

est_lat = lat0 + rad2deg * est_pos[:,0] / R_E
est_lon = lon0 + rad2deg * est_pos[:,1] / (R_E * np.cos(deg2rad * lat0))

rmse_val = uf.calculate_position_rmse(est_lat, est_lon, gps_lat, gps_lon, first_fix_idx)
print(f"Position RMSE: {rmse_val:.2f} meters")



# ------------------- Geoplot trajectory plot -----------------------------

plt.figure(figsize=(8, 8))
plt.plot(gps_lon[gps_fix], gps_lat[gps_fix], 'r.', label='GPS')
plt.plot(est_lon, est_lat, 'b-', label='EKF')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('GPS vs EKF Estimated Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
