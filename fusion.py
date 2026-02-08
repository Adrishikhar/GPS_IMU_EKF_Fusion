import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from filterpy.kalman import ExtendedKalmanFilter
import utility_functions as uf
import ekf_models as model
import pynmea2

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
#gps_alt = dataLog[:, 8]
gps_alt = 10.0 * np.ones_like(gps_lat)  # Placeholder altitude

gps_fix = dataLog[:, 9].astype(bool)

gps_vx = dataLog[:, 10]
gps_vy = dataLog[:, 11]

N = len(timeLog)


# Convert GPS LLA --> local NED (flat-earth)

R_E = 6378137.0

lat0 = gps_lat[gps_fix][0]
lon0 = gps_lon[gps_fix][0]
# alt0 = gps_alt[gps_fix][0]
# Altitude data takes time to reach, so assuming a constant for a basic filter, otherwise a dedicated GPS handler is needed
alt0 = 10.0

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

# Start from first GPS fix, otherwise intial estimate is poor
first_fix_idx = np.where(gps_fix == True)[0][0]

acc_bias, gyro_bias = uf.remove_imu_bias(dataLog)

ax0, ay0, az0 = np.mean(imu[0:20, 0:3], axis=0)
initial_roll = np.arctan2(ay0, az0)
initial_pitch = np.arctan2(-ax0, np.sqrt(ay0**2 + az0**2))

# Calculate Yaw from GPS Velocity (Course)
vx0 = gps_vel[first_fix_idx, 0]
vy0 = gps_vel[first_fix_idx, 1]
initial_yaw = np.arctan2(vy0, vx0)
print(f"Initial Yaw (deg): {np.rad2deg(initial_yaw):.2f}")

q_init = uf.euler_to_quat(initial_roll, initial_pitch, initial_yaw)


ekf.x = np.zeros(16)
ekf.x[0:3] = gps_pos[first_fix_idx]
ekf.x[6] = 1.0
ekf.x[6:10] = q_init
ekf.x[10:13] = acc_bias
ekf.x[13:16] = gyro_bias


ekf.P = np.diag([
    2, 2, 2,          # Position
    1,1,1,            # Velocity
    0.1,0.1,0.1,0.1,  # Attitude Quaternion
    0.01,0.01,0.01,   # Accel Bias
    0.001,0.001,0.001 # Gyro Bias
])
ekf.Q = np.diag([
    0.1**2, 0.1**2, 0.1**2,               # x, y, z
    0.2**2, 0.2**2, 0.2**2,               # vx, vy, vz
    0.01**2, 0.01**2, 0.01**2,0.01**2,    # q0, q1, q2, q3
    1e-6**2, 1e-6**2, 1e-6**2,            # Accel bias
    1e-10**2, 1e-10**2, 1e-10**2          # Gyro bias
])
ekf.R = np.diag([
    2**2, 2**2, 100.0,  # X, Y, Z (High variance on Z allows it to float)
    2, 2, 2   # VX, VY, VZ
])


est_pos = np.zeros((N, 3))
ms_to_knots = 1.94384

# --------------------------------------------------------------

print("--- Starting NMEA Stream ---")
# check = []

est_lat = np.full(N, lat0)
est_lon = np.full(N, lon0)
last_gps_pos = np.zeros(3)

# ------------------- Run EKF --------------------
for k in range(first_fix_idx + 1, N):
    if k == 0:
        dt = 0.01
    else:
        dt = timeLog[k] - timeLog[k-1]

    if dt > 1.0: dt = 0.01

    # Update State 
    ekf.x = model.fx(ekf.x, imu[k], dt)

    # Update State Jacobian Matrix 
    ekf.F = model.compute_F_2(ekf.x, imu[k], dt, acc_bias)
    
    # Predict Covariance
    ekf.P = ekf.F @ ekf.P @ ekf.F.T + ekf.Q

    # Normalize quaternion
    ekf.x[6:10] = uf.normalize_quat(ekf.x[6:10])

    # NMEA String Generation
    vx, vy = gps_vel[k, 0], gps_vel[k, 1]
    speed_ms = np.sqrt(vx**2 + vy**2)
    speed_knots = speed_ms * ms_to_knots
    course = np.rad2deg(np.arctan2(vy, vx)) % 360

    est_lat[k] = lat0 + rad2deg * ekf.x[0] / R_E
    est_lon[k] = lon0 + rad2deg * ekf.x[1] / (R_E * np.cos(deg2rad * lat0))

    nmea_string = uf.generate_nmea_gpgga(timeLog[k]*1e-6, est_lat[k], est_lon[k])
    # try:
    #     msg = pynmea2.parse(nmea_string)
        
    #     check.append({
    #         'lat': msg.latitude, 
    #         'lon': msg.longitude
    #     })
    #     print(msg)
    # except pynmea2.ParseError as e:
    #     print(f"Parse error: {e}")
    #     continue

    # Update Step (GPS)

    current_gps_pos = gps_pos[k]
    is_new_measurement = not np.array_equal(current_gps_pos, last_gps_pos)

    if gps_fix[k] and is_new_measurement:
        # Measurement vector
        z = np.hstack((gps_pos[k], gps_vel[k]))

        ekf.update(z, HJacobian=model.H_jacobian, Hx=model.hx)

        last_gps_pos = current_gps_pos
        print(f"GPS Update at k={k}: {z[0:3]}")

    est_pos[k] = ekf.x[0:3]


# Convert estimated NED --> Lat/Lon

rmse_val = uf.calculate_position_rmse(est_lat, est_lon, gps_lat, gps_lon, first_fix_idx)
print(f"Position RMSE: {rmse_val:.2f} meters")



# ------------------- Geoplot trajectory plot -----------------------------
# Quick 2D plot without map background

# plt.figure(figsize=(8, 8))
# plt.plot(gps_lon[gps_fix], gps_lat[gps_fix], 'r.', label='GPS')
# plt.plot(est_lon, est_lat, 'b-', label='EKF')
# plt.xlabel('Longitude (deg)')
# plt.ylabel('Latitude (deg)')
# plt.title('GPS vs EKF Estimated Trajectory')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.show()

# ------------------- Plotly map plot -----------------------------
# Takes roughly 60-90 seconds to load the map tiles
import plotly.graph_objects as go

fig = go.Figure()

# 1. Plot GPS Data (Red Markers)
fig.add_trace(go.Scattermap(
    lat=gps_lat[gps_fix],
    lon=gps_lon[gps_fix],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='GPS'
))

# 2. Plot EKF Estimate (Blue Line)
fig.add_trace(go.Scattermap(
    lat=est_lat,
    lon=est_lon,
    # lat = [data['lat'] for data in check],
    # lon = [data['lon'] for data in check],
    mode='lines',
    line=dict(width=3, color='blue'),
    name='EKF'
))

# 3. Configure the Map Layout
fig.update_layout(
    map_style="open-street-map",  # Try "satellite" to check imagery alignment
    map=dict(
        center=dict(lat=gps_lat.mean(), lon=gps_lon.mean()),
        zoom=18
    ),
    margin={"r":0,"t":40,"l":0,"b":0},
    title="GPS vs EKF Estimated Trajectory"
)

fig.show()

# ------------------- Contextily map plot -----------------------------
# Takes roughly 2-3 minutes to load the map tiles
# import contextily as cx

# # 1. Create the plot
# fig, ax = plt.subplots(figsize=(10, 10))

# # 2. Plot data (Note: ax.plot still uses lon, lat order)
# ax.plot(gps_lon[gps_fix], gps_lat[gps_fix], 'r.', markersize=4, label='GPS', alpha=0.5)
# ax.plot(est_lon, est_lat, 'b-', linewidth=2, label='EKF')

# # 3. Add the basemap
# # crs="EPSG:4326" tells contextily your data is in standard Lat/Lon
# cx.add_basemap(ax, crs="EPSG:4326", zoom=19, source=cx.providers.OpenStreetMap.Mapnik)

# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('GPS vs EKF Trajectory on Map')
# ax.legend()
# plt.show()