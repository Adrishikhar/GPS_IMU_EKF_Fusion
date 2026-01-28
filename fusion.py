import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from filterpy.kalman import ExtendedKalmanFilter

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
gps_pos = np.zeros((N, 3))
gps_pos[:, 0] = (lat - lat0) * deg2rad * R_E                                # North
gps_pos[:, 1] = (lon - lon0) * deg2rad * R_E * np.cos(lat0 * deg2rad)       # East
gps_pos[:, 2] = -(gps_alt - alt0)                                           # Down

gps_vel = np.zeros((N, 3))
gps_vel[:, 0] = gps_vx
gps_vel[:, 1] = gps_vy
gps_vel[:, 2] = 0.0

imu = np.column_stack((gx, gy, gz, ax, ay, az))



# Utility functions

def normalize_quat(q):
    return q / np.linalg.norm(q)

def quat_to_dcm(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3),   1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   1-2*(q1*q1+q2*q2)]
    ])

import datetime

def generate_nmea_gprmc(timestamp, lat, lon, speed_knots, course_deg):
    """
    Generates a $GPRMC string
    """
    lat_safe = lat if not np.isnan(lat) else 0.0
    lon_safe = lon if not np.isnan(lon) else 0.0
    speed_safe = speed_knots if not np.isnan(speed_knots) else 0.0
    course_safe = course_deg if not np.isnan(course_deg) else 0.0
    ts_safe = timestamp if not np.isnan(timestamp) else 0.0

    def dec_to_nmea(value, is_lat=True):
        abs_val = abs(value)
        degrees = int(abs_val)
        minutes = (abs_val - degrees) * 60
        if is_lat:
            direction = "N" if value >= 0 else "S"
            return f"{degrees:02d}{minutes:07.4f},{direction}"
        else:
            direction = "E" if value >= 0 else "W"
            return f"{degrees:03d}{minutes:07.4f},{direction}"

    try:
        dt_obj = datetime.datetime.fromtimestamp(ts_safe)
        time_str = dt_obj.strftime('%H%M%S')
        date_str = dt_obj.strftime('%d%m%y')
    except (ValueError, OSError):
        time_str = "000000"
        date_str = "010100"
    
    lat_nmea = dec_to_nmea(lat_safe, is_lat=True)
    lon_nmea = dec_to_nmea(lon_safe, is_lat=False)
    
    # Construct message: Status 'A' for valid, 'V' (void) if data was NaN
    status = "A" if not np.isnan(lat) else "V"
    
    msg = f"GPRMC,{time_str},{status},{lat_nmea},{lon_nmea},{speed_safe:.2f},{course_safe:.2f},{date_str},,,A"
    
    checksum = 0
    for char in msg:
        checksum ^= ord(char)
    
    return f"${msg}*{checksum:02X}"


def generate_nmea_gprmc(timestamp, lat, lon, speed_knots, course_deg):
    """
    Generates a $GPRMC string with NaN protection.
    """
    # Replace NaN with 0.0
    lat_safe = lat if not np.isnan(lat) else 0.0
    lon_safe = lon if not np.isnan(lon) else 0.0
    speed_safe = speed_knots if not np.isnan(speed_knots) else 0.0
    course_safe = course_deg if not np.isnan(course_deg) else 0.0
    ts_safe = timestamp if not np.isnan(timestamp) else 0.0

    def dec_to_nmea(value, is_lat=True):
        abs_val = abs(value)
        degrees = int(abs_val)
        minutes = (abs_val - degrees) * 60
        if is_lat:
            direction = "N" if value >= 0 else "S"
            return f"{degrees:02d}{minutes:07.4f},{direction}"
        else:
            direction = "E" if value >= 0 else "W"
            return f"{degrees:03d}{minutes:07.4f},{direction}"

    try:
        dt_obj = datetime.datetime.fromtimestamp(ts_safe)
        time_str = dt_obj.strftime('%H%M%S')
        date_str = dt_obj.strftime('%d%m%y')
    except (ValueError, OSError):
        time_str = "000000000"
        date_str = "010100"
    
    lat_nmea = dec_to_nmea(lat_safe, is_lat=True)
    lon_nmea = dec_to_nmea(lon_safe, is_lat=False)
    
    # Construct message: Status 'A' for valid, 'V' (void) if data was NaN
    status = "A" if not np.isnan(lat) else "V"
    
    msg = f"GPRMC,{time_str},{status},{lat_nmea},{lon_nmea},{speed_safe:.2f},{course_safe:.2f},{date_str},,,A"
    
    checksum = 0
    for char in msg:
        checksum ^= ord(char)
    
    return f"${msg}*{checksum:02X}"


# EKF process & measurement models

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

    q = normalize_quat(q + 0.5 * Omega @ q * dt)

    Cbn = quat_to_dcm(q)
    g_n = np.array([0.0, 0.0, 9.81])
    acc_n = Cbn @ acc_b

    v = v + (acc_n + g_n) * dt
    p = p + v * dt

    x[0:3] = p
    x[3:6] = v
    x[6:10] = q

    return x

def hx(x):
    return np.hstack((x[0:3], x[3:6]))

def H_jacobian(x):
    H = np.zeros((6, 16))
    H[0:3, 0:3] = np.eye(3)
    H[3:6, 3:6] = np.eye(3)
    return H


# EKF initialization

ekf = ExtendedKalmanFilter(dim_x=16, dim_z=6)

ekf.x = np.zeros(16)
ekf.x[6] = 1.0

ekf.P = np.diag([
    10,10,10,
    1,1,1,
    0.1,0.1,0.1,0.1,
    0.01,0.01,0.01,
    0.001,0.001,0.001
])

ekf.Q = np.eye(16) * 1e-3
ekf.R = np.diag([5,5,5, 0.5,0.5,0.5])


# Jacobian function to connect Position and Velocity
def compute_F(x, dt):
    F = np.eye(16)

    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    
    return F


est_pos = np.zeros((N, 3))
ms_to_knots = 1.94384

print("--- Starting NMEA Stream ---")

# ------------------- Run EKF --------------------
for k in range(N):
    if k == 0:
        dt = 0.01
    else:
        dt = timeLog[k] - timeLog[k-1]

    if dt > 1.0: dt = 0.01

    # Update State 
    ekf.x = fx(ekf.x, imu[k], dt)
    
    # Update Jacobian 
    ekf.F = compute_F(ekf.x, dt)
    
    # Predict Step
    ekf.predict()

    # Calculate Speed and Course
    vx, vy = gps_vel[k, 0], gps_vel[k, 1]
    speed_ms = np.sqrt(vx**2 + vy**2)
    speed_knots = speed_ms * ms_to_knots
    course = np.rad2deg(np.arctan2(vy, vx)) % 360

    # Generate the string
    nmea_string = generate_nmea_gprmc(timeLog[k]*1e-6, gps_lat[k], gps_lon[k], speed_knots, course)
    print(nmea_string, flush=True)

    # 4. Update Step (GPS)
    if gps_fix[k]:
        z = np.hstack((gps_pos[k], gps_vel[k]))
        ekf.update(z, HJacobian=H_jacobian, Hx=hx)

    est_pos[k] = ekf.x[0:3]


# Convert estimated NED --> Lat/Lon
rad2deg = 180.0 / np.pi

est_lat = lat0 + rad2deg * est_pos[:,0] / R_E
est_lon = lon0 + rad2deg * est_pos[:,1] / (R_E * np.cos(deg2rad * lat0))



# Geoplot trajectory plot

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
