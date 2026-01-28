import numpy as np
import datetime

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

def calculate_position_rmse(est_lat, est_lon, meas_lat, meas_lon, gps_first_fix_idx):
    """
    Calculates the RMSE in meters between estimated and measured Lat/Lon.
    """
    # Constants for WGS84 Ellipsoid
    R_E = 6378137.0  # Earth radius in meters

    e_lat = est_lat[gps_first_fix_idx:]
    e_lon = est_lon[gps_first_fix_idx:]
    m_lat = meas_lat[gps_first_fix_idx:]
    m_lon = meas_lon[gps_first_fix_idx:] 

    # Filter out indices where GPS might be NaN or invalid
    mask = ~np.isnan(m_lat) & ~np.isnan(m_lon)
    e_lat, e_lon = e_lat[mask], e_lon[mask]
    m_lat, m_lon = m_lat[mask], m_lon[mask]

    # Convert angular difference to meters (Equirectangular approximation)
    # North error (latitude)
    d_north = (e_lat - m_lat) * deg2rad * R_E
    
    # East error (longitude) - corrected for latitude
    avg_lat = m_lat * deg2rad
    d_east = (e_lon - m_lon) * deg2rad * R_E * np.cos(avg_lat)

    # Calculate Euclidean distance error for each point
    dist_error = np.sqrt(d_north**2 + d_east**2)

    # Calculate Root Mean Square
    rmse = np.sqrt(np.mean(dist_error**2))

    return rmse
  
def normalize_quat(q):
    return q / np.linalg.norm(q)

def quat_kinematic_matrix(q):
    """Returns the 4x3 matrix mapping angular velocity to quaternion derivatives."""
    q0, q1, q2, q3 = q
    return 0.5 * np.array([
        [-q1, -q2, -q3],
        [ q0, -q3,  q2],
        [ q3,  q0, -q1],
        [-q2,  q1,  q0]
    ])

def quat_to_dcm(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3),   1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   1-2*(q1*q1+q2*q2)]
    ])

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

def generate_nmea_gpgga(timestamp, lat, lon, alt=0.0):
    """
    Generates a $GPGGA string with the requested precision and formatting.
    """
    # Replace NaN with 0.0
    lat_safe = lat if not np.isnan(lat) else 0.0
    lon_safe = lon if not np.isnan(lon) else 0.0
    alt_safe = alt if not np.isnan(alt) else 0.0
    ts_safe = timestamp if not np.isnan(timestamp) else 0.0

    def dec_to_nmea(value, is_lat=True):
        abs_val = abs(value)
        degrees = int(abs_val)
        minutes = (abs_val - degrees) * 60
        if is_lat:
            direction = "N" if value >= 0 else "S"
            # Lat format: DDMM.MMMMMMM
            return f"{degrees:02d}{minutes:012.7f},{direction}"
        else:
            direction = "E" if value >= 0 else "W"
            # Lon format: DDDMM.MMMMMMM
            return f"{degrees:03d}{minutes:012.7f},{direction}"

    try:
        # Assuming timestamp is Unix. Convert to HHMMSS.SS
        dt_obj = datetime.datetime.fromtimestamp(ts_safe)
        time_str = dt_obj.strftime('%H%M%S.00')
    except:
        time_str = "000000.00"
    
    lat_nmea = dec_to_nmea(lat_safe, is_lat=True)
    lon_nmea = dec_to_nmea(lon_safe, is_lat=False)
    
    # 4 = RTK Fixed (Quality), 13 = Satellites, 1.0 = HDOP
    # 0.0,M = Altitude, 0.0,M = Geoidal Separation
    msg = f"GPGGA,{time_str},{lat_nmea},{lon_nmea},4,13,1.00,{alt_safe:.3f},M,0.000,M,0.1,0000"
    
    checksum = 0
    for char in msg:
        checksum ^= ord(char)
    
    return f"${msg}*{checksum:02X}"

def remove_imu_bias(dataLog):
    # Configuration
  bias_samples = 500
  gyro_bias = np.zeros(3)
  acc_bias = np.zeros(3)

  print("Removing Static Bias...")
  acc_bias[0:2] = np.mean(dataLog[:bias_samples, 0:2], axis=0) * 9.81
  acc_bias[2] = 0.0 

  gyro_bias = np.mean(dataLog[:bias_samples, 3:6], axis=0) * deg2rad

  print(f"Bias Estimation Complete")
  print(f"Accel Bias (X, Y): {acc_bias[0:2]}")
  print(f"Gyro Bias (X, Y, Z): {gyro_bias}")

  return acc_bias, gyro_bias