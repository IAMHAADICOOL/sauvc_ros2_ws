"""imu_check_common — shared quaternion math + sanity checks for the two IMU bring-up
scripts (Taobotics HFI-A9 and Pixhawk-via-mavros). Not a ROS node itself."""
import math


def quat_to_euler_deg(x, y, z, w):
    """Returns (roll, pitch, yaw) in degrees, ZYX convention."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    norm_ok = abs(n - 1.0) < 0.05 if n > 1e-9 else False
    if n < 1e-9:
        return 0.0, 0.0, 0.0, False
    x, y, z, w = x / n, y / n, z / n, w / n
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw), norm_ok


def accel_sanity(ax, ay, az):
    """Returns (magnitude, ok). Stationary specific-force magnitude should be ~9.8 m/s^2
    (gravity reaction). Returns ok=None if the reading looks unpopulated (all ~0), which
    is normal for AHRS-only drivers that only publish orientation."""
    mag = math.sqrt(ax * ax + ay * ay + az * az)
    if mag < 0.05:
        return mag, None   # driver likely doesn't populate linear_acceleration
    return mag, 4.0 < mag < 20.0   # generous band: stationary ~9.8, but allow handling
