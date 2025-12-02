import numpy as np
import cv2
import matplotlib.pyplot as plt


SHIP1_COLOR = [50, 50, 200]   # approx (blue-ish)
SHIP2_COLOR = [73, 235, 189]   # approx (green-ish)

def extract_centroid(mask):
    """
    Returns (cx, cy), contour
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None, contour

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), contour


def extract_orientation(contour):
    """
    Uses PCA to determine ship orientation.
    Returns angle in radians or None if contour not found.
    """

    if contour is None or len(contour) < 5:
        return None

    pts = contour.reshape(-1, 2).astype(np.float32)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
    # longest axis (principal direction)
    vx, vy = eigenvectors[0]

    angle = np.arctan2(vy, vx)
    return angle


def detect_ship_centroids(obs_frame):
    # If stacked frames: take most recent frame
    if obs_frame.shape[2] == 12:
        frame = obs_frame[:, :, -3:]  # last RGB frame
    else:
        frame = obs_frame

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Blue ship mask (tune as needed)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Green ship mask (tune as needed)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    blue_centroid, blue_contour = extract_centroid(mask_blue)
    green_centroid, green_contour   = extract_centroid(mask_green)

    blue_angle = extract_orientation(blue_contour)
    green_angle  = extract_orientation(green_contour)

    return {
        "blue_pos": blue_centroid,
        "blue_angle": blue_angle,
        "green_pos": green_centroid,
        "green_angle": green_angle,
    }

def estimate_orientation(prev_cx, prev_cy, cx, cy):
    dx = cx - prev_cx
    dy = cy - prev_cy
    angle = np.arctan2(dy, dx)
    return angle

def build_feature_vector(agent, centroid_info):
    if agent.startswith("second"):
        own_pos = centroid_info["blue_pos"]
        own_angle = centroid_info["blue_angle"]
        enemy_pos = centroid_info["green_pos"]
        enemy_angle = centroid_info["green_angle"]
    elif agent.startswith("first"):
        own_pos = centroid_info["green_pos"]
        own_angle = centroid_info["green_angle"]
        enemy_pos = centroid_info["blue_pos"]
        enemy_angle = centroid_info["blue_angle"]
    else:
        # Unknown agent
        return None

    # Check data validity
    if None in (own_pos, own_angle, enemy_pos, enemy_angle):
        return None

    # Positions
    own_x, own_y = own_pos
    enemy_x, enemy_y = enemy_pos

    # Compute relative vector
    dx = enemy_x - own_x
    dy = enemy_y - own_y

    # Distance
    dist = np.sqrt(dx*dx + dy*dy)

    # Relative angle from own heading to enemy vector
    angle_to_enemy = np.arctan2(dy, dx)
    relative_angle = angle_to_enemy - own_angle

    # Normalize relative_angle to [-pi, pi]
    relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

    feature_vec = np.array([
        own_x, own_y,
        own_angle,
        enemy_x, enemy_y,
        enemy_angle,
        dist,
        relative_angle
    ], dtype=np.float32)

    return feature_vec