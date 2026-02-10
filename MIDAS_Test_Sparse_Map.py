import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque

# -------------------------------
# 1. Initialize
# -------------------------------
model = YOLO("runs/detect/train9/weights/best.pt")  # Custom Trained Pothole Model
cap = cv2.VideoCapture(0)

# SLAM state
position = np.array([0.0, 0.0])  # x, y in meters
heading = 0.0  # radians
path = [position.copy()]
potholes = []  # List of (x, y) positions

# Feature tracking for visual odometry
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
prev_gray = None
prev_kp = None

# Pothole tracking (to avoid duplicates)
recent_potholes = deque(maxlen=30)  # Remember last N frames
POTHOLE_MIN_DISTANCE = 0.5  # meters

# Visualization settings
FEATURE_COLOR = (0, 255, 0)  # Green for tracked features
POTHOLE_COLOR = (0, 0, 255)   # Red for potholes
PATH_COLOR = (255, 255, 0)    # Yellow for path overlay

# For drawing feature history
feature_history = []

# -------------------------------
# 2. Helper Functions
# -------------------------------
def detect_movement(old_kp, new_kp):
    """Estimate camera movement from feature points"""
    if len(old_kp) < 8 or len(new_kp) < 8:
        return 0, 0
    
    # Simple translation estimate (average movement)
    dx = np.mean([new[0] - old[0] for old, new in zip(old_kp, new_kp)])
    dy = np.mean([new[1] - old[1] for old, new in zip(old_kp, new_kp)])
    
    # Convert pixels to meters (rough approximation)
    scale = 0.001  # meters per pixel movement
    return dx * scale, dy * scale

def is_new_pothole(pos, existing_potholes):
    """Check if pothole is far enough from existing ones"""
    for p in existing_potholes:
        if np.linalg.norm(pos - p) < POTHOLE_MIN_DISTANCE:
            return False
    return True

def draw_features(frame, features, color=(0, 255, 0)):
    """Draw green squares around tracked features"""
    for feature in features:
        x, y = feature.ravel()
        # Draw a small square around each feature
        cv2.rectangle(frame, 
                     (int(x-3), int(y-3)), 
                     (int(x+3), int(y+3)), 
                     color, 1)
    return frame

def draw_pothole(frame, box, color=(0, 0, 255)):
    """Draw red rectangle around detected pothole"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Draw a filled circle at center
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.circle(frame, (cx, cy), 6, color, -1)
    return frame

# -------------------------------
# 3. Main Loop
# -------------------------------
plt.ion()  # Interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: Camera feed with annotations
ax1.set_title("Live Camera Feed with SLAM Features")
ax1.axis('off')
img_display = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# Right: 2D SLAM Map
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title("2D SLAM Map")
ax2.set_xlabel("X (meters)")
ax2.set_ylabel("Y (meters)")

path_line, = ax2.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Path')
car_marker, = ax2.plot([], [], 'go', markersize=10, label='Current Position')
pothole_scatter = ax2.scatter([], [], c='red', s=100, alpha=0.7, label='Potholes')
ax2.legend()

print("Starting SLAM... Press 'q' to quit")
print("Green squares: SLAM tracking features")
print("Red boxes: Detected potholes")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Make a copy for drawing
    display_frame = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # -------------------------------
    # Visual Odometry (Simplified)
    # -------------------------------
    if prev_gray is not None:
        # Find good features to track
        kp = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                     minDistance=10, blockSize=7)
        
        if kp is not None and prev_kp is not None:
            # Track features using optical flow
            kp, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_kp, None, **lk_params)
            
            # Keep only good points
            if kp is not None:
                good_new = kp[status == 1]
                good_old = prev_kp[status == 1]
                
                if len(good_new) > 10:
                    # Draw ALL tracked features as GREEN squares
                    for point in good_new:
                        x, y = point.ravel()
                        cv2.rectangle(display_frame, 
                                     (int(x-4), int(y-4)), 
                                     (int(x+4), int(y+4)), 
                                     FEATURE_COLOR, 2)
                    
                    # Estimate movement
                    dx, dy = detect_movement(good_old, good_new)
                    
                    # Update position (convert to 2D world coordinates)
                    position[0] += dx * np.cos(heading) - dy * np.sin(heading)
                    position[1] += dx * np.sin(heading) + dy * np.cos(heading)
                    
                    # Simple heading update
                    heading += dy * 0.1
                    
                    # Add to path
                    path.append(position.copy())
    
    # Update tracking points for next frame
    prev_kp = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                      minDistance=10, blockSize=7)
    prev_gray = gray
    
    # -------------------------------
    # Pothole Detection
    # -------------------------------
    results = model(frame, verbose=False)[0]
    pothole_detected_this_frame = False
    
    for box in results.boxes:
        conf = float(box.conf)
        if conf > 0.5:  # Your confidence threshold
            # Draw RED bounding box for pothole
            display_frame = draw_pothole(display_frame, box, POTHOLE_COLOR)
            pothole_detected_this_frame = True
            
            # Add label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.putText(display_frame, f"Pothole: {conf:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, POTHOLE_COLOR, 2)
    
    # If pothole detected and not recently detected
    if pothole_detected_this_frame:
        # Convert camera position to string for checking
        pos_key = f"{position[0]:.2f}_{position[1]:.2f}"
        
        # Check if we've seen a pothole near here recently
        is_new = True
        for recent_pos in recent_potholes:
            dx = abs(position[0] - recent_pos[0])
            dy = abs(position[1] - recent_pos[1])
            if dx < POTHOLE_MIN_DISTANCE and dy < POTHOLE_MIN_DISTANCE:
                is_new = False
                break
        
        if is_new and is_new_pothole(position, potholes):
            potholes.append(position.copy())
            recent_potholes.append(position.copy())
            print(f"📍 Pothole marked at: ({position[0]:.2f}, {position[1]:.2f})")
    
    # -------------------------------
    # Display Information on Frame
    # -------------------------------
    cv2.putText(display_frame, f"Position: ({position[0]:.2f}, {position[1]:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Potholes: {len(potholes)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Features: {len(prev_kp) if prev_kp is not None else 0}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw a circle showing camera direction
    direction_length = 30
    end_x = int(50 + direction_length * np.cos(heading))
    end_y = int(120 + direction_length * np.sin(heading))
    cv2.circle(display_frame, (50, 120), 25, (255, 255, 255), 1)
    cv2.arrowedLine(display_frame, (50, 120), (end_x, end_y), (255, 255, 255), 2)
    
    # -------------------------------
    # Update Plots
    # -------------------------------
    # Update camera feed display
    img_display.set_data(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    
    # Update 2D map
    if len(path) > 1:
        path_array = np.array(path)
        path_line.set_data(path_array[:, 0], path_array[:, 1])
    
    car_marker.set_data([position[0]], [position[1]])
    
    if len(potholes) > 0:
        pothole_array = np.array(potholes)
        pothole_scatter.set_offsets(pothole_array)
    
    # Adjust map limits if needed
    if len(path) > 10:
        path_array = np.array(path)
        x_min, x_max = path_array[:, 0].min(), path_array[:, 0].max()
        y_min, y_max = path_array[:, 1].min(), path_array[:, 1].max()
        margin = 1.0
        ax2.set_xlim(x_min - margin, x_max + margin)
        ax2.set_ylim(y_min - margin, y_max + margin)
    
    # Update matplotlib plot
    plt.pause(0.001)
    
    # Show camera feed in OpenCV window too
    cv2.imshow("Live Feed with SLAM Features", display_frame)
    
    # Break on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------
# 4. Final Visualization
# -------------------------------
plt.ioff()
plt.figure(figsize=(10, 8))
plt.plot([p[0] for p in path], [p[1] for p in path], 'b-', linewidth=2, alpha=0.7, label='Path')
plt.plot(path[-1][0], path[-1][1], 'go', markersize=15, label='Final Position')

if len(potholes) > 0:
    potholes_array = np.array(potholes)
    plt.scatter(potholes_array[:, 0], potholes_array[:, 1],
                c='red', s=100, alpha=0.7, label='Potholes', marker='s')  # Square markers

plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('Final 2D SLAM Map with Pothole Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

print(f"\nSLAM Summary:")
print(f"- Path length: {len(path)} points")
print(f"- Distance traveled: {np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)):.2f} meters")
print(f"- Potholes detected: {len(potholes)}")