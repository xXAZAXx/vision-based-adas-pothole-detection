import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
# ----------------------------------------------------

# Load Custom Trained Pothole Detection Model
model = YOLO("runs/detect/train9/weights/best.pt")

# ----------------------------------------------------
# Camera Setup & Optical Flow Parametersq
# ----------------------------------------------------
cap = cv2.VideoCapture(0)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ----------------------------------------------------
# Read first frame
# ----------------------------------------------------
ret, old_frame = cap.read()
if not ret:
    print("Failed to read camera")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
if p0 is None:
    print("No features found initially")
    cap.release()
    exit()

# ----------------------------------------------------
# Path plot setup
# ----------------------------------------------------
x, y = 0, 0
# Separate path data for normal (blue) and pothole (red) segments
# Use NaN to create breaks in lines (matplotlib won't draw lines through NaN)
normal_path_x, normal_path_y = [x], [y]
pothole_path_x, pothole_path_y = [], []
current_pothole_state = False  # tracks if currently in pothole zone
prev_pothole_state = False  # tracks previous state to detect transitions

plt.ion()
fig, ax_plot = plt.subplots()
normal_line, = ax_plot.plot(normal_path_x, normal_path_y, '-b', label="Normal Path")
pothole_line, = ax_plot.plot(pothole_path_x, pothole_path_y, '-r', label="Pothole Zone")
ax_plot.set_xlim(-5, 5)
ax_plot.set_ylim(-5, 5)
ax_plot.set_title('Camera Path & Potholes')
ax_plot.legend()

# ----------------------------------------------------
# Pothole detection filter
# ----------------------------------------------------
pothole_detected_frames = 0
frames_required = 3  # must detect pothole for 3 consecutive frames to count
confidence_threshold = 0.5  # minimum YOLO confidence to consider valid

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------------------
        # Optical Flow Tracking
        # ----------------------------------------------------
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None and st is not None and st.sum() > 0:
            good_new = p1[st.flatten() == 1]
            good_old = p0[st.flatten() == 1]

            if good_new.shape[1] == 1:
                good_new = good_new.reshape(-1, 2)
            if good_old.shape[1] == 1:
                good_old = good_old.reshape(-1, 2)

            if len(good_new) > 0:
                dx = np.mean(good_new[:, 0] - good_old[:, 0])
                dy = np.mean(good_new[:, 1] - good_old[:, 1])

                if not (np.isnan(dx) or np.isnan(dy)):
                    x += dx * 0.01  # scale factor
                    y += dy * 0.01

                    # Add point to appropriate line based on pothole state
                    if current_pothole_state:
                        # Switching from normal to pothole - start new segment
                        if not prev_pothole_state:
                            pothole_path_x.append(np.nan)
                            pothole_path_y.append(np.nan)
                        pothole_path_x.append(x)
                        pothole_path_y.append(y)
                    else:
                        # Switching from pothole to normal - start new segment
                        if prev_pothole_state:
                            normal_path_x.append(np.nan)
                            normal_path_y.append(np.nan)
                        normal_path_x.append(x)
                        normal_path_y.append(y)
                    
                    prev_pothole_state = current_pothole_state

                    normal_line.set_xdata(normal_path_x)
                    normal_line.set_ydata(normal_path_y)
                    pothole_line.set_xdata(pothole_path_x)
                    pothole_line.set_ydata(pothole_path_y)
                    ax_plot.relim()
                    ax_plot.autoscale_view()
                    plt.pause(0.01)

                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)

        # Update features if too few
        if len(p0) < 10:
            new_p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            if new_p0 is not None:
                p0 = new_p0

        # ----------------------------------------------------
        # Pothole Detection
        # ----------------------------------------------------
        results = model.predict(frame, conf=confidence_threshold, device=0, stream=False)
        pothole_found = False
        for r in results:
            boxes = r.boxes.xyxy
            if len(boxes) > 0:
                # Only consider **first detection per frame**
                cx = int((boxes[0][0] + boxes[0][2]) / 2)
                cy = int((boxes[0][1] + boxes[0][3]) / 2)
                pothole_found = True
                break  # stop after first box

        # ----------------------------------------------------
        # Temporal filtering
        # ----------------------------------------------------
        if pothole_found:
            pothole_detected_frames += 1
        else:
            pothole_detected_frames = 0

        # Update pothole state - red line when pothole detected for enough frames
        if pothole_detected_frames >= frames_required:
            current_pothole_state = True
        else:
            current_pothole_state = False

        # ----------------------------------------------------
        # Show camera feed with bounding boxes 
        # ----------------------------------------------------
        for r in results:
            frame = r.plot()
        cv2.imshow("Pothole Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()