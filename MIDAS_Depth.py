import cv2
import open3d as o3d
import numpy as np
import torch

# Load MiDaS depth model
print("Loading MiDaS model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print(f"Model loaded on {device}")

# Feature detector
orb = cv2.ORB_create(500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
h, w = frame.shape[:2]

# Camera intrinsics (approximate)
focal = w * 0.9
cx, cy = w / 2, h / 2
K = np.array([[focal, 0, cx],
              [0, focal, cy],
              [0, 0, 1]], dtype=np.float64)

# Global map
points_3d_global = []
colors_global = []

# Camera pose
R_global = np.eye(3)
t_global = np.zeros((3, 1))

# Previous frame data
prev_kp = None
prev_des = None
prev_depth = None

frame_num = 0
skip_frames = 2
last_depth_colored = None


def backproject_to_3d(kp_list, depth_map, K):
    """Convert 2D keypoints + depth to 3D points"""
    points_3d = []
    for kp in kp_list:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        z = depth_map[y, x]
        
        if z > 0.1 and z < 10:  # Valid depth range
            # Back-project using pinhole model
            X = (x - K[0, 2]) * z / K[0, 0]
            Y = (y - K[1, 2]) * z / K[1, 1]
            Z = z
            points_3d.append([X, -Y, Z])  # Flip Y for right-handed coords
    
    return np.array(points_3d)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Skip frames for performance
    if frame_num % skip_frames != 0:
        if last_depth_colored is not None:
            combined = np.hstack([frame, last_depth_colored])
        else:
            combined = np.hstack([frame, np.zeros_like(frame)])
        cv2.putText(combined, f"Points: {len(points_3d_global)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", combined)
        if cv2.waitKey(1) == 27:
            break
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints
    kp, des = orb.detectAndCompute(gray, None)

    # Estimate depth with MiDaS
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # MiDaS outputs inverse depth, convert to metric-like depth
    depth = depth.max() - depth  # Invert
    depth = depth / depth.max() * 5.0  # Scale to ~0-5 range

    if prev_des is not None and des is not None and len(kp) > 20:
        # Match features
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) > 15:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

            # Estimate motion
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)

            if E is not None and mask.sum() > 10:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

                # Update global pose
                t_global = t_global + R_global @ t
                R_global = R @ R_global

                # Back-project current keypoints to 3D
                pts_3d_local = backproject_to_3d(kp, depth, K)

                if len(pts_3d_local) > 0:
                    # Transform to global coordinates
                    pts_3d_world = (R_global @ pts_3d_local.T + t_global).T

                    # Get colors for these points
                    for i, kp_pt in enumerate(kp[:len(pts_3d_world)]):
                        x, y = int(kp_pt.pt[0]), int(kp_pt.pt[1])
                        if 0 <= x < w and 0 <= y < h:
                            bgr = frame[y, x]
                            colors_global.append([bgr[2]/255, bgr[1]/255, bgr[0]/255])

                    points_3d_global.extend(pts_3d_world.tolist()[:len(colors_global) - len(points_3d_global) + len(pts_3d_world)])

    # Update previous frame
    prev_kp = kp
    prev_des = des
    prev_depth = depth

    # Visualization
    depth_vis = (depth / depth.max() * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    last_depth_colored = depth_colored.copy()

    # Draw keypoints
    frame_vis = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)

    combined = np.hstack([frame_vis, depth_colored])
    cv2.putText(combined, f"Points: {len(points_3d_global)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera", combined)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Visualize the sparse map
print(f"Total points mapped: {len(points_3d_global)}")

if len(points_3d_global) > 0:
    # Ensure colors match points
    min_len = min(len(points_3d_global), len(colors_global))
    points_3d_global = points_3d_global[:min_len]
    colors_global = colors_global[:min_len]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points_3d_global))
    
    # Set all points to green
    pcd.paint_uniform_color([0, 1, 0])

    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Center for visualization
    pcd.translate(-pcd.get_center())

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    print(f"Points after filtering: {len(pcd.points)}")
    o3d.visualization.draw_geometries([pcd, coord_frame],
                                       window_name="Sparse Map",
                                       width=1280, height=720)