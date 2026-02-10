#This script was used to load sparse points saved in npz format from 
import numpy as np
import open3d as o3d

data = np.load("sparse_map_points.npz")
points = data["merged_points"]

print(f"Loaded {points.shape[0]} map points")

# Scale map for visibility
points *= 5.0

# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Color by depth
z = points[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
colors = np.zeros((points.shape[0], 3))
colors[:, 0] = z_norm
colors[:, 2] = 1.0 - z_norm
pcd.colors = o3d.utility.Vector3dVector(colors)

# Load trajectory if available
geometries = [pcd]

if "trajectory" in data.files:
    traj = data["trajectory"]
    traj *= 5.0

    lines = [[i, i + 1] for i in range(len(traj) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(traj),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color([1, 0, 0])  # red trajectory
    geometries.append(line_set)
    print("Trajectory loaded")

# Coordinate axes
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
geometries.append(axes)

o3d.visualization.draw_geometries(
    geometries,
    window_name="ORB-SLAM3 Sparse Map + Trajectory",
    width=1280,
    height=800
)
