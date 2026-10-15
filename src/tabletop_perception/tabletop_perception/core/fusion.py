"""Multi-view fusion: merge world-frame colored clouds from several viewpoints into one.

Each added view is already in the world frame (the caller transforms it with the camera pose at
the frame's timestamp). Merging is a voxel downsample of the union - points falling in the same
voxel are averaged (positions and colors) - so overlapping views don't pile up duplicates and the
map's size stays bounded by the observed surface area, not the number of views.
"""

import numpy as np
import open3d as o3d


class VoxelMap:
    def __init__(self, voxel=0.002):
        self.voxel = float(voxel)
        self.clear()

    def clear(self):
        self.points = np.zeros((0, 3))
        self.colors = np.zeros((0, 3))
        self.n_views = 0

    def add(self, points, colors):
        """Merge one view (world frame, colors 0..1). Returns the number of map points after."""
        pts = np.vstack([self.points, np.asarray(points, float)])
        cols = np.vstack([self.colors, np.asarray(colors, float)])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))
        pcd = pcd.voxel_down_sample(self.voxel)
        self.points, self.colors = np.asarray(pcd.points), np.asarray(pcd.colors)
        self.n_views += 1
        return len(self.points)

    def __len__(self):
        return len(self.points)
