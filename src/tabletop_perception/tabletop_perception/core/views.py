"""Viewpoint generation for active perception (camera poses only - no robot knowledge).

orbit_views() returns camera poses on a ring around a target, each looking at it. The caller
turns a camera pose into its robot's tool pose (fixed tool->camera transform), filters by
reachability, and moves there; then asks the node to add the view.

Camera frame convention is the ROS optical frame: +z along the view, +x image right, +y image down.
"""

import numpy as np


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Optical-frame camera pose at `eye` looking at `target`, image x kept horizontal (no roll)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:                 # looking straight down/up: any horizontal x
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


def orbit_views(target, radius, elevations_deg, azimuths_deg):
    """Camera poses on spheres around `target`: for each (elevation, azimuth) the camera sits at
    `radius` from the target, `elevation` above its horizontal plane, looking at it.
    Azimuth 0 = +x of the target. Returns [(elevation, azimuth, T_world_camera)]."""
    target = np.asarray(target, float)
    out = []
    for el in elevations_deg:
        for az in azimuths_deg:
            e, a = np.radians(el), np.radians(az)
            eye = target + radius * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
            out.append((float(el), float(az), look_at(eye, target)))
    return out


def object_views(center, yaw, radius, elevations_deg, n_azimuth=4, offset_deg=45.0):
    """Views around ONE detected object, at azimuths offset from its faces (yaw + offset + k*360/n):
    with the default 4 views at 45 deg off the faces, each view sees two adjacent side faces and
    every side face is seen twice. Returns [(elevation, azimuth, T_world_camera)] like orbit_views."""
    az = [np.degrees(yaw) + offset_deg + k * 360.0 / n_azimuth for k in range(n_azimuth)]
    return orbit_views(center, radius, elevations_deg, az)
