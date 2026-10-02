#!/usr/bin/env python3
"""Wrist depth image -> PointCloud2 for the MuJoCo backend (its CameraPlugin publishes none).

Started by ur_bringup_sim/robot_mujoco.launch.py (pointcloud:=true, the default); standalone:
  ros2 run ur_camera_sim depth_to_cloud.py [--ros-args -p stride:=3 -p max_range:=1.2]

Back-projects /camera/depth/image_rect_raw with the intrinsics from /camera/color/camera_info
(depth and color are rendered by the same MuJoCo camera) and publishes /camera/depth/color/points
- the topic name the Gazebo bridge (config/camera_bridge.yaml) and the real realsense2_camera
driver use, so consumers such as MoveIt's octomap (wrist_depth_sensors_3d.yaml) are
backend-agnostic. Unlike those, this cloud is XYZ only (no color): MoveIt needs only geometry.

Frame: points are stamped in camera_color_toein_frame. The MuJoCo camera renders from that site,
and mujoco_ros2_control's URDF->MJCF conversion already applies the ROS-optical -> MuJoCo camera
rotation to it (see ur5e_2f85_camera_mujoco.urdf.xacro), so in TF it is a proper optical frame:
+z along the view, +x = image right, +y = image down. Overridable with the `frame_id` parameter
if a check against the table (TABLE_TOP_Z) ever shows otherwise.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]


class DepthToCloud(Node):
    def __init__(self):
        super().__init__("depth_to_cloud")
        self.stride = int(self.declare_parameter("stride", 3).value)     # 640x480 / 3 -> ~34k points
        self.min_range = float(self.declare_parameter("min_range", 0.05).value)
        self.max_range = float(self.declare_parameter("max_range", 1.2).value)
        self.frame_id = self.declare_parameter("frame_id", "camera_color_toein_frame").value
        self._rays = None        # (h', w', 2) normalized (x/z, y/z) per strided pixel
        self._rays_key = None
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/depth/image_rect_raw", self._on_depth, qos_profile_sensor_data)
        self.pub = self.create_publisher(PointCloud2, "/camera/depth/color/points", qos_profile_sensor_data)

    def _on_info(self, msg):
        key = (msg.width, msg.height, *msg.k)
        if key == self._rays_key:
            return
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        u = np.arange(0, msg.width, self.stride)
        v = np.arange(0, msg.height, self.stride)
        uu, vv = np.meshgrid(u, v)
        self._rays = np.stack([(uu - cx) / fx, (vv - cy) / fy], axis=-1).astype(np.float32)
        self._rays_key = key

    def _on_depth(self, msg):
        if self._rays is None:
            return
        if msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width]
        elif msg.encoding in ("16UC1", "mono16"):
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
            depth = depth.astype(np.float32) * 1e-3
        else:
            self.get_logger().error(f"unsupported depth encoding {msg.encoding}", throttle_duration_sec=5.0)
            return
        z = depth[::self.stride, ::self.stride]
        if z.shape != self._rays.shape[:2]:
            self.get_logger().error("depth/camera_info size mismatch", throttle_duration_sec=5.0)
            return
        valid = np.isfinite(z) & (z > self.min_range) & (z < self.max_range)
        z = z[valid]
        rays = self._rays[valid]
        pts = np.empty((z.size, 3), dtype=np.float32)
        pts[:, 0] = rays[:, 0] * z
        pts[:, 1] = rays[:, 1] * z
        pts[:, 2] = z

        cloud = PointCloud2()
        cloud.header.stamp = msg.header.stamp
        cloud.header.frame_id = self.frame_id
        cloud.height, cloud.width = 1, pts.shape[0]
        cloud.fields = _FIELDS
        cloud.is_bigendian = False
        cloud.point_step, cloud.row_step = 12, 12 * pts.shape[0]
        cloud.is_dense = True
        cloud.data = pts.tobytes()
        self.pub.publish(cloud)


def main():
    rclpy.init()
    node = DepthToCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
