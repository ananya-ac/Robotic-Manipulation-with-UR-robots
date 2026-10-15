"""ROS 2 front end for tabletop_perception's core (segmentation, fusion, antipodal grasps).

The node never moves a robot: clients choose viewpoints, move there, and call add_view.

Subscribes (synchronized): color image, depth image, camera_info; uses TF for camera and
self-filter frame poses (in `world_frame`, at each frame's stamp).

Services:
  ~/reset_map       std_srvs/Trigger   empty the fused map
  ~/add_view        std_srvs/Trigger   wait for a frame newer than the call, self-filter it, add it
                                       to the fused map (2 mm voxels, world frame)
  ~/detect_objects  DetectObjects      segment the fused map (use_map) or a single frame
  ~/plan_grasps     PlanGrasps         antipodal grasps for one object of the latest detection

Publishes: ~/objects, ~/grasps (MarkerArray), ~/map (PointCloud2, after each add_view).

Parameters: see config/tabletop_mujoco.yaml.
"""

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped, Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from tabletop_perception.core import antipodal_grasps, backproject, detect, pixel_rays
from tabletop_perception.core.fusion import VoxelMap
from tabletop_perception.core.grasping import GraspConfig
from tabletop_perception.params import configs_from_params
from tabletop_perception_msgs.msg import GraspCandidate
from tabletop_perception_msgs.srv import DetectObjects, PlanGrasps


def quat_to_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return x, y, z, w


def pose_msg(T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, matrix_to_quat(T[:3, :3]))
    return p


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node", automatically_declare_parameters_from_overrides=True)
        params = {name: p.value for name, p in self.get_parameters_by_prefix("").items()}
        self.seg_cfg, self.grasp_cfg, self.gripper, self.self_filter = configs_from_params(params)
        self.world = params.get("world_frame", "world")
        self.stride = int(params.get("stride", 2))
        self.max_range = float(params.get("max_range", 1.2))
        self.map = VoxelMap(float(params.get("map_voxel", 0.002)))

        self.tf = Buffer(cache_time=Duration(seconds=30))
        TransformListener(self.tf, self)
        self._lock = threading.Lock()
        self._frame = None              # (stamp_ns, frame_id, depth, rgb, K)
        self._rays_key, self._rays = None, None
        self._scene, self._source = None, ""

        sub_group = ReentrantCallbackGroup()
        srv_group = MutuallyExclusiveCallbackGroup()
        subs = [Subscriber(self, Image, params.get("color_topic", "/camera/color/image_raw"),
                           qos_profile=qos_profile_sensor_data, callback_group=sub_group),
                Subscriber(self, Image, params.get("depth_topic", "/camera/depth/image_rect_raw"),
                           qos_profile=qos_profile_sensor_data, callback_group=sub_group),
                Subscriber(self, CameraInfo, params.get("info_topic", "/camera/color/camera_info"),
                           qos_profile=qos_profile_sensor_data, callback_group=sub_group)]
        self._sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.02)
        self._sync.registerCallback(self._on_frame)

        self.create_service(Trigger, "~/reset_map", self._reset_map, callback_group=srv_group)
        self.create_service(Trigger, "~/add_view", self._add_view, callback_group=srv_group)
        self.create_service(DetectObjects, "~/detect_objects", self._detect, callback_group=srv_group)
        self.create_service(PlanGrasps, "~/plan_grasps", self._plan_grasps, callback_group=srv_group)
        self.pub_objects = self.create_publisher(MarkerArray, "~/objects", 1)
        self.pub_grasps = self.create_publisher(MarkerArray, "~/grasps", 1)
        self.pub_map = self.create_publisher(PointCloud2, "~/map", 1)
        self.get_logger().info(f"ready: palette {list(self.seg_cfg.palette)}, grasp method {self.grasp_cfg.method}, "
                               f"self-filter frames {[f for f, _, _ in self.self_filter]}")

    # ---- frames -----------------------------------------------------------------------------

    def _on_frame(self, color, depth, info):
        if depth.encoding == "32FC1":
            d = np.frombuffer(depth.data, np.float32).reshape(depth.height, depth.step // 4)[:, :depth.width]
        elif depth.encoding in ("16UC1", "mono16"):
            d = np.frombuffer(depth.data, np.uint16).reshape(depth.height, depth.step // 2)[:, :depth.width] * 1e-3
        else:
            self.get_logger().error(f"unsupported depth encoding {depth.encoding}", throttle_duration_sec=5.0)
            return
        if color.encoding not in ("rgb8", "bgr8"):
            self.get_logger().error(f"unsupported color encoding {color.encoding}", throttle_duration_sec=5.0)
            return
        rgb = np.frombuffer(color.data, np.uint8).reshape(color.height, color.step)[:, :color.width * 3]
        rgb = rgb.reshape(color.height, color.width, 3)
        if color.encoding == "bgr8":
            rgb = rgb[:, :, ::-1]
        stamp = Time.from_msg(depth.header.stamp).nanoseconds
        with self._lock:
            self._frame = (stamp, depth.header.frame_id, d, rgb, np.array(info.k, float))

    def _wait_frame(self, after_ns, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                f = self._frame
            if f is not None and f[0] > after_ns:
                return f
            time.sleep(0.02)
        return None

    def _latest_stamp(self):
        with self._lock:
            return self._frame[0] if self._frame else -1

    def _lookup(self, frame, stamp_ns):
        t = self.tf.lookup_transform(self.world, frame, Time(nanoseconds=stamp_ns), timeout=Duration(seconds=0.5))
        q, p = t.transform.rotation, t.transform.translation
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(q.x, q.y, q.z, q.w)
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def _frame_cloud(self, frame):
        """One frame -> world-frame (points, colors), self-filtered."""
        stamp, cam_frame, depth, rgb, K = frame
        key = (depth.shape, tuple(K))
        if key != self._rays_key:
            self._rays = pixel_rays(depth.shape[1], depth.shape[0], K[0], K[4], K[2], K[5], self.stride)
            self._rays_key = key
        pts, cols = backproject(depth, self._rays, self.stride, max_range=self.max_range, rgb=rgb)
        T = self._lookup(cam_frame, stamp)
        pts = pts @ T[:3, :3].T + T[:3, 3]
        for frame_id, center, half in self.self_filter:
            B = self._lookup(frame_id, stamp)
            c = B[:3, :3] @ center + B[:3, 3]
            inside = np.all(np.abs((pts - c) @ B[:3, :3]) <= half, axis=1)
            pts, cols = pts[~inside], cols[~inside]
        return pts, cols

    # ---- services ---------------------------------------------------------------------------

    def _reset_map(self, req, res):
        self.map.clear()
        res.success, res.message = True, "map cleared"
        return res

    def _add_view(self, req, res):
        frame = self._wait_frame(self._latest_stamp())
        if frame is None:
            res.success, res.message = False, "no new RGB-D frame within 3 s"
            return res
        try:
            pts, cols = self._frame_cloud(frame)
        except Exception as exc:      # TF lookup failures etc.
            res.success, res.message = False, f"add_view: {exc}"
            return res
        n = self.map.add(pts, cols)
        self.pub_map.publish(self._cloud_msg(self.map.points, frame[0]))
        res.success, res.message = True, f"{self.map.n_views} views, {n} map points (+{len(pts)} this view)"
        return res

    def _detect(self, req, res):
        t0 = time.monotonic()
        try:
            if req.use_map:
                if len(self.map) == 0:
                    raise ValueError("map is empty - call add_view first")
                pts, cols, stamp, source = self.map.points, self.map.colors, self._latest_stamp(), "map"
            else:
                frame = self._wait_frame(self._latest_stamp()) if req.fresh else self._frame
                if frame is None:
                    raise ValueError("no RGB-D frame")
                pts, cols = self._frame_cloud(frame)
                stamp, source = frame[0], "frame"
            scene = detect(pts, cols, self.seg_cfg)
        except Exception as exc:
            res.success, res.message = False, f"detect: {exc}"
            return res
        self._scene, self._source = scene, source
        res.detections = Detection3DArray()
        res.detections.header.frame_id = self.world
        res.detections.header.stamp = Time(nanoseconds=max(stamp, 0)).to_msg()
        for o in scene.objects:
            if req.names and o.name not in req.names:
                continue
            det = Detection3D()
            det.header = res.detections.header
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = o.R, o.center
            det.bbox.center = pose_msg(T)
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = map(float, o.extent)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = o.name
            hyp.hypothesis.score = 1.0 if o.top_visible else 0.5
            hyp.pose.pose = det.bbox.center
            det.results.append(hyp)
            res.detections.detections.append(det)
        missing = [n for n in req.names if scene.get(n) is None]
        res.success = not missing
        res.message = f"{scene.message} from {source} ({len(pts):,} pts, {time.monotonic() - t0:.2f} s)" + \
            (f"; not found: {missing}" if missing else "")
        self.pub_objects.publish(self._object_markers(scene))
        return res

    def _plan_grasps(self, req, res):
        t0 = time.monotonic()
        obj = self._scene.get(req.name) if self._scene else None
        if obj is None:
            res.success, res.message = False, f"{req.name!r} not in the latest detection - call detect_objects first"
            return res
        cfg = self.grasp_cfg
        if req.max_candidates > 0:
            cfg = GraspConfig(**{**cfg.__dict__, "max_candidates": int(req.max_candidates)})
        try:
            grasps = antipodal_grasps(obj, self._scene, self.gripper, cfg)
            method = cfg.method
            if not grasps and cfg.method == "antipodal_3d" and self._source == "frame":
                # A single view rarely shows opposite faces; the outline sampler handles that case.
                grasps = antipodal_grasps(obj, self._scene, self.gripper, GraspConfig(**{**cfg.__dict__, "method": "outline"}))
                method = "outline (single-frame fallback)"
        except Exception as exc:
            res.success, res.message = False, f"plan_grasps: {exc}"
            return res
        now = self.get_clock().now().to_msg()
        for g in grasps:
            c = GraspCandidate()
            c.tip_pose = PoseStamped()
            c.tip_pose.header.frame_id, c.tip_pose.header.stamp = self.world, now
            c.tip_pose.pose = pose_msg(g.T)
            c.width, c.score = float(g.width), float(g.score)
            c.approach = Vector3(x=float(g.T[0, 2]), y=float(g.T[1, 2]), z=float(g.T[2, 2]))
            res.grasps.append(c)
        res.success = bool(grasps)
        res.message = f"{len(grasps)} grasps for {req.name} ({method}, from {self._source}, {time.monotonic() - t0:.2f} s)"
        self.get_logger().info(res.message)
        self.pub_grasps.publish(self._grasp_markers(grasps))
        return res

    # ---- visualization ----------------------------------------------------------------------

    def _cloud_msg(self, pts, stamp_ns):
        msg = PointCloud2()
        msg.header.frame_id, msg.header.stamp = self.world, Time(nanoseconds=max(stamp_ns, 0)).to_msg()
        msg.height, msg.width = 1, len(pts)
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        msg.point_step, msg.row_step, msg.is_dense = 12, 12 * len(pts), True
        msg.data = np.asarray(pts, np.float32).tobytes()
        return msg

    def _object_markers(self, scene):
        arr = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for i, o in enumerate(scene.objects):
            col = self.seg_cfg.palette.get(o.name, np.array([0.7, 0.7, 0.7]))
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = o.R, o.center
            box = Marker(type=Marker.CUBE, id=2 * i, ns="objects", pose=pose_msg(T),
                         color=ColorRGBA(r=float(col[0]), g=float(col[1]), b=float(col[2]), a=0.5))
            box.header.frame_id = self.world
            box.scale.x, box.scale.y, box.scale.z = map(float, np.maximum(o.extent, 1e-3))
            label = Marker(type=Marker.TEXT_VIEW_FACING, id=2 * i + 1, ns="labels", text=o.name,
                           color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
            label.header.frame_id = self.world
            label.pose.position.x, label.pose.position.y = float(o.center[0]), float(o.center[1])
            label.pose.position.z = float(o.top_z + 0.03)
            label.scale.z = 0.02
            arr.markers += [box, label]
        return arr

    def _grasp_markers(self, grasps, show=5):
        arr = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for i, g in enumerate(grasps[:show]):
            a = 1.0 if i == 0 else 0.35
            m = Marker(type=Marker.LINE_LIST, id=i, ns="grasps", color=ColorRGBA(r=1.0, g=0.8, b=0.0, a=a))
            m.header.frame_id = self.world
            m.scale.x = 0.003
            T, h = g.T, 0.5 * g.width
            x, z, o = T[:3, 0], T[:3, 2], T[:3, 3]
            pts = [o + h * x, o - h * x,                       # closing axis between the contacts
                   o + h * x, o + h * x - 0.05 * z,            # fingers
                   o - h * x, o - h * x - 0.05 * z,
                   o - 0.05 * z, o - 0.12 * z]                 # approach (palm side)
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
            arr.markers.append(m)
        return arr


def main():
    rclpy.init()
    node = PerceptionNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
