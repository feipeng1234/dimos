# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from queue import Empty, Queue
import re
import threading
from pathlib import Path

from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import (
    CameraInfo as ROSCameraInfo,
    CompressedImage as ROSCompressedImage,
    Image as ROSImage,
    PointCloud2 as ROSPointCloud2,
)
from std_msgs.msg import Header as ROSHeader
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import (
    Detection2DArray as ROSDetection2DArray,
    Detection3DArray as ROSDetection3DArray,
)

from dimos.core import Module, rpc
from dimos.msgs.geometry_msgs import Quaternion, Transform, Vector3
from dimos.msgs.sensor_msgs import CameraInfo, Image
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector, YoloePromptMode
from dimos.perception.detection.mesh_pose_client import MeshPoseClient
from dimos.perception.detection.objectDB import ObjectDB
from dimos.perception.detection.type import ImageDetections2D
from dimos.perception.detection.type.detection3d.object import (
    Object,
    aggregate_pointclouds,
    to_detection3d_array,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class ObjectSceneRegistrationModule(Module):
    """Module for detecting objects in camera images using YOLO-E with 2D and 3D detection.

    Optionally supports mesh/pose enhancement via hosted service (SAM3 + SAM3D + FoundationPose).
    """

    _detector: Yoloe2DDetector | None = None
    _node: Node | None = None
    _bridge: CvBridge | None = None
    _spin_thread: threading.Thread | None = None
    _processing_thread: threading.Thread | None = None
    _processing_queue: Queue | None = None
    _running: bool = False
    _camera_info: CameraInfo | None = None
    _tf_buffer: tf2_ros.Buffer | None = None
    _tf_listener: tf2_ros.TransformListener | None = None
    _object_db: ObjectDB | None = None
    _mesh_pose_client: MeshPoseClient | None = None
    
    # Async mesh generation worker
    _mesh_request_queue: Queue | None = None
    _mesh_request_states: dict[str, str] = {}
    _mesh_worker_thread: threading.Thread | None = None

    def __init__(
        self,
        image_topic: str = "/camera/color/image_raw/compressed",
        depth_topic: str = "/camera/aligned_depth_to_color/image_raw/compressedDepth",
        camera_info_topic: str = "/camera/color/camera_info",
        detections_2d_topic: str = "/object_detections/bbox",
        detections_3d_topic: str = "/object_detections/bbox3d",
        overlay_topic: str = "/object_detections/overlay",
        pointcloud_topic: str = "/object_detections/pointcloud",
        target_frame: str = "map",
        mesh_pose_service_url: str | None = None,
        auto_mesh_pose: bool = True,
        mesh_pose_use_box_prompt: bool = True,
        object_db_require_same_name_for_dedup: bool | None = None,
        mesh_store_dir: str = "/home/dimensional/dimos/meshes",
        mesh_marker_topic: str = "/object_detections/mesh_markers",
    ) -> None:
        """
        Initialize ObjectSceneRegistrationModule.

        Args:
            image_topic: ROS topic for compressed color images.
            depth_topic: ROS topic for compressed depth images.
            camera_info_topic: ROS topic for camera intrinsics.
            detections_2d_topic: ROS topic to publish 2D detections.
            detections_3d_topic: ROS topic to publish 3D detections.
            overlay_topic: ROS topic to publish detection overlay image.
            pointcloud_topic: ROS topic to publish aggregated pointclouds.
            target_frame: Target TF frame for 3D detections (e.g., "map").
            mesh_pose_service_url: Optional URL of hosted mesh/pose service
                (e.g., "http://localhost:8080"). If provided, detections
                will be enhanced with mesh data and accurate 6D pose.
            auto_mesh_pose: Auto-enhance detections with mesh/pose (default True).
                Set to False for interactive-only mode.
            mesh_pose_use_box_prompt: Use box-only prompting (ignore YOLO-E labels).
                Default True to avoid passing garbage labels to SAM3.
        """
        super().__init__()
        self._image_topic = image_topic
        self._depth_topic = depth_topic
        self._camera_info_topic = camera_info_topic
        self._detections_2d_topic = detections_2d_topic
        self._detections_3d_topic = detections_3d_topic
        self._overlay_topic = overlay_topic
        self._pointcloud_topic = pointcloud_topic
        self._target_frame = target_frame
        self._mesh_pose_service_url = mesh_pose_service_url
        self._auto_mesh_pose = auto_mesh_pose
        self._mesh_pose_use_box_prompt = mesh_pose_use_box_prompt
        self._object_db_require_same_name_for_dedup = object_db_require_same_name_for_dedup
        self._mesh_store_dir = mesh_store_dir
        self._mesh_marker_topic = mesh_marker_topic
        self._bridge = CvBridge()

        # Track latest data for interactive queries
        self._latest_lock = threading.RLock()
        self._latest_color_image: Image | None = None
        self._latest_depth_image: Image | None = None
        self._latest_detections_2d: ImageDetections2D | None = None
        self._latest_objects: list[Object] = []
        
        # Async mesh generation
        self._mesh_request_queue = Queue()
        self._mesh_request_states = {}

    @rpc
    def start(self) -> None:
        super().start()
        self._running = True

        # Initialize ObjectDB for spatial memory
        require_same_name = (
            self._object_db_require_same_name_for_dedup
            if self._object_db_require_same_name_for_dedup is not None
            else (not self._mesh_pose_use_box_prompt)
        )
        self._object_db = ObjectDB(require_same_name_for_distance_match=require_same_name)
        logger.info("Initialized ObjectDB for spatial memory")

        # Initialize mesh/pose client if URL provided
        if self._mesh_pose_service_url:
            self._mesh_pose_client = MeshPoseClient(
                service_url=self._mesh_pose_service_url,
            )
            if self._mesh_pose_client.health_check():
                logger.info(f"Mesh/pose enhancement enabled via {self._mesh_pose_service_url}")
            else:
                logger.warning(
                    f"Mesh/pose service at {self._mesh_pose_service_url} is not healthy, "
                    "detections will not be enhanced"
                )
                self._mesh_pose_client.close()
                self._mesh_pose_client = None
        else:
            logger.info("Mesh/pose enhancement disabled (no service URL provided)")

        # Initialize detector (uses yoloe-11l-seg-pf.pt for LRPC mode by default)
        self._detector = Yoloe2DDetector(
            prompt_mode=YoloePromptMode.LRPC,
        )
        logger.info("Initialized YOLO-E detector in LRPC (prompt-free) mode")

        # Initialize ROS
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("object_scene_registration")

        # Initialize TF2 buffer and listener
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        # Wait for camera info once (it rarely changes)
        logger.info(f"Waiting for camera_info on {self._camera_info_topic}...")
        success, camera_info_msg = wait_for_message(
            ROSCameraInfo,
            self._node,
            self._camera_info_topic,
            time_to_wait=10.0,
        )
        if success and camera_info_msg:
            self._camera_info = CameraInfo.from_ros_msg(camera_info_msg)
            logger.info(f"Received camera_info: {camera_info_msg.width}x{camera_info_msg.height}")
        else:
            raise RuntimeError(f"Timeout waiting for camera_info on {self._camera_info_topic}")

        # Publishers
        self._detections_2d_pub = self._node.create_publisher(
            ROSDetection2DArray,
            self._detections_2d_topic,
            10,
        )
        self._detections_3d_pub = self._node.create_publisher(
            ROSDetection3DArray,
            self._detections_3d_topic,
            10,
        )
        self._overlay_pub = self._node.create_publisher(
            ROSImage,
            self._overlay_topic,
            10,
        )
        self._pointcloud_pub = self._node.create_publisher(
            ROSPointCloud2,
            self._pointcloud_topic,
            10,
        )
        self._mesh_markers_pub = self._node.create_publisher(
            MarkerArray,
            self._mesh_marker_topic,
            10,
        )

        # Set up synchronized subscribers for color and depth
        self._color_sub = message_filters.Subscriber(
            self._node,
            ROSCompressedImage,
            self._image_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._depth_sub = message_filters.Subscriber(
            self._node,
            ROSCompressedImage,
            self._depth_topic,
            qos_profile=qos_profile_sensor_data,
        )

        # Synchronize color and depth with approximate time sync
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub],
            queue_size=10,
            slop=0.1,  # 100ms tolerance
        )
        self._sync.registerCallback(self._on_synced_images)

        logger.info("Synchronized subscribers:")
        logger.info(f"  Color: {self._image_topic}")
        logger.info(f"  Depth: {self._depth_topic}")
        logger.info("Publishing:")
        logger.info(f"  2D detections: {self._detections_2d_topic}")
        logger.info(f"  3D detections: {self._detections_3d_topic}")
        logger.info(f"  Overlay: {self._overlay_topic}")
        logger.info(f"  Pointcloud: {self._pointcloud_topic}")
        logger.info(f"  Mesh markers: {self._mesh_marker_topic}")

        # Start spin thread
        self._spin_thread = threading.Thread(
            target=self._spin_node,
            daemon=True,
            name="ObjectSceneRegistrationSpinThread",
        )
        self._spin_thread.start()

        # Start processing thread (keeps callback fast, processes in background)
        self._processing_queue = Queue(maxsize=1)
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="ObjectSceneRegistrationProcessingThread",
        )
        self._processing_thread.start()
        
        # Start mesh worker thread (async mesh generation)
        if self._mesh_pose_client is not None and self._auto_mesh_pose:
            self._mesh_worker_thread = threading.Thread(
                target=self._mesh_worker_loop,
                daemon=True,
                name="ObjectSceneRegistrationMeshWorkerThread",
            )
            self._mesh_worker_thread.start()
            logger.info("Started async mesh worker thread")

    def _spin_node(self) -> None:
        """Spin the ROS node."""
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def _processing_loop(self) -> None:
        """Background thread for heavy image processing."""
        while self._running:
            try:
                color_msg, depth_msg = self._processing_queue.get(timeout=0.5)
                self._process_images(color_msg, depth_msg)
            except Empty:
                continue
            except Exception:
                logger.exception("Error processing images")

    def _mesh_worker_loop(self) -> None:
        """Background thread for async mesh generation requests."""
        while self._running:
            try:
                object_id, color_image, depth_image, bbox = self._mesh_request_queue.get(
                    timeout=1.0
                )
            except Empty:
                continue
            
            try:
                result = self._mesh_pose_client.get_mesh_and_pose(
                    bbox=bbox,
                    color_image=color_image,
                    depth_image=depth_image,
                    camera_info=self._camera_info,
                    use_box_prompt=self._mesh_pose_use_box_prompt,
                )

                mesh_obj = result.get("mesh_obj")
                mesh_path_str: str | None = None

                # Save mesh to persistent folder (repo-local)
                if isinstance(mesh_obj, (bytes, bytearray)) and len(mesh_obj) > 0:
                    mesh_dir = Path(self._mesh_store_dir)
                    mesh_dir.mkdir(parents=True, exist_ok=True)

                    # Sanitize object name for filename
                    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", "object").strip("_")
                    if self._object_db is not None:
                        with self._object_db._lock:
                            obj_for_name = self._object_db.get_by_object_id(object_id)
                            if obj_for_name is not None and obj_for_name.name:
                                safe_name = re.sub(
                                    r"[^A-Za-z0-9._-]+", "_", obj_for_name.name
                                ).strip("_") or safe_name

                    mesh_file = mesh_dir / f"{safe_name}_{object_id}.obj"
                    mesh_file.write_bytes(mesh_obj)
                    mesh_path_str = str(mesh_file.resolve())

                # Update object in DB
                if self._object_db:
                    with self._object_db._lock:
                        obj = self._object_db.get_by_object_id(object_id)
                        if obj is not None:
                            obj.mesh_obj = mesh_obj
                            obj.mesh_path = mesh_path_str
                            obj.mesh_dimensions = result.get("mesh_dimensions")
                            obj.fp_position = result.get("fp_position")
                            obj.fp_orientation = result.get("fp_orientation")
                
                self._mesh_request_states[object_id] = "DONE"
                logger.info(f"Mesh complete for object_id={object_id}")
                
            except Exception as e:
                logger.warning(f"Mesh generation failed for object_id={object_id}: {e}")
                self._mesh_request_states[object_id] = "FAILED"

    @rpc
    def stop(self) -> None:
        """Stop the module and clean up resources."""
        self._running = False

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
        
        if self._mesh_worker_thread and self._mesh_worker_thread.is_alive():
            self._mesh_worker_thread.join(timeout=2.0)

        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)

        if self._detector:
            self._detector.stop()
            self._detector = None

        if self._mesh_pose_client:
            self._mesh_pose_client.close()
            self._mesh_pose_client = None

        if self._node:
            self._node.destroy_node()
            self._node = None

        if self._object_db:
            self._object_db.clear()
            self._object_db = None

        logger.info("ObjectSceneRegistrationModule stopped")
        super().stop()

    @property
    def object_db(self) -> ObjectDB | None:
        """Access the ObjectDB for querying detected objects."""
        return self._object_db

    @rpc
    def set_auto_mesh_pose(self, enabled: bool) -> None:
        """Turn auto /process enhancement on/off (interactive sessions usually want this OFF)."""
        self._auto_mesh_pose = enabled
        logger.info(f"Auto mesh/pose enhancement: {'enabled' if enabled else 'disabled'}")

    @rpc
    def get_latest_detections(self) -> list[dict]:
        """Get latest detections for interactive selection (index/track_id/bbox/confidence)."""
        with self._latest_lock:
            dets = self._latest_detections_2d.detections if self._latest_detections_2d else []
        return [
            {
                "index": i,
                "track_id": int(getattr(d, "track_id", -1)),
                "name": str(getattr(d, "name", "")),
                "confidence": float(getattr(d, "confidence", 0.0)),
                "bbox": [float(x) for x in d.bbox],
            }
            for i, d in enumerate(dets)
        ]

    @rpc
    def run_hosted_pipeline(
        self,
        pipeline: str = "process",
        *,
        track_id: int | None = None,
        index: int | None = None,
        prompt: str | None = None,
        use_box_prompt: bool | None = None,
        gripper_type: str = "robotiq_2f_140",
        filter_collisions: bool = True,
    ) -> dict:
        """
        Interactive trigger for hosted service pipelines.

        Args:
            pipeline: "process" (full: mesh+pose+grasps) or "grasp" (fast: grasps only)
            track_id: Select detection by YOLO-E track_id
            index: Select detection by list index
            prompt: Text prompt for SAM3 (if use_box_prompt=False)
            use_box_prompt: Force box-only or text prompting (default: auto)
            gripper_type: Gripper type for grasp generation
            filter_collisions: Filter colliding grasps

        Returns:
            Result dict from hosted service (/process or /grasp endpoint)

        Example:
            # Box-only prompting for top detection (default)
            result = module.run_hosted_pipeline("process")

            # Text prompting with custom label
            result = module.run_hosted_pipeline("process", prompt="red coffee mug", use_box_prompt=False)

            # Fast grasp-only pipeline
            result = module.run_hosted_pipeline("grasp", track_id=5)
        """
        if self._mesh_pose_client is None:
            raise RuntimeError("mesh_pose_service_url not set (no hosted service client)")
        if self._camera_info is None:
            raise RuntimeError("No camera_info yet")

        with self._latest_lock:
            color_image = self._latest_color_image
            depth_image = self._latest_depth_image
            detections_2d = self._latest_detections_2d
            objects = list(self._latest_objects)

        if color_image is None or depth_image is None or detections_2d is None:
            raise RuntimeError("No latest RGBD/detections available yet")

        dets = detections_2d.detections
        if not dets:
            raise RuntimeError("No detections in latest frame")

        # Choose detection
        selected = None
        if track_id is not None:
            selected = next((d for d in dets if getattr(d, "track_id", -1) == track_id), None)
            if selected is None:
                raise ValueError(f"track_id {track_id} not found")
        elif index is not None:
            if index < 0 or index >= len(dets):
                raise ValueError(f"index out of range: {index}")
            selected = dets[index]
        else:
            selected = max(dets, key=lambda d: float(getattr(d, "confidence", 0.0)))

        bbox = tuple(float(x) for x in selected.bbox)
        selected_track_id = int(getattr(selected, "track_id", -1))

        prompt_str = (prompt or "").strip()
        if use_box_prompt is None:
            use_box_prompt = (prompt_str == "")

        if pipeline == "process":
            result = self._mesh_pose_client.get_mesh_and_pose(
                bbox=bbox,
                color_image=color_image,
                depth_image=depth_image,
                camera_info=self._camera_info,
                use_box_prompt=use_box_prompt,
                label=None if use_box_prompt else prompt_str,
            )

            # Update in-memory Object instance (so it shows up in published 3D detections)
            mesh_obj = result.get("mesh_obj")
            mesh_bytes = len(mesh_obj) if isinstance(mesh_obj, (bytes, bytearray)) else 0

            if selected_track_id >= 0:
                for obj in objects:
                    if obj.track_id == selected_track_id:
                        obj.mesh_obj = mesh_obj
                        obj.mesh_dimensions = result.get("mesh_dimensions")
                        obj.fp_position = result.get("fp_position")
                        obj.fp_orientation = result.get("fp_orientation")
                        break

            # Don't return raw mesh bytes over LCM RPC (can be huge)
            result["mesh_obj"] = None
            result["mesh_obj_bytes"] = mesh_bytes
            return result

        if pipeline == "grasp":
            return self._mesh_pose_client.get_grasps(
                bbox=bbox,
                color_image=color_image,
                depth_image=depth_image,
                camera_info=self._camera_info,
                use_box_prompt=use_box_prompt,
                label=None if use_box_prompt else prompt_str,
                filter_collisions=filter_collisions,
                gripper_type=gripper_type,
            )

        raise ValueError("pipeline must be 'process' or 'grasp'")

    def _convert_compressed_depth_image(self, msg: ROSCompressedImage) -> Image | None:
        """Convert ROS compressedDepth image to internal Image type."""
        if not self._bridge:
            return None

        try:
            import cv2

            # compressedDepth format has a 12-byte header followed by PNG data
            # cv_bridge doesn't handle this format correctly
            if "compressedDepth" in msg.format:
                depth_header_size = 12
                raw_data = np.frombuffer(msg.data, dtype=np.uint8)

                if len(raw_data) <= depth_header_size:
                    logger.warning("compressedDepth data too small")
                    return None

                # Decode the PNG portion (after header)
                png_data = raw_data[depth_header_size:]
                depth_cv = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)

                if depth_cv is None:
                    logger.warning("Failed to decode compressedDepth PNG data")
                    return None
            else:
                # Regular compressed image
                depth_cv = self._bridge.compressed_imgmsg_to_cv2(msg)

            # Convert to meters if uint16 (assuming mm depth)
            if depth_cv.dtype == np.uint16:
                depth_cv = depth_cv.astype(np.float32) / 1000.0
            elif depth_cv.dtype != np.float32:
                depth_cv = depth_cv.astype(np.float32)

            depth_image = Image.from_numpy(depth_cv)
            depth_image.from_ros_header(msg.header)
            return depth_image
        except Exception as e:
            logger.warning(f"Failed to convert compressed depth image: {e}")
            return None

    def _on_synced_images(
        self, color_msg: ROSCompressedImage, depth_msg: ROSCompressedImage
    ) -> None:
        """Queue synchronized images for processing (fast callback)."""
        if not self._processing_queue:
            return

        # Drop old data if queue is full (we always want the latest)
        if self._processing_queue.full():
            try:
                self._processing_queue.get_nowait()
            except Exception:
                pass

        self._processing_queue.put((color_msg, depth_msg))

    def _process_images(self, color_msg: ROSCompressedImage, depth_msg: ROSCompressedImage) -> None:
        """Process synchronized color and depth images (runs in background thread)."""
        if not self._detector or not self._bridge or not self._camera_info:
            return

        # Convert color image
        cv_image = self._bridge.compressed_imgmsg_to_cv2(color_msg, "rgb8")
        color_image = Image.from_numpy(cv_image)
        color_image.from_ros_header(color_msg.header)

        # Convert compressed depth image
        depth_image = self._convert_compressed_depth_image(depth_msg)
        if depth_image is None:
            return

        # Run 2D detection
        detections_2d: ImageDetections2D = self._detector.process_image(color_image)

        # Store latest data for interactive queries
        with self._latest_lock:
            self._latest_color_image = color_image
            self._latest_depth_image = depth_image
            self._latest_detections_2d = detections_2d

        ros_detections_2d = detections_2d.to_ros_detection2d_array()
        self._detections_2d_pub.publish(ros_detections_2d)

        # Publish overlay image
        overlay_image = detections_2d.overlay()
        overlay_msg = self._bridge.cv2_to_imgmsg(overlay_image.to_opencv(), encoding="rgb8")
        overlay_msg.header = color_msg.header
        self._overlay_pub.publish(overlay_msg)

        # Process 3D detections
        self._process_3d_detections(detections_2d, color_image, depth_image, color_msg.header)

    def _process_3d_detections(
        self,
        detections_2d: ImageDetections2D,
        color_image: Image,
        depth_image: Image,
        header: ROSHeader,
    ) -> None:
        """Convert 2D detections to 3D and publish."""
        if self._camera_info is None:
            return

        # Look up transform from camera frame to target frame (e.g., map)
        if self._tf_buffer is not None:
            try:
                ros_transform = self._tf_buffer.lookup_transform(
                    self._target_frame,
                    color_image.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.4),
                )
                camera_transform = Transform.from_ros_transform_stamped(ros_transform)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                logger.warning("Failed to lookup transform from camera frame to target frame")
                return

        objects = Object.from_2d(
            detections_2d=detections_2d,
            color_image=color_image,
            depth_image=depth_image,
            camera_info=self._camera_info,
            camera_transform=camera_transform,
        )

        if not objects:
            return

        # Add objects to spatial memory database FIRST (deduplication happens here)
        # This returns matched/updated objects so we can skip mesh/pose for known objects
        if self._object_db is not None:
            objects = self._object_db.add_objects(objects)

        # Store latest objects for interactive queries
        with self._latest_lock:
            self._latest_objects = list(objects)

        # Optionally enhance objects with mesh and accurate 6D pose from hosted service
        # Uses async worker thread to avoid blocking the detection loop
        # Default now: use box-only prompting, do NOT forward YOLO-E labels
        if self._mesh_pose_client is not None and self._auto_mesh_pose:
            for obj in objects:
                # Only queue mesh jobs for permanent objects (deduped/stable)
                if self._object_db is None or not self._object_db.is_permanent(obj.object_id):
                    continue
                if obj.has_mesh:
                    continue  # Skip - already has mesh from previous detection
                
                status = self._mesh_request_states.get(obj.object_id)
                if status in (None,):  # Only queue if never requested
                    self._mesh_request_states[obj.object_id] = "PENDING"
                    self._mesh_request_queue.put((
                        obj.object_id,
                        color_image,
                        depth_image,
                        obj.bbox,
                    ))

        detections_3d = to_detection3d_array(objects)
        ros_detections_3d = detections_3d.to_ros_msg()
        self._detections_3d_pub.publish(ros_detections_3d)

        aggregated_pc = aggregate_pointclouds(self._object_db.get_objects())
        if aggregated_pc is not None:
            ros_pc = aggregated_pc.to_ros_msg()
            self._pointcloud_pub.publish(ros_pc)

        # Publish mesh markers for RViz (permanent objects with mesh_path)
        if self._mesh_markers_pub is not None and self._object_db is not None:
            self._publish_mesh_markers(camera_transform=camera_transform)

    def _publish_mesh_markers(self, camera_transform: Transform) -> None:
        """Publish RViz mesh markers for permanent objects with saved mesh files."""
        if self._object_db is None:
            return

        objs = self._object_db.get_objects()
        marker_array = MarkerArray()

        # Clear previous markers each publish (simplest, avoids stale meshes)
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for obj in objs:
            if not obj.mesh_path:
                continue

            # Determine pose in target_frame (map)
            if obj.fp_position is not None and obj.fp_orientation is not None:
                T_camera_object = Transform(
                    translation=Vector3(*obj.fp_position),
                    rotation=Quaternion(*obj.fp_orientation),
                    frame_id=camera_transform.child_frame_id,
                    child_frame_id=obj.object_id,
                    ts=obj.ts,
                )
                T_map_object = camera_transform + T_camera_object
                pos = T_map_object.translation
                rot = T_map_object.rotation
                frame_id = camera_transform.frame_id
            elif obj.center is not None:
                pos = obj.center
                rot = Quaternion(0.0, 0.0, 0.0, 1.0)
                frame_id = obj.frame_id
            else:
                continue

            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = self._node.get_clock().now().to_msg()
            m.ns = "object_meshes"
            try:
                m.id = int(obj.object_id, 16) % (2**31 - 1)
            except Exception:
                m.id = abs(hash(obj.object_id)) % (2**31 - 1)
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.mesh_resource = Path(obj.mesh_path).resolve().as_uri()
            m.pose.position.x = float(pos.x)
            m.pose.position.y = float(pos.y)
            m.pose.position.z = float(pos.z)
            m.pose.orientation.x = float(rot.x)
            m.pose.orientation.y = float(rot.y)
            m.pose.orientation.z = float(rot.z)
            m.pose.orientation.w = float(rot.w)
            m.scale.x = 1.0
            m.scale.y = 1.0
            m.scale.z = 1.0
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 1.0
            marker_array.markers.append(m)

        self._mesh_markers_pub.publish(marker_array)


object_scene_registration_module = ObjectSceneRegistrationModule.blueprint

__all__ = ["ObjectSceneRegistrationModule", "object_scene_registration_module"]
