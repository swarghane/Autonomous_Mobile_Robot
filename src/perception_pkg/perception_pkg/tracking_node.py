import numpy as np
import rclpy
from filterpy.kalman import KalmanFilter
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scipy.optimize import linear_sum_assignment
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class Track:
    """Maintain one tracked object using an 8-state Kalman filter."""

    def __init__(
        self,
        bbox,
        class_name,
        score,
        track_id,
        covariance_scale=10.0,
        measurement_noise=1.0,
        process_noise=0.01,
    ):
        self.id = track_id
        self.class_name = class_name
        self.score = score
        self.hits = 1
        self.age = 0

        self.covariance_scale = covariance_scale
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.kf = self.create_kf(bbox)

    def create_kf(self, bbox):
        """Create a constant-velocity Kalman filter for the bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State: [cx, cy, width, height, vx, vy, vwidth, vheight].
        kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurements provide only box center and size, not velocity.
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        kf.P *= self.covariance_scale
        kf.R *= self.measurement_noise
        kf.Q *= self.process_noise
        kf.x = np.array(
            [
                [cx],
                [cy],
                [width],
                [height],
                [0],
                [0],
                [0],
                [0],
            ]
        )

        return kf

    def predict(self):
        """Predict the next bounding-box state and age the track."""
        self.kf.predict()
        self.age += 1

    def update(self, bbox, score):
        """Correct the prediction using a matched detection."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        measurement = np.array([[cx], [cy], [width], [height]])
        self.kf.update(measurement)
        self.score = score
        self.age = 0
        self.hits += 1

    def bbox(self):
        """Return the current Kalman state as [x1, y1, x2, y2]."""
        cx = self.kf.x[0][0]
        cy = self.kf.x[1][0]
        width = self.kf.x[2][0]
        height = self.kf.x[3][0]

        return [
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        ]


class TrackerNode(Node):
    """Associate detections across frames and publish stable track IDs."""

    def __init__(self):
        super().__init__('tracker_node')

        # ROS topics and tracking thresholds.
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter(
            'tracked_detections_topic',
            '/tracked_detections',
        )
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('max_age', 30)
        self.declare_parameter('min_hits', 4)
        self.declare_parameter('max_publish_age', 2)

        # Kalman-filter tuning values.
        self.declare_parameter('kalman_covariance_scale', 10.0)
        self.declare_parameter('kalman_measurement_noise', 1.0)
        self.declare_parameter('kalman_process_noise', 0.01)

        self.detections_topic = str(
            self.get_parameter('detections_topic').value
        )
        self.tracked_detections_topic = str(
            self.get_parameter('tracked_detections_topic').value
        )
        self.iou_threshold = float(
            self.get_parameter('iou_threshold').value
        )
        self.max_age = int(self.get_parameter('max_age').value)
        self.min_hits = int(self.get_parameter('min_hits').value)
        self.max_publish_age = int(
            self.get_parameter('max_publish_age').value
        )
        self.kalman_covariance_scale = float(
            self.get_parameter('kalman_covariance_scale').value
        )
        self.kalman_measurement_noise = float(
            self.get_parameter('kalman_measurement_noise').value
        )
        self.kalman_process_noise = float(
            self.get_parameter('kalman_process_noise').value
        )

        self.tracks = []
        self.next_track_id = 1

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.detections_sub = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detection_callback,
            qos,
        )
        self.tracked_detections_pub = self.create_publisher(
            Detection2DArray,
            self.tracked_detections_topic,
            qos,
        )

        self.get_logger().info('Real SORT tracker started')

    @staticmethod
    def det_to_xyxy(det):
        """Convert a Detection2D center-size box to corner coordinates."""
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        width = det.bbox.size_x
        height = det.bbox.size_y

        return [
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        ]

    @staticmethod
    def iou(a, b):
        """Calculate intersection over union for two corner-format boxes."""
        x_left = max(a[0], b[0])
        y_top = max(a[1], b[1])
        x_right = min(a[2], b[2])
        y_bottom = min(a[3], b[3])

        intersection = max(0, x_right - x_left) * max(
            0,
            y_bottom - y_top,
        )

        if intersection == 0:
            return 0.0

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])

        return intersection / (area_a + area_b - intersection)

    def assign(self, detections):
        """Match tracks to detections using IoU and Hungarian assignment."""
        if not self.tracks:
            return [], [], list(range(len(detections)))

        cost = np.ones((len(self.tracks), len(detections)))

        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_name == detection['class_name']:
                    cost[track_index, detection_index] = 1 - self.iou(
                        track.bbox(),
                        detection['bbox'],
                    )

        rows, columns = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        for row, column in zip(rows, columns):
            if 1 - cost[row, column] < self.iou_threshold:
                continue

            matches.append((row, column))

            if row in unmatched_tracks:
                unmatched_tracks.remove(row)

            if column in unmatched_detections:
                unmatched_detections.remove(column)

        return matches, unmatched_tracks, unmatched_detections

    def detection_callback(self, msg: Detection2DArray):
        """Update all tracks from one incoming detection array."""
        detections = []

        for detection in msg.detections:
            if not detection.results:
                continue

            detections.append(
                {
                    'bbox': self.det_to_xyxy(detection),
                    'class_name': detection.results[0].hypothesis.class_id,
                    'score': detection.results[0].hypothesis.score,
                }
            )

        for track in self.tracks:
            track.predict()

        matches, _, unmatched_detections = self.assign(detections)

        for track_index, detection_index in matches:
            detection = detections[detection_index]
            self.tracks[track_index].update(
                detection['bbox'],
                detection['score'],
            )

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            self.tracks.append(
                Track(
                    detection['bbox'],
                    detection['class_name'],
                    detection['score'],
                    self.next_track_id,
                    covariance_scale=self.kalman_covariance_scale,
                    measurement_noise=self.kalman_measurement_noise,
                    process_noise=self.kalman_process_noise,
                )
            )
            self.next_track_id += 1

        self.tracks = [
            track for track in self.tracks if track.age <= self.max_age
        ]

        output = Detection2DArray()
        output.header = msg.header

        for track in self.tracks:
            if track.hits < self.min_hits:
                continue

            if track.age > self.max_publish_age:
                continue

            x1, y1, x2, y2 = track.bbox()

            detection = Detection2D()
            detection.header = msg.header
            detection.id = str(track.id)
            detection.bbox.center.position.x = (x1 + x2) / 2
            detection.bbox.center.position.y = (y1 + y2) / 2
            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = track.class_name
            hypothesis.hypothesis.score = track.score

            detection.results.append(hypothesis)
            output.detections.append(detection)

        self.tracked_detections_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info('Tracker node stopping...')
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
