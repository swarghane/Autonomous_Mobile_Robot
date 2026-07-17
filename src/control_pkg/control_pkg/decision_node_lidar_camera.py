import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Bool, Float32, String
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy,DurabilityPolicy
import time


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        self.last_target_time = time.time()
        self.target_timeout = 2.0

        self.obstacle_start_time = None

        self.frame_width=640
        self.center_x=self.frame_width//2

        self.turn_threshold=60
        self.stop_width=450

        self.target_id=None
        self.search_direction=0.3

        self.obstacle_detected=False
        self.front_distance=999.0

        self.free_direction = 'LEFT'


        qos=QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.detection_sub=self.create_subscription(
            Detection2DArray,
            '/tracked_detections',
            self.detection_callback,
            qos
        )

        self.obstacle_sub=self.create_subscription(
            Bool,
            '/obstacle_detected',
            self.obstacle_callback,
            qos
        )

        self.distance_sub=self.create_subscription(
            Float32,
            '/front_distance',
            self.distance_callback,
            qos
        )

        self.direction_sub = self.create_subscription(
            String,
            '/free_direction',
            self.direction_callback,
            qos
        )

        self.pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos
        )

        self.get_logger().info('Decision node started')

    def obstacle_callback(self,msg):
        self.obstacle_detected=msg.data

    def distance_callback(self,msg):
        self.front_distance=msg.data

    def direction_callback(self,msg):
        self.free_direction = msg.data

    def detection_callback(self,msg):
        twist=Twist()
        
        if self.obstacle_detected:
            if self.obstacle_start_time is None:
                self.obstacle_start_time = time.time()

            elapsed = time.time() - self.obstacle_start_time

            if elapsed < 1.0:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                if self.free_direction == 'LEFT':
                    twist.angular.z = 0.8
                else:
                    twist.angular.z = -0.8

            self.pub.publish(twist)

            self.get_logger().warn(
                f'Obstacle detected | Distance: '
                f'{self.front_distance:.2f} m'
            )

            return
        else:
            self.obstacle_start_time = None

        person_detections=[]

        for det in msg.detections:
            if len(det.results)==0:
                continue

            class_name=det.results[0].hypothesis.class_id

            if class_name=='person':
                person_detections.append(det)

        if len(person_detections) == 0:
            elapsed = time.time() - self.last_target_time
            if elapsed < self.target_timeout:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.pub.publish(twist)

                self.get_logger().info(
                    'Temporary target loss'
                )
                return

            twist.linear.x = 0.4
            twist.angular.z = 0.15

            self.pub.publish(twist)

            self.get_logger().info(
                'Exploring environment'
            )
            return

        target=None

        if self.target_id is not None:
            for det in person_detections:
                if det.id==self.target_id:
                    target=det
                    break

        if target is None:
            target=max(
                person_detections,
                key=lambda d:d.bbox.size_x*d.bbox.size_y
            )

            self.target_id=target.id
            
        self.last_target_time = time.time()
        cx=target.bbox.center.position.x
        width=target.bbox.size_x

        error=cx-self.center_x

        if width > self.stop_width:
            twist.linear.x=0.0
            twist.angular.z=0.0

            self.get_logger().info(
                f'Target close stop | ID:{self.target_id}'
            )

        elif width > 250:
            twist.linear.x=0.08
            twist.angular.z=-error*0.002

        else:
            twist.linear.x=0.5
            twist.angular.z=-error*0.003

        if abs(error) < 25:
            twist.angular.z=0.0

        self.get_logger().info(
            f'Following target | '
            f'Error:{error} | '
            f'Distance:{self.front_distance:.2f} m'
        )

        self.pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node=DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__=='__main__':
    main()