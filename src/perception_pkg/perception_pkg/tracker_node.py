import numpy as np
import rclpy
from rclpy.node import Node
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from vision_msgs.msg import Detection2DArray,Detection2D,ObjectHypothesisWithPose
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy,DurabilityPolicy

class Track:
    def __init__(self,bbox,class_name,score,track_id):
        self.id=track_id
        self.class_name=class_name
        self.score=score
        self.hits=1
        self.age=0
        self.kf=self.create_kf(bbox)

    def create_kf(self,bbox):
        x1,y1,x2,y2=bbox
        cx=(x1+x2)/2
        cy=(y1+y2)/2
        w=x2-x1
        h=y2-y1
        kf=KalmanFilter(dim_x=8,dim_z=4)
        kf.F=np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ])
        kf.H=np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ])
        kf.P*=10.
        kf.R*=1.
        kf.Q*=0.01
        kf.x=np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]])
        return kf

    def predict(self):
        self.kf.predict()
        self.age+=1

    def update(self,bbox,score):
        x1,y1,x2,y2=bbox
        cx=(x1+x2)/2
        cy=(y1+y2)/2
        w=x2-x1
        h=y2-y1
        z=np.array([[cx],[cy],[w],[h]])
        self.kf.update(z)
        self.score=score
        self.age=0
        self.hits+=1

    def bbox(self):
        cx=self.kf.x[0][0]
        cy=self.kf.x[1][0]
        w=self.kf.x[2][0]
        h=self.kf.x[3][0]
        return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.declare_parameter("iou_threshold",0.3)
        self.declare_parameter("max_age",15)
        self.declare_parameter("min_hits",2)
        self.iou_threshold=float(self.get_parameter("iou_threshold").value)
        self.max_age=int(self.get_parameter("max_age").value)
        self.min_hits=int(self.get_parameter("min_hits").value)
        self.tracks=[]
        self.next_track_id=1
        qos=QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        self.sub=self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            qos
        )
        self.pub=self.create_publisher(
            Detection2DArray,
            '/tracked_detections',
            qos
        )
        self.get_logger().info('Real SORT tracker started')

    def det_to_xyxy(self,det):
        cx=det.bbox.center.position.x
        cy=det.bbox.center.position.y
        w=det.bbox.size_x
        h=det.bbox.size_y
        return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

    def iou(self,a,b):
        xA=max(a[0],b[0])
        yA=max(a[1],b[1])
        xB=min(a[2],b[2])
        yB=min(a[3],b[3])
        inter=max(0,xB-xA)*max(0,yB-yA)
        if inter==0:
            return 0.0
        areaA=(a[2]-a[0])*(a[3]-a[1])
        areaB=(b[2]-b[0])*(b[3]-b[1])
        return inter/(areaA+areaB-inter)

    def assign(self,detections):
        if len(self.tracks)==0:
            return [],[],list(range(len(detections)))
        cost=np.ones((len(self.tracks),len(detections)))
        for i,t in enumerate(self.tracks):
            for j,d in enumerate(detections):
                if t.class_name!=d["class_name"]:
                    cost[i,j]=1.0
                else:
                    cost[i,j]=1-self.iou(t.bbox(),d["bbox"])
        rows,cols=linear_sum_assignment(cost)
        matches=[]
        unmatched_tracks=list(range(len(self.tracks)))
        unmatched_dets=list(range(len(detections)))
        for r,c in zip(rows,cols):
            if 1-cost[r,c] < self.iou_threshold:
                continue
            matches.append((r,c))
            if r in unmatched_tracks:
                unmatched_tracks.remove(r)
            if c in unmatched_dets:
                unmatched_dets.remove(c)
        return matches,unmatched_tracks,unmatched_dets

    def detection_callback(self,msg):
        detections=[]
        for det in msg.detections:
            if len(det.results)==0:
                continue
            detections.append({
                "bbox":self.det_to_xyxy(det),
                "class_name":det.results[0].hypothesis.class_id,
                "score":det.results[0].hypothesis.score
            })

        for t in self.tracks:
            t.predict()

        matches,unmatched_tracks,unmatched_dets=self.assign(detections)

        for t_idx,d_idx in matches:
            d=detections[d_idx]
            self.tracks[t_idx].update(d["bbox"],d["score"])

        for d_idx in unmatched_dets:
            d=detections[d_idx]
            self.tracks.append(
                Track(
                    d["bbox"],
                    d["class_name"],
                    d["score"],
                    self.next_track_id
                )
            )
            self.next_track_id+=1

        self.tracks=[
            t for t in self.tracks
            if t.age<=self.max_age
        ]

        out=Detection2DArray()
        out.header=msg.header

        for t in self.tracks:
            if t.hits<self.min_hits:
                continue
            x1,y1,x2,y2=t.bbox()
            det=Detection2D()
            det.bbox.center.position.x=(x1+x2)/2
            det.bbox.center.position.y=(y1+y2)/2
            det.bbox.size_x=x2-x1
            det.bbox.size_y=y2-y1
            hyp=ObjectHypothesisWithPose()
            hyp.hypothesis.class_id=f'{t.class_name} [ID {t.id}]'
            hyp.hypothesis.score=t.score
            det.results.append(hyp)
            out.detections.append(det)

        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node=TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__=="__main__":
    main()




# import rclpy
# from rclpy.node import Node

# from vision_msgs.msg import (
#     Detection2DArray,
#     Detection2D,
#     ObjectHypothesisWithPose
# )

# from rclpy.qos import QoSProfile
# from rclpy.qos import ReliabilityPolicy
# from rclpy.qos import HistoryPolicy
# from rclpy.qos import DurabilityPolicy


# class TrackerNode(Node):

#     def __init__(self):
#         super().__init__('tracker_node')

#         # ---------------------------------
#         # Parameters
#         # ---------------------------------
#         self.declare_parameter("iou_threshold", 0.3)
#         self.declare_parameter("max_age", 10)
#         self.declare_parameter("min_hits", 3)

#         self.iou_threshold = float(
#             self.get_parameter(
#                 "iou_threshold"
#             ).value
#         )

#         self.max_age = int(
#             self.get_parameter(
#                 "max_age"
#             ).value
#         )

#         self.min_hits = int(
#             self.get_parameter(
#                 "min_hits"
#             ).value
#         )

#         # ---------------------------------
#         # Track state
#         # ---------------------------------
#         self.tracks = []
#         self.next_track_id = 1

#         qos = QoSProfile(
#             history=HistoryPolicy.KEEP_LAST,
#             depth=10,
#             reliability=ReliabilityPolicy.RELIABLE,
#             durability=DurabilityPolicy.VOLATILE
#         )

#         self.sub = self.create_subscription(
#             Detection2DArray,
#             '/detections',
#             self.detection_callback,
#             qos
#         )

#         self.pub = self.create_publisher(
#             Detection2DArray,
#             '/tracked_detections',
#             qos
#         )

#         self.get_logger().info(
#             '✅ SORT-lite tracker started'
#         )

#     # -------------------------------
#     # Convert bbox
#     # -------------------------------
#     def det_to_xyxy(self, det):

#         cx = det.bbox.center.position.x
#         cy = det.bbox.center.position.y

#         w = det.bbox.size_x
#         h = det.bbox.size_y

#         x1 = cx - w/2
#         y1 = cy - h/2

#         x2 = cx + w/2
#         y2 = cy + h/2

#         return [
#             x1, y1,
#             x2, y2
#         ]

#     # -------------------------------
#     # IoU
#     # -------------------------------
#     def iou(self, boxA, boxB):

#         xA = max(
#             boxA[0],
#             boxB[0]
#         )

#         yA = max(
#             boxA[1],
#             boxB[1]
#         )

#         xB = min(
#             boxA[2],
#             boxB[2]
#         )

#         yB = min(
#             boxA[3],
#             boxB[3]
#         )

#         inter_w = max(
#             0,
#             xB - xA
#         )

#         inter_h = max(
#             0,
#             yB - yA
#         )

#         inter = (
#             inter_w *
#             inter_h
#         )

#         if inter == 0:
#             return 0.0

#         areaA = (
#             (boxA[2]-boxA[0]) *
#             (boxA[3]-boxA[1])
#         )

#         areaB = (
#             (boxB[2]-boxB[0]) *
#             (boxB[3]-boxB[1])
#         )

#         return inter / (
#             areaA +
#             areaB -
#             inter
#         )

#     # -------------------------------
#     # Greedy assignment
#     # (Hungarian-like approximation)
#     # -------------------------------
#     def assign(
#         self,
#         detections
#     ):

#         matches = []

#         unmatched_tracks = list(
#             range(len(self.tracks))
#         )

#         unmatched_dets = list(
#             range(len(detections))
#         )

#         pairs = []

#         for t_idx, track in enumerate(self.tracks):

#             for d_idx, det in enumerate(detections):

#                 if (
#                     track["class_name"]
#                     !=
#                     det["class_name"]
#                 ):
#                     continue

#                 score = self.iou(
#                     track["bbox"],
#                     det["bbox"]
#                 )

#                 if (
#                     score >
#                     self.iou_threshold
#                 ):
#                     pairs.append(
#                         (
#                             score,
#                             t_idx,
#                             d_idx
#                         )
#                     )

#         # highest IoU first
#         pairs.sort(
#             reverse=True
#         )

#         used_tracks = set()
#         used_dets = set()

#         for score, t_idx, d_idx in pairs:

#             if (
#                 t_idx in used_tracks
#                 or
#                 d_idx in used_dets
#             ):
#                 continue

#             matches.append(
#                 (
#                     t_idx,
#                     d_idx
#                 )
#             )

#             used_tracks.add(
#                 t_idx
#             )

#             used_dets.add(
#                 d_idx
#             )

#         unmatched_tracks = [
#             i for i in range(
#                 len(self.tracks)
#             )
#             if i not in used_tracks
#         ]

#         unmatched_dets = [
#             i for i in range(
#                 len(detections)
#             )
#             if i not in used_dets
#         ]

#         return (
#             matches,
#             unmatched_tracks,
#             unmatched_dets
#         )

#     # -------------------------------
#     # Callback
#     # -------------------------------
#     def detection_callback(
#         self,
#         msg
#     ):

#         detections = []

#         for det in msg.detections:

#             if len(det.results)==0:
#                 continue

#             detections.append({

#                 "bbox":
#                     self.det_to_xyxy(det),

#                 "class_name":
#                     det.results[0]
#                     .hypothesis
#                     .class_id,

#                 "score":
#                     det.results[0]
#                     .hypothesis
#                     .score
#             })

#         (
#             matches,
#             unmatched_tracks,
#             unmatched_dets

#         ) = self.assign(
#             detections
#         )

#         # -------------------------
#         # Update matched tracks
#         # -------------------------
#         for (
#             t_idx,
#             d_idx

#         ) in matches:

#             det = detections[d_idx]

#             track = self.tracks[t_idx]

#             track["bbox"] = (
#                 det["bbox"]
#             )

#             track["score"] = (
#                 det["score"]
#             )

#             track["age"] = 0

#             track["hits"] += 1

#         # -------------------------
#         # Age unmatched tracks
#         # -------------------------
#         for t_idx in unmatched_tracks:

#             self.tracks[t_idx]["age"] += 1

#         # -------------------------
#         # Create new tracks
#         # -------------------------
#         for d_idx in unmatched_dets:

#             det = detections[d_idx]

#             self.tracks.append({

#                 "id":
#                     self.next_track_id,

#                 "bbox":
#                     det["bbox"],

#                 "class_name":
#                     det["class_name"],

#                 "score":
#                     det["score"],

#                 "age":
#                     0,

#                 "hits":
#                     1
#             })

#             self.next_track_id += 1

#         # -------------------------
#         # Remove stale tracks
#         # -------------------------
#         self.tracks = [

#             t for t in self.tracks

#             if t["age"] <= self.max_age
#         ]

#         # -------------------------
#         # Publish tracked detections
#         # -------------------------
#         out = Detection2DArray()

#         out.header = msg.header

#         for t in self.tracks:

#             if t["hits"] < self.min_hits:
#                 continue

#             x1,y1,x2,y2 = t["bbox"]

#             det = Detection2D()

#             det.bbox.center.position.x = (
#                 (x1+x2)/2
#             )

#             det.bbox.center.position.y = (
#                 (y1+y2)/2
#             )

#             det.bbox.size_x = (
#                 x2-x1
#             )

#             det.bbox.size_y = (
#                 y2-y1
#             )

#             hyp = (
#                 ObjectHypothesisWithPose()
#             )

#             hyp.hypothesis.class_id = (

#                 f'{t["class_name"]}'
#                 f' [ID {t["id"]}]'
#             )

#             hyp.hypothesis.score = (
#                 t["score"]
#             )

#             det.results.append(
#                 hyp
#             )

#             out.detections.append(
#                 det
#             )

#         self.pub.publish(out)


# def main(args=None):
#     rclpy.init(args=args)
#     node = TrackerNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         if rclpy.ok():
#             node.destroy_node()
#             rclpy.shutdown()


# if __name__ == "__main__":
#     main()