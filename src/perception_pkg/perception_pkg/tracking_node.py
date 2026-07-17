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
        self.declare_parameter("max_age",30)
        self.declare_parameter("min_hits",4)
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
            if t.age > 2:
                continue    
            x1,y1,x2,y2=t.bbox()
            det=Detection2D()
            # TRACK ID
            det.id = str(t.id)
            det.bbox.center.position.x=(x1+x2)/2
            det.bbox.center.position.y=(y1+y2)/2
            det.bbox.size_x=x2-x1
            det.bbox.size_y=y2-y1
            hyp=ObjectHypothesisWithPose()
            hyp.hypothesis.class_id=t.class_name
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
        node.get_logger().info('Tracker node intercepting Ctrl+C shutdown safely...')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__=="__main__":
    main()




