import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cam_interfaces.msg import Curve # 생성한 메시지 import
import message_filters # 시간 동기화를 위해 import
from cam_interfaces.msg import Detections


class TargetLanePlanner(Node):
    def __init__(self):
        super().__init__('target_lane_planner')
        self.bridge = CvBridge()

        # 원본 이미지와 커브 토픽을 동시에 구독
        self.image_subscription = message_filters.Subscriber(self, Image, '/image_raw') # 이전 백파일 사용할 때는 '/stamped_image_raw'로 변경
        self.curve_subscription = message_filters.Subscriber(self, Curve, '/center_curve')
        self.yolo_subscription = message_filters.Subscriber(self, Detections, '/yolo_detections')


        # 두 토픽의 타임스탬프를 기준으로 메시지를 동기화
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.curve_subscription, self.yolo_subscription],
            queue_size=10, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        
        self.get_logger().info('target lane planner node has been started.')

    def synchronized_callback(self, image_msg, curve_msg, yolo_msg):
        # self.get_logger().info('Received synchronized image, curve, yolo_detections')
        try:
            # 이미지를 OpenCV 포맷으로 변환
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Curve 메시지에서 점들의 리스트를 numpy 배열로 변환
        if curve_msg.points:
            curve_pts = np.array([[p.x, p.y] for p in curve_msg.points], dtype=np.int32)
            curve_pts = curve_pts.reshape((-1, 1, 2))
            
            # OpenCV를 사용해 받은 점들로 폴리라인 그리기
            cv2.polylines(frame, [curve_pts], isClosed=False, color=(255, 0, 255), thickness=1)
        
        if curve_msg.state:
            curve_state = curve_msg.state
            cv2.putText(frame, curve_state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # --- YOLO 장애물차량 바운딩 박스 ---
        if yolo_msg.detections:
            detections = [
                det for det in yolo_msg.detections
                if det.class_name == 'obstacle_vehicle'
            ]
            for det in detections:
                x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)


        # 결과 이미지 출력
        cv2.imshow("target_lane_planner : Intergrated Visualization", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TargetLanePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt detected, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()