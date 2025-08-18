# target_lane_planner.py

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from std_msgs.msg import String
from custom_interfaces.msg import Curve, Detections, ObstacleState, Ultrasonic
import message_filters
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
import yaml
import os

class TargetLanePlanner(Node):
    def __init__(self):
        super().__init__('target_lane_planner')
        pkg_share = get_package_share_directory('cam')
        intrinsic_file = os.path.join(pkg_share, 'config', 'fisheye_calib.yaml')
        extrinsic_file = os.path.join(pkg_share, 'config', 'extrinsic.yaml')
        with open(intrinsic_file, 'r') as f: intrinsic_calib = yaml.safe_load(f)
        self.img_size = (intrinsic_calib['image_width'], intrinsic_calib['image_height'])
        self.mtx = np.array(intrinsic_calib['K'])
        with open(extrinsic_file, 'r') as f: extrinsic_calib = yaml.safe_load(f)
        self.R, self.t = np.array(extrinsic_calib['R']), np.array(extrinsic_calib['t']).reshape((3, 1))
        self.bridge = CvBridge()

        self.image_subscription = message_filters.Subscriber(self, Image, '/image_raw')
        self.curve_subscription = message_filters.Subscriber(self, Curve, '/center_curve')
        self.yolo_subscription = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.scan_subscription = message_filters.Subscriber(self, LaserScan, '/scan', qos_profile=qos_profile_sensor_data)
        self.ultra_subscription = message_filters.Subscriber(self, Ultrasonic, '/ultrasonic')
        
        self.obstacle_state_pub = self.create_publisher(ObstacleState, '/obstacle_state', 10)
        self.override_pub = self.create_publisher(String, '/lane_override_cmd', 10)
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.curve_subscription, self.yolo_subscription, self.scan_subscription, self.ultra_subscription],
            queue_size=10, slop=0.1)
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        self.OVERTAKE_THRESHOLD_M = 2.0
        # --- [추가] 상태 머신 관련 변수 ---
        self.driving_state = "CENTER_DRIVING"  # "CENTER_DRIVING" or "OVERTAKING"
        self.overtake_timer = None
        self.OVERTAKE_DURATION_S = 4.0  # 추월에 필요한 시간 (초), 튜닝 필요
        self.last_override_cmd = 'reset'

        self.get_logger().info('🎯 Advanced Target Lane Planner (v3 with State Machine) has been started!')

    def project_lidar_on_image(self, image, detections_msg, points_3d, curve_msg):
        # 1. 모든 장애물 정보 수집 (기존과 동일)
        processed_obstacles = []
        detections = [det for det in detections_msg.detections if det.class_name == 'obstacle_vehicle']
        # ... (이하 정보 수집 로직은 이전과 동일) ...
        h, w, _ = image.shape
        if points_3d.shape[0] == 0: img_pts = np.array([])
        else:
            cam_pts = (self.R @ points_3d.T + self.t).T
            img_pts, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), self.mtx, None)
            img_pts = img_pts.squeeze(axis=1)
        has_curve = curve_msg and len(curve_msg.points) >= 2
        if has_curve:
            curve_points = np.array([[p.x, p.y] for p in curve_msg.points])
            sorted_indices = np.argsort(curve_points[:, 1])
            curve_x_coords, curve_y_coords = curve_points[sorted_indices, 0], curve_points[sorted_indices, 1]
        for det in detections:
            xmin, ymin, xmax, ymax = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
            position_str = "unknown"
            if has_curve:
                bbox_center_x, bbox_center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
                lane_x_at_bbox_y = np.interp(bbox_center_y, curve_y_coords, curve_x_coords)
                position_str = "left" if bbox_center_x < lane_x_at_bbox_y else "right"
            avg_distance = -1.0
            if img_pts.shape[0] > 0:
                indices = np.where((img_pts[:, 0] >= xmin) & (img_pts[:, 0] < xmax) & (img_pts[:, 1] >= ymin) & (img_pts[:, 1] < ymax))[0]
                if len(indices) > 0:
                    ranges_in_bbox = np.sqrt(points_3d[indices, 0]**2 + points_3d[indices, 1]**2)
                    avg_distance = np.mean(ranges_in_bbox)
            if avg_distance > 0 and position_str != "unknown":
                processed_obstacles.append({'distance': avg_distance, 'position': position_str, 'xmin': xmin, 'ymin': ymin})
                cv2.putText(image, f"{avg_distance:.1f}m", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, position_str, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 2. 상태 머신 기반 의사결정 로직
        obstacle_msg = ObstacleState()
        target_obstacle = None
        current_override_cmd = self.last_override_cmd # 기본적으로 이전 명령 유지

        # [상태 1] 중앙 주행 중일 때의 판단
        if self.driving_state == "CENTER_DRIVING":
            if len(processed_obstacles) >= 2:
                # 2대 이상: 가장 먼 차 추종 (차선 변경 시작 안함, 컨트롤러가 따라가기만 함)
                target_obstacle = max(processed_obstacles, key=lambda obs: obs['distance'])
                current_override_cmd = 'reset' # 중앙 차선에서 따라감
            elif len(processed_obstacles) == 1:
                target_obstacle = processed_obstacles[0]
                # 1대: 거리가 가까우면 "추월 상태"로 전환
                if target_obstacle['distance'] < self.OVERTAKE_THRESHOLD_M:
                    self.driving_state = "OVERTAKING"
                    self.overtake_timer = self.get_clock().now()
                    target_position = target_obstacle['position']
                    current_override_cmd = 'go_left' if target_position == 'right' else 'go_right'
                    self.get_logger().warn(f"STATE CHANGE: CENTER_DRIVING -> OVERTAKING ({current_override_cmd}). Timer started.")
                else:
                    current_override_cmd = 'reset'
            else: # 장애물 없음
                current_override_cmd = 'reset'

        # [상태 2] 추월 중일 때의 판단
        elif self.driving_state == "OVERTAKING":
            duration = self.get_clock().now() - self.overtake_timer
            # 타이머가 만료되면 "중앙 주행 상태"로 복귀
            if duration.nanoseconds / 1e9 > self.OVERTAKE_DURATION_S:
                self.driving_state = "CENTER_DRIVING"
                self.overtake_timer = None
                current_override_cmd = 'reset'
                self.get_logger().info("STATE CHANGE: OVERTAKING -> CENTER_DRIVING. Overtake complete.")
            else:
                # 타이머가 끝나기 전까지는 기존 차선 변경 명령을 유지 (YOLO 결과 무시)
                current_override_cmd = self.last_override_cmd 
                self.get_logger().info(f"Overtaking in progress... ({duration.nanoseconds / 1e9:.1f}s / {self.OVERTAKE_DURATION_S}s)")

        # ... (이하 명령 및 상태 발행 로직은 이전과 동일) ...
        if current_override_cmd != self.last_override_cmd:
            cmd_msg = String(data=current_override_cmd)
            self.override_pub.publish(cmd_msg)
            self.get_logger().warn(f"💡 Lane command sent: '{current_override_cmd}'")
            self.last_override_cmd = current_override_cmd
        if target_obstacle:
            obstacle_msg.distance_m = float(target_obstacle['distance'])
            obstacle_msg.position = target_obstacle['position']
            tx, ty = target_obstacle['xmin'], target_obstacle['ymin']
            cv2.putText(image, "TARGET", (tx, ty - 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        else:
            obstacle_msg.distance_m = 999.0
            obstacle_msg.position = "none"
        obstacle_msg.vehicle_count = len(detections)
        self.obstacle_state_pub.publish(obstacle_msg)
        return image

    def synchronized_callback(self, image_msg, curve_msg, yolo_msg, scan_msg, ultra_msg):
        # ... (이하 synchronized_callback 및 main 함수는 기존과 동일) ...
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        ranges, angles = np.array(scan_msg.ranges), scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment
        mask = (ranges > scan_msg.range_min) & (ranges < 5.0)
        x, y = ranges[mask] * np.cos(angles[mask]), ranges[mask] * np.sin(angles[mask])
        points_3d = np.stack([x, y, np.zeros_like(x)], axis=1)
        if curve_msg.points:
            curve_pts = np.array([[p.x, p.y] for p in curve_msg.points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [curve_pts], isClosed=False, color=(255, 0, 255), thickness=2)
        if curve_msg.state:
            cv2.putText(frame, f"Lane State: {curve_msg.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 초음파 센서값 리스트
        ultra_ranges = np.array(ultra_msg.data)
        left_ultra = ultra_ranges[0]  # 좌측 초음파센서 값
        right_ultra = ultra_ranges[4] # 우측 초음파센서 값

        vehicle_count = sum(1 for det in yolo_msg.detections if det.class_name == 'obstacle_vehicle')
        for det in yolo_msg.detections:
            if det.class_name == 'obstacle_vehicle':
                x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 현재 주행 상태를 화면에 표시
        cv2.putText(frame, f"State: {self.driving_state}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)

        frame = self.project_lidar_on_image(frame, yolo_msg, points_3d, curve_msg)
        cv2.imshow("target_lane_planner : Integrated Visualization", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TargetLanePlanner()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info('KeyboardInterrupt detected, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()