# integrated_target_lane_planner.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from std_msgs.msg import String, Int32MultiArray, Header
from custom_interfaces.msg import Curve, Detections, ObstacleState, Ultrasonic
import message_filters
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
import yaml
import os
from sklearn.cluster import DBSCAN

# ================================
# (중요) 센서 매핑 및 사용 순서
# 1: 좌측(left)
# 5: 우측(right)
# 6: 우측후면(right_back)
# 7: 후면(back)
# 8: 좌측후면(left_back)
# ※ 2,3,4번 센서는 무시
# 내부적으로 0-인덱스이므로 1→0, 5→4, 6→5, 7→6, 8→7
# ================================

class TargetLanePlanner(Node):
    def __init__(self):
        super().__init__('integrated_target_lane_planner')
        
        # ================================
        # 칼리브레이션 설정 (기존)
        # ================================
        pkg_share = get_package_share_directory('cam')
        intrinsic_file = os.path.join(pkg_share, 'config', 'fisheye_calib.yaml')
        extrinsic_file = os.path.join(pkg_share, 'config', 'extrinsic.yaml')
        with open(intrinsic_file, 'r') as f: intrinsic_calib = yaml.safe_load(f)
        self.img_size = (intrinsic_calib['image_width'], intrinsic_calib['image_height'])
        self.mtx = np.array(intrinsic_calib['K'])
        with open(extrinsic_file, 'r') as f: extrinsic_calib = yaml.safe_load(f)
        self.R, self.t = np.array(extrinsic_calib['R']), np.array(extrinsic_calib['t']).reshape((3, 1))
        self.bridge = CvBridge()

        # ================================
        # 초음파 관련 설정 파라미터
        # ================================
        self.TRIGGER_ENTER_CM = 20        # <= 이 값 이하를 '근접'으로 본다
        self.TRIGGER_EXIT_CM  = 50        # >= 이 값 이상으로 '이탈' 시 타이머 시작
        self.CENTER_DELAY_SEC = 2.0       # 타이머 만료 후 center(=reset) 보낼 지연 시간

        # === QoS 설정 ===
        us_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # ================================
        # 구독자 설정
        # ================================
        self.image_subscription = message_filters.Subscriber(self, Image, '/image_raw')
        self.curve_subscription = message_filters.Subscriber(self, Curve, '/center_curve')
        self.yolo_subscription = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.scan_subscription = message_filters.Subscriber(self, LaserScan, '/scan', qos_profile=qos_profile_sensor_data)
        
        # 초음파 원본 데이터 구독 (통합)
        self.us_sub = self.create_subscription(
            Int32MultiArray, 'xycar_ultrasonic', self.ultra_callback, us_qos
        )
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.curve_subscription, self.yolo_subscription, self.scan_subscription],
            queue_size=10, slop=0.1)
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        # ================================
        # 퍼블리셔 설정
        # ================================
        self.obstacle_state_pub = self.create_publisher(ObstacleState, '/obstacle_state', 10)
        self.override_pub = self.create_publisher(String, '/lane_override_cmd', 10)
        self.ultrasonic_pub = self.create_publisher(Ultrasonic, '/ultrasonic', 10)  # 커스텀 초음파 메시지

        # ================================
        # 상태 변수들
        # ================================
        # Lane Planner 관련
        self.OVERTAKE_THRESHOLD_M = 2.0
        self.driving_state = "CENTER_DRIVING"  # "CENTER_DRIVING", "OVERTAKING", "Time_Gap"
        self.overtake_timer = None
        self.OVERTAKE_DURATION_S = 4.0
        self.last_override_cmd = 'reset'

        # 초음파 관련
        self.ultra_msg = None
        self.last_ultra_vals = None
        self.last_ultra_stamp_ns = None
        self.override_mode = None  # None | 'go_right' | 'go_left' | 'center'
        self.reset_timer = None
        self.reset_timer_armed_from = None

        # 히스테리시스 플래그
        self.close_seen_left = False   # go_right 모드에서 1번 센서 기준
        self.close_seen_right = False  # go_left 모드에서 5번 센서 기준
        self.ultra_trigger_ready = False  # 초음파 기반 추월 조건 만족 여부

        # 초음파 데이터 대기
        self.get_logger().info("Waiting for ultrasonic data...")
        self.wait_for_ultrasonic_message()
        self.get_logger().info("Ultrasonic Ready ----------")

        # 정기적으로 커스텀 초음파 메시지 퍼블리시
        self.ultrasonic_timer = self.create_timer(0.1, self.ultrasonic_timer_callback)

        self.get_logger().info('🎯 Integrated Target Lane Planner with Ultrasonic started!')

    # ================================
    # 초음파 관련 메소드들
    # ================================
    def ultra_callback(self, msg):
        """초음파 원본 데이터 콜백 - 기존 UltrasonicDetector 로직 통합"""
        # 기존 데이터 저장
        self.ultra_msg = msg.data
        self.last_ultra_vals = list(msg.data) if msg.data is not None else None
        self.last_ultra_stamp_ns = self.get_clock().now().nanoseconds

        # 트리거 감시 로직 (override_mode가 있을 때만)
        if not self.override_mode or self.reset_timer is not None or self.last_ultra_vals is None:
            return

        # 안전 인덱싱
        try:
            left_front_val  = int(self.last_ultra_vals[0])  # 1번 센서 (좌측)
            right_front_val = int(self.last_ultra_vals[4])  # 5번 센서 (우측)
        except Exception:
            return

        # go_right: 좌측(1번) 센서 기준
        if self.override_mode == 'go_right':
            # enter: 근접(<=20)하면 플래그 세트 및 트리거 조건 만족
            if not self.close_seen_left and left_front_val <= self.TRIGGER_ENTER_CM:
                self.close_seen_left = True
                self.ultra_trigger_ready = True
                self.get_logger().info(f"👀 go_right: left sensor ENTER detected (<= {self.TRIGGER_ENTER_CM}cm). Ultra trigger READY!")
            # exit: 한 번 근접을 봤고, 멀어짐(>=50)을 보면 타이머 암
            elif self.close_seen_left and left_front_val >= self.TRIGGER_EXIT_CM:
                self._arm_reset_timer(from_mode='go_right')
                self.close_seen_left = False

        # go_left: 우측(5번) 센서 기준
        elif self.override_mode == 'go_left':
            if not self.close_seen_right and right_front_val <= self.TRIGGER_ENTER_CM:
                self.close_seen_right = True
                self.ultra_trigger_ready = True
                self.get_logger().info(f"👀 go_left: right sensor ENTER detected (<= {self.TRIGGER_ENTER_CM}cm). Ultra trigger READY!")
            elif self.close_seen_right and right_front_val >= self.TRIGGER_EXIT_CM:
                self._arm_reset_timer(from_mode='go_left')
                self.close_seen_right = False

    def wait_for_ultrasonic_message(self):
        """초음파 메시지 대기"""
        while self.ultra_msg is None and rclpy.ok():
            rclpy.spin_once(self)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

    def ultrasonic_timer_callback(self):
        """커스텀 초음파 메시지 정기 퍼블리시"""
        if self.ultra_msg is not None:
            ultra_msg = Ultrasonic()
            ultra_msg.header.stamp = self.get_clock().now().to_msg()
            ultra_msg.header.frame_id = 'ultrasonic'
            ultra_msg.data = self.ultra_msg
            self.ultrasonic_pub.publish(ultra_msg)

    def _arm_reset_timer(self, from_mode: str):
        """리셋 타이머 설정"""
        if self.reset_timer is not None:
            return  # 이미 동작 중
        self.reset_timer_armed_from = from_mode
        self.reset_timer = self.create_timer(self.CENTER_DELAY_SEC, self._on_reset_timer)
        self.get_logger().info(f"⏱️  Triggered by {from_mode}: will reset to center in {self.CENTER_DELAY_SEC:.1f}s")

    def _on_reset_timer(self):
        """리셋 타이머 콜백"""
        self._cancel_reset_timer()
        
        # 상태 갱신
        self.override_mode = 'center'
        self._reset_hysteresis_flags()
        self.ultra_trigger_ready = False

        self.get_logger().info(f"✅ Reset to center after trigger from {self.reset_timer_armed_from}.")
        self.reset_timer_armed_from = None

    def _cancel_reset_timer(self):
        """리셋 타이머 취소"""
        if self.reset_timer is not None:
            try:
                self.reset_timer.cancel()
            except Exception:
                pass
            self.reset_timer = None

    def _reset_hysteresis_flags(self):
        """히스테리시스 플래그 리셋"""
        self.close_seen_left = False
        self.close_seen_right = False

    # ================================
    # Lane Planner 관련 메소드들
    # ================================
    def project_lidar_on_image(self, image, detections_msg, points_3d, curve_msg):
        # 1. 모든 장애물 정보 수집
        processed_obstacles = []
        detections = [det for det in detections_msg.detections if det.class_name == 'obstacle_vehicle']
        
        h, w, _ = image.shape
        if points_3d.shape[0] == 0: 
            img_pts = np.array([])
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
                indices = np.where((img_pts[:, 0] >= xmin) & (img_pts[:, 0] < xmax) & 
                                 (img_pts[:, 1] >= ymin) & (img_pts[:, 1] < ymax))[0]
                if len(indices) > 5:
                    points_in_bbox = points_3d[indices]
                    dbscan = DBSCAN(eps=0.3, min_samples=3)
                    clusters = dbscan.fit_predict(points_in_bbox)
                    
                    unique_labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
                    if len(counts) > 0:
                        largest_cluster_label = unique_labels[np.argmax(counts)]
                        main_cluster_points = points_in_bbox[clusters == largest_cluster_label]
                        ranges_in_cluster = np.sqrt(main_cluster_points[:, 0]**2 + main_cluster_points[:, 1]**2)
                        avg_distance = np.mean(ranges_in_cluster)
            
            if avg_distance > 0 and position_str != "unknown":
                processed_obstacles.append({'distance': avg_distance, 'position': position_str, 
                                          'xmin': xmin, 'ymin': ymin})
                cv2.putText(image, f"{avg_distance:.1f}m", (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, position_str, (xmin, ymin - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 2. 상태 머신 로직 (수정된 부분)
        obstacle_msg = ObstacleState()
        target_obstacle = None
        current_override_cmd = self.last_override_cmd
        num_detected_vehicles = len(detections)

        # [상태 1] 중앙 주행 중 (CENTER_DRIVING)
        if self.driving_state == "CENTER_DRIVING":
            if num_detected_vehicles >= 2:
                self.driving_state = "Time_Gap"
                self.get_logger().warn("STATE CHANGE: CENTER_DRIVING -> Time_Gap (2+ vehicles detected).")
                if processed_obstacles:
                    target_obstacle = max(processed_obstacles, key=lambda obs: obs['distance'])
                    current_override_cmd = 'go_left' if target_obstacle['position'] == 'left' else 'go_right'
            elif num_detected_vehicles == 1 and processed_obstacles:
                obstacle = processed_obstacles[0]
                # ★ 수정된 부분: 초음파 트리거 조건 추가 ★
                if obstacle['distance'] < self.OVERTAKE_THRESHOLD_M and self.ultra_trigger_ready:
                    self.driving_state = "OVERTAKING"
                    self.overtake_timer = self.get_clock().now()
                    target_obstacle = obstacle
                    current_override_cmd = 'go_left' if target_obstacle['position'] == 'right' else 'go_right'
                    self.get_logger().warn(f"STATE CHANGE: CENTER_DRIVING -> OVERTAKING ({current_override_cmd}) with ULTRA TRIGGER!")
                else:
                    current_override_cmd = 'reset'
                    if obstacle['distance'] < self.OVERTAKE_THRESHOLD_M and not self.ultra_trigger_ready:
                        self.get_logger().info("Vehicle close but ultra trigger not ready yet...")
            else:
                current_override_cmd = 'reset'

        # [상태 2] 추월 중 (OVERTAKING) - 초음파 트리거 조건이 이미 만족된 상태에서만 진입 가능
        elif self.driving_state == "OVERTAKING":
            duration = self.get_clock().now() - self.overtake_timer
            if duration.nanoseconds / 1e9 > self.OVERTAKE_DURATION_S:
                self.driving_state = "CENTER_DRIVING"
                self.overtake_timer = None
                current_override_cmd = 'reset'
                self.get_logger().info("STATE CHANGE: OVERTAKING -> CENTER_DRIVING.")
            else:
                current_override_cmd = self.last_override_cmd
                self.get_logger().info(f"Overtaking with ultra trigger... ({duration.nanoseconds / 1e9:.1f}s)")
        
        # [상태 3] 다중 차량 추종 (Time_Gap)
        elif self.driving_state == "Time_Gap":
            if num_detected_vehicles < 2:
                self.driving_state = "CENTER_DRIVING"
                current_override_cmd = 'reset'
                self.get_logger().warn("STATE CHANGE: Time_Gap -> CENTER_DRIVING (Vehicles < 2).")
            else:
                if processed_obstacles:
                    target_obstacle = max(processed_obstacles, key=lambda obs: obs['distance'])
                    current_override_cmd = 'go_left' if target_obstacle['position'] == 'left' else 'go_right'
                else:
                    current_override_cmd = 'reset'

        # Override 명령 업데이트
        if current_override_cmd != self.last_override_cmd:
            # 새로운 override 모드 설정 (초음파 모니터링용)
            if current_override_cmd in ('go_right', 'go_left'):
                self.override_mode = current_override_cmd
                self._cancel_reset_timer()
                self._reset_hysteresis_flags()
                self.ultra_trigger_ready = False  # 새로운 명령 시 트리거 리셋
            elif current_override_cmd in ('reset', 'center'):
                self._cancel_reset_timer()
                self._reset_hysteresis_flags()
                self.override_mode = 'center'
                self.ultra_trigger_ready = False

            cmd_msg = String(data=current_override_cmd)
            self.override_pub.publish(cmd_msg)
            self.get_logger().warn(f"💡 Lane command sent: '{current_override_cmd}'")
            self.last_override_cmd = current_override_cmd

        # 장애물 상태 메시지 설정
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

    def synchronized_callback(self, image_msg, curve_msg, yolo_msg, scan_msg):
        """동기화된 콜백 함수"""
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # LiDAR 데이터 처리
        ranges, angles = np.array(scan_msg.ranges), scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment
        mask = (ranges > scan_msg.range_min) & (ranges < 5.0)
        x, y = ranges[mask] * np.cos(angles[mask]), ranges[mask] * np.sin(angles[mask])
        points_3d = np.stack([x, y, np.zeros_like(x)], axis=1)
        
        # 곡선 그리기
        if curve_msg.points:
            curve_pts = np.array([[p.x, p.y] for p in curve_msg.points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [curve_pts], isClosed=False, color=(255, 0, 255), thickness=2)
        
        if curve_msg.state:
            cv2.putText(frame, f"Lane State: {curve_msg.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 초음파 센서값 표시
        if self.ultra_msg is not None:
            ultra_ranges = np.array(self.ultra_msg)
            left_ultra = ultra_ranges[0]
            right_ultra = ultra_ranges[4]
            cv2.putText(frame, f"Ultra L:{left_ultra} R:{right_ultra} Trigger:{self.ultra_trigger_ready}", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 차량 감지 결과 표시
        vehicle_count = sum(1 for det in yolo_msg.detections if det.class_name == 'obstacle_vehicle')
        for det in yolo_msg.detections:
            if det.class_name == 'obstacle_vehicle':
                x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"State: {self.driving_state}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)

        # 메인 프로젝션 함수 호출
        frame = self.project_lidar_on_image(frame, yolo_msg, points_3d, curve_msg)
        cv2.imshow("Integrated Target Lane Planner with Ultrasonic", frame)
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