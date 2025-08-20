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
from sklearn.cluster import DBSCAN
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

        self.ULTAR_THRESHOLD = 30
        self.ULTAR_BACK_THRESHOLD = 50
        self.OVERTAKE_THRESHOLD_M = 5.0
        self.driving_state = "CENTER_DRIVING"  # "CENTER_DRIVING" or "OVERTAKING"
        self.last_override_cmd = 'reset'

        self.is_passing_obstacle = False    # 장애물 옆을 통과하고 있는지 여부
        self.pass_complete_timer = None     # 장애물을 지나친 후 안전거리 확보를 위한 타이머
        self.PASS_TIMER_DURATION_S = 0.1  # 안전거리 확보 시간 (0.1초)

        self.get_logger().info('🎯 Target Lane Planner has been started!')

    def project_lidar_on_image(self, image, detections_msg, points_3d, curve_msg,left_ultra,right_ultra,left_back_ultra,right_back_ultra):
        # 1. 모든 장애물 정보 수집 (기존과 동일)
        processed_obstacles = []
        detections = [det for det in detections_msg.detections if det.class_name == 'obstacle_vehicle']

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
                if len(indices) > 5: # 최소 5개 이상의 포인트가 있을 때만 클러스터링 수행
                    points_in_bbox = points_3d[indices]

                    dbscan = DBSCAN(eps=0.3, min_samples=3)
                    clusters = dbscan.fit_predict(points_in_bbox)
                    
                    # 노이즈(-1)를 제외하고 가장 큰 클러스터 찾기
                    unique_labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
                    if len(counts) > 0:
                        largest_cluster_label = unique_labels[np.argmax(counts)]
                        
                        # 가장 큰 클러스터에 속하는 포인트들만 선택
                        main_cluster_points = points_in_bbox[clusters == largest_cluster_label]
                        
                        # 해당 클러스터의 평균 거리 계산
                        ranges_in_cluster = np.sqrt(main_cluster_points[:, 0]**2 + main_cluster_points[:, 1]**2)
                        avg_distance = np.mean(ranges_in_cluster)
            if avg_distance > 0 and position_str != "unknown":
                processed_obstacles.append({'distance': avg_distance, 'position': position_str, 'xmin': xmin, 'ymin': ymin})
                cv2.putText(image, f"{avg_distance:.1f}m", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, position_str, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        obstacle_msg = ObstacleState()
        target_obstacle = None
        current_override_cmd = self.last_override_cmd
        num_detected_vehicles = len(detections) # YOLO가 탐지한 차량 총 수\

        is_curve = curve_msg and curve_msg.state.lower() == 'curve'

        if is_curve and num_detected_vehicles >= 1 and self.driving_state != "OVERTAKING":
            # 1-1. 커브길일 경우, 무조건 Time_Gap 모드로 전환하여 차선 변경 없이 앞차를 추종
            self.driving_state = "Time_Gap"
            if processed_obstacles:
                    target_obstacle = min(processed_obstacles, key=lambda obs: obs['distance'])
                    current_override_cmd = 'go_left' if target_obstacle['position'] == 'left' else 'go_right'
            self.get_logger().info("CURVE DETECTED: Forcing Time_Gap mode to follow lane safely.")
            
            # 거리 측정이 가능한 차량 중 가장 가까운 차를 추종 타겟으로 설정
            if processed_obstacles:
                target_obstacle = min(processed_obstacles, key=lambda obs: obs['distance'])
                self.get_logger().info(f"Curve Following: Target at {target_obstacle['distance']:.1f}m.")

        else:
            # 1-2. 직선 주로일 경우, 기존의 3-state 로직 수행
            # [상태 1] 중앙 주행 중 (CENTER_DRIVING)
            if self.driving_state == "CENTER_DRIVING":
                if num_detected_vehicles >= 2:
                    # 2대 이상 감지되면 무조건 Time_Gap 모드로 전환
                    self.driving_state = "Time_Gap"
                    self.get_logger().warn("STATE CHANGE: CENTER_DRIVING -> Time_Gap.")
                    if processed_obstacles: # 거리 측정된 차가 있으면 가장 가까운 차 추종
                        target_obstacle = min(processed_obstacles, key=lambda obs: obs['distance'])
                    current_override_cmd = 'reset'

                elif num_detected_vehicles == 1 and processed_obstacles:
                    # 1대 감지되고 거리 측정도 성공했다면, 추월 여부 판단
                    obstacle = processed_obstacles[0]

                    if obstacle['distance'] < self.OVERTAKE_THRESHOLD_M:
                        self.driving_state = "OVERTAKING"
                        self.is_passing_obstacle = False
                        self.pass_complete_timer = None
                        target_obstacle = obstacle
                        current_override_cmd = 'go_left' if target_obstacle['position'] == 'right' else 'go_right'
                        self.get_logger().warn(f"STATE CHANGE: CENTER_DRIVING -> OVERTAKING ({current_override_cmd}).")
                    else: # 1대 있지만 멀리 있으면 중앙 주행 유지
                        current_override_cmd = 'reset'
                else: # 감지된 차량이 없거나, 있어도 거리 측정을 못하면 중앙 주행 유지
                    current_override_cmd = 'reset'

            # [상태 2] 추월 중 (OVERTAKING)
            elif self.driving_state == "OVERTAKING":
                if processed_obstacles:
                    target_obstacle = min(processed_obstacles, key=lambda obs: obs['distance'])

                current_override_cmd = self.last_override_cmd # 추월 중에는 차선 변경 명령 유지

                # 1단계: 아직 장애물 옆을 지나치기 시작하지 않았을 때
                if not self.is_passing_obstacle:
                    is_side_detected = (self.last_override_cmd == 'go_left' and right_ultra < self.ULTAR_THRESHOLD) or \
                                       (self.last_override_cmd == 'go_right' and left_ultra < self.ULTAR_THRESHOLD)
                    if is_side_detected:
                        self.is_passing_obstacle = True
                        self.get_logger().info("Side of obstacle detected. Now passing alongside.")
                    else:
                        self.get_logger().info(f"Approaching side... Left: {left_ultra}, Right: {right_ultra}")
                
                # 2단계: 장애물 옆을 지나치고 있을 때
                else:
                    is_side_cleared = (self.last_override_cmd == 'go_left' and right_ultra > self.ULTAR_THRESHOLD and right_back_ultra > self.ULTAR_BACK_THRESHOLD) or \
                                      (self.last_override_cmd == 'go_right' and left_ultra > self.ULTAR_THRESHOLD and left_back_ultra > self.ULTAR_BACK_THRESHOLD)

                    # 3단계: 장애물 끝을 통과하여 타이머를 시작해야 할 때
                    if is_side_cleared and self.pass_complete_timer is None:
                        self.pass_complete_timer = self.get_clock().now()
                        self.get_logger().info(f"Side of obstacle cleared. Starting {self.PASS_TIMER_DURATION_S}s safety timer.")
                    
                    # 타이머가 시작되었다면, 시간이 다 지났는지 확인
                    if self.pass_complete_timer is not None:
                        duration = self.get_clock().now() - self.pass_complete_timer
                        # 4단계: 타이머 종료 -> 추월 완료!
                        if duration.nanoseconds / 1e9 > self.PASS_TIMER_DURATION_S:
                            self.driving_state = "CENTER_DRIVING"
                            current_override_cmd = 'reset'
                            self.get_logger().warn("STATE CHANGE: OVERTAKING -> CENTER_DRIVING (Timer complete).")
                            # 상태 변수 초기화
                            self.is_passing_obstacle = False
                            self.pass_complete_timer = None
                        else:
                            self.get_logger().info(f"Safety timer running... ({duration.nanoseconds / 1e9:.2f}s)")
                    # 아직 장애물 옆을 지나고 있는 경우 (통과 중 센서 값이 20 미만인 상태)
                    else:
                         self.get_logger().info(f"Passing alongside... Left: {left_ultra}, Right: {right_ultra}")

            # [상태 3] 다중 차량 추종 (Time_Gap)
            elif self.driving_state == "Time_Gap":
                # 조건 1: 차량이 2대 미만이면 즉시 중앙 주행으로 복귀 (상태 탈출)
                if num_detected_vehicles < 2:
                    self.driving_state = "CENTER_DRIVING"
                    current_override_cmd = 'reset'
                    self.get_logger().warn("STATE CHANGE: Time_Gap -> CENTER_DRIVING (Vehicles < 2).")
                
                # 조건 2: 2대 이상이면 Time_Gap 상태 유지
                else:
                    # Time_Gap 상태에서는 항상 중앙 차선을 유지합니다.
                    current_override_cmd = 'reset'
                    
                    # 거리 측정이 가능한 차량이 있다면, 그중 가장 가까운 차를 추종 타겟으로 삼습니다.
                    if processed_obstacles:
                        target_obstacle = min(processed_obstacles, key=lambda obs: obs['distance'])
                        self.get_logger().info(f"Time_Gap: Following closest vehicle at {target_obstacle['distance']:.1f}m.")
                    else:
                        self.get_logger().info("Time_Gap: 2+ vehicles detected, but no distance info. Driving straight.")
        # =======================> 수정된 로직 끝 <=========================

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
        # self.get_logger().info('--- 동기화 콜백 실행됨! ---')
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # 기본값으로 초기화
        left_ultra = 100
        left_back_ultra = 999
        right_ultra = 100
        right_back_ultra = 999

        # ultra_msg.data의 길이를 확인하는 방어 코드 추가
        if len(ultra_msg.data) > 0:
            left_ultra = ultra_msg.data[0]
        if len(ultra_msg.data) > 7:
            left_back_ultra = ultra_msg.data[7]      
        if len(ultra_msg.data) > 5:
            right_back_ultra = ultra_msg.data[5]          
        if len(ultra_msg.data) > 4:
            right_ultra = ultra_msg.data[4]
        else:
            # 만약 오른쪽 센서 데이터가 없다면, 다른 값(예: 마지막 값)을 사용하거나 경고를 출력할 수 있습니다.
            self.get_logger().warn(f"Ultrasonic data only has {len(ultra_msg.data)} elements. Can't access index 4.")

        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment
        
        # 유효한 거리 값만 필터링 (5m 이내)
        mask = (ranges > scan_msg.range_min) & (ranges < 5.0)
        
        # 3D 월드 좌표계 포인트 생성
        x = ranges[mask] * np.cos(angles[mask])
        y = ranges[mask] * np.sin(angles[mask])
        points_3d = np.stack([x, y, np.zeros_like(x)], axis=1)
        
        # 3D 포인트를 2D 이미지 좌표로 투영
        img_pts = np.array([])
        if points_3d.shape[0] > 0:
            cam_pts = (self.R @ points_3d.T + self.t).T
            img_pts_raw, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), self.mtx, None)
            img_pts = img_pts_raw.squeeze(axis=1)
            
            # 이미지 위에 라이다 포인트 그리기 (초록색 점)
            for pt in img_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        if curve_msg.points:
            curve_pts = np.array([[p.x, p.y] for p in curve_msg.points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [curve_pts], isClosed=False, color=(255, 0, 255), thickness=2)
        if curve_msg.state:
            cv2.putText(frame, f"Lane State: {curve_msg.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        vehicle_count = sum(1 for det in yolo_msg.detections if det.class_name == 'obstacle_vehicle')
        for det in yolo_msg.detections:
            if det.class_name == 'obstacle_vehicle':
                x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 현재 주행 상태를 화면에 표시
        cv2.putText(frame, f"State: {self.driving_state}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Ultra: {left_ultra}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Back Ultra: {left_back_ultra}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right Ultra: {right_ultra}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right Back Ultra: {right_back_ultra}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        frame = self.project_lidar_on_image(frame, yolo_msg, points_3d, curve_msg,left_ultra,right_ultra,left_back_ultra,right_back_ultra)
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