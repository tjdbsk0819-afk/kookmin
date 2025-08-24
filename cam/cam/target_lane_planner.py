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
        self.OVERTAKE_THRESHOLD_M = 4.0
        self.CURVE_OVERTAKE_THRESHOLD_M = 1.0
        self.MULTI_VEHICLE_OVERTAKE_GAP_M = 1.2
        self.driving_state = "CENTER_DRIVING"  # "CENTER_DRIVING" or "OVERTAKING"
        self.last_override_cmd = 'reset'

        self.is_passing_obstacle = False    # 장애물 옆을 통과하고 있는지 여부
        self.pass_complete_timer = None     # 장애물을 지나친 후 안전거리 확보를 위한 타이머
        self.PASS_TIMER_DURATION_S = 0.1  # 안전거리 확보 시간 (0.1초)

        self.E_STOP_DISTANCE_M = 0.5  # 급정지를 발동할 거리 (m)
        self.E_STOP_X_MIN = 50     # 급정지를 감지할 전방 카메라 x좌표 (최소)
        self.E_STOP_X_MAX = 600   # 급정지를 감지할 전방 카메라 x좌표 (최대)

        self.E_STOP_RECT_X_MIN = 280
        self.E_STOP_RECT_X_MAX = 360
        self.E_STOP_RECT_Y_MIN = 260
        self.E_STOP_RECT_Y_MAX = 400
        self.E_STOP_RECT_PIXEL_COUNT_THRESHOLD = 3

        self.PROXIMITY_THRESHOLD_M = 0.6
        self.smoothed_obstacle_position = 0.0  # -1 (left) ~ +1 (right) 사이의 값을 가짐
        self.EMA_ALPHA = 0.3  # 스무딩 강도 (0.1: 매우 부드러움, 0.9: 매우 민감)

        self.get_logger().info('🎯 Target Lane Planner has been started!')

    def project_lidar_on_image(self, image, detections_msg, points_3d, curve_msg,left_ultra,right_ultra,left_back_ultra,right_back_ultra,img_pts):
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
            position_str_final = "unknown"
            avg_distance = -1.0
            points_in_path_count = 0     


            # ================================ LiDAR 포인트 클라우드 클러스터링 ================================
            if img_pts.shape[0] > 0:
                indices = np.where((img_pts[:, 0] >= xmin) & (img_pts[:, 0] < xmax) & (img_pts[:, 1] >= ymin) & (img_pts[:, 1] < ymax))[0]
                if len(indices) > 2: # 최소 5개 이상의 포인트가 있을 때만 클러스터링 수행
                    points_in_bbox = points_3d[indices]

                    dbscan = DBSCAN(eps=0.3, min_samples=3)
                    clusters = dbscan.fit_predict(points_in_bbox)
                    img_pts_in_bbox = img_pts[indices]
                    unique_labels = np.unique(clusters[clusters != -1])

                    if len(unique_labels) > 0:
                        # --- 가장 가까운 클러스터만 선택하고, 해당 클러스터로 거리 계산 및 시각화 ---
                        
                        # 1. 각 클러스터의 평균 거리를 계산합니다.
                        cluster_info_list = []
                        for label in unique_labels:
                            # 현재 클러스터에 속하는 3D 포인트
                            current_cluster_points_3d = points_in_bbox[clusters == label]
                            # 거리 계산
                            ranges = np.sqrt(current_cluster_points_3d[:, 0]**2 + current_cluster_points_3d[:, 1]**2)
                            avg_dist = np.mean(ranges)
                            cluster_info_list.append({'label': label, 'distance': avg_dist})

                        # 2. 가장 가까운 클러스터 정보를 찾습니다.
                        closest_cluster_info = min(cluster_info_list, key=lambda x: x['distance'])
                        closest_cluster_label = closest_cluster_info['label']
                        
                        # 3. 가장 가까운 클러스터의 포인트들만 'main_cluster_points'로 확정합니다.
                        main_cluster_points = points_in_bbox[clusters == closest_cluster_label]
                        
                        # 4. 확정된 포인트들로 최종 평균 거리를 다시 계산합니다. (더 정확함)
                        final_ranges = np.sqrt(main_cluster_points[:, 0]**2 + main_cluster_points[:, 1]**2)
                        avg_distance = np.mean(final_ranges)

                        img_pts_in_cluster = img_pts[indices][clusters == closest_cluster_label]
                        points_in_path = [pt for pt in img_pts_in_cluster if self.E_STOP_X_MIN < pt[0] < self.E_STOP_X_MAX]
                        points_in_path_count = len(points_in_path)

                        # 5. 시각화: 가장 가까운 클러스터의 포인트들만 그립니다.
                        points_to_draw = img_pts_in_bbox[clusters == closest_cluster_label]
                        for pt in points_to_draw:
                            # 가장 중요한 클러스터이므로 눈에 잘 띄게 빨간색으로 그립니다.
                            cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1) 

                    else:
                        # 클러스터링 실패 시, bbox 내 모든 포인트의 평균 거리라도 사용
                        self.get_logger().warn("DBSCAN failed to find a cluster. Using all points in bbox as fallback.")
                        if points_in_bbox.shape[0] > 0:
                            all_ranges = np.sqrt(points_in_bbox[:, 0]**2 + points_in_bbox[:, 1]**2)
                            avg_distance = np.mean(all_ranges)
                            for pt in img_pts_in_bbox:
                                cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (128, 128, 128), -1) # 실패: 회색

            # ================================ LiDAR 포인트 클라우드 클러스터링 ================================

            # ================================ 장애물 차량 State 판별 (외적) ================================
            # 거리 계산이 성공했고, 중앙선 정보가 있을 때만 좌/우 판단 수행
            if avg_distance > 0 and has_curve:
                position_str_raw = "unknown"
                # 거리에 따라 판단 로직을 전환
                if avg_distance < self.PROXIMITY_THRESHOLD_M:
                    # ====== Rule 2: 초근접 상황 (0.8m 미만) ======
                    # 단순 x좌표 비교 로직으로 전환 (가장 직관적이고 확실함)
                    self.get_logger().warn(f"Proximity Override! Dist: {avg_distance:.2f}m. Using X-coord comparison.")
                    
                    bbox_center_x = (xmin + xmax) / 2
                    bbox_center_y = (ymin + ymax) / 2
                    path_x_at_bbox_y = np.interp(bbox_center_y, curve_y_coords, curve_x_coords)
                    
                    position_str_raw = "right" if bbox_center_x > path_x_at_bbox_y else "left"

                else:
                    # ====== Rule 1: 일반적인 원거리 상황 ======
                    # 안정적인 벡터 외적 로직 사용
                    p_obs = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
                    p1 = np.array([w / 2, h - 1])
                    path_x_at_obstacle_y = np.interp(ymax, curve_y_coords, curve_x_coords)
                    p2 = np.array([path_x_at_obstacle_y, ymax])

                    v_path = p2 - p1
                    v_obs = p_obs - p1
                    
                    cross_product = v_path[0] * v_obs[1] - v_path[1] * v_obs[0]
                    
                    if cross_product > 0:
                        position_str_raw = "right"
                    elif cross_product < 0:
                        position_str_raw = "left"


                current_numeric_pos = 0.0
                if position_str_raw == "left": current_numeric_pos = -1.0
                elif position_str_raw == "right": current_numeric_pos = 1.0

                self.smoothed_obstacle_position = (self.EMA_ALPHA * current_numeric_pos) + ((1 - self.EMA_ALPHA) * self.smoothed_obstacle_position)

                # 3. 필터링된 값을 최종 위치로 변환 ('center' 상태 제거)
                if self.smoothed_obstacle_position < -0.25:
                    position_str_final = "left"
                elif self.smoothed_obstacle_position > 0.25:
                    position_str_final = "right"
                else:
                    position_str_final = "unknown" # 중간값은 'unknown'으로 처리

                self.get_logger().info(f"Raw: {position_str_raw} -> Smoothed Val: {self.smoothed_obstacle_position:.2f} -> Final: {position_str_final}")

            else:
                # 거리 인식이 안 된 경우, 필터 값을 리셋하여 다음 탐지에 영향 없도록 함
                self.smoothed_obstacle_position = 0.0
                # position_str_final은 초기값인 "unknown"을 그대로 유지 
            # ================================ 장애물 차량 State 판별 (외적) ================================

            if position_str_final != "unknown":
                processed_obstacles.append({'distance': avg_distance, 'position': position_str_final, 'xmin': xmin, 'ymin': ymin, 'points_in_path_count': points_in_path_count})
                cv2.putText(image, f"{avg_distance:.1f}m", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # 최종 필터링된 결과값을 화면에 표시
                cv2.putText(image, position_str_final, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

       # ================================ E_STOP ================================


        cv2.rectangle(image, (self.E_STOP_RECT_X_MIN, self.E_STOP_RECT_Y_MIN), (self.E_STOP_RECT_X_MAX, self.E_STOP_RECT_Y_MAX), (0, 0, 255), 1)
        is_pixel_emergency = False
        if img_pts.shape[0] > 0 and len(detections) > 0:
            points_in_danger_zone = img_pts[
                (img_pts[:, 0] > self.E_STOP_RECT_X_MIN) & (img_pts[:, 0] < self.E_STOP_RECT_X_MAX) &
                (img_pts[:, 1] > self.E_STOP_RECT_Y_MIN) & (img_pts[:, 1] < self.E_STOP_RECT_Y_MAX)
            ]
            
            if len(points_in_danger_zone) > 0:
                for det in detections:
                    xmin, ymin, xmax, ymax = int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)
                    points_in_both_zones = points_in_danger_zone[
                        (points_in_danger_zone[:, 0] > xmin) & (points_in_danger_zone[:, 0] < xmax) &
                        (points_in_danger_zone[:, 1] > ymin) & (points_in_danger_zone[:, 1] < ymax)
                    ]

                    if len(points_in_both_zones) > self.E_STOP_RECT_PIXEL_COUNT_THRESHOLD:
                        is_pixel_emergency = True
                        self.get_logger().error(f"🚨 RECT E-STOP TRIGGER! Found {len(points_in_both_zones)} points in both E-Stop Rect and Vehicle BBox.")
                        cv2.rectangle(image, (self.E_STOP_RECT_X_MIN, self.E_STOP_RECT_Y_MIN), (self.E_STOP_RECT_X_MAX, self.E_STOP_RECT_Y_MAX), (0, 0, 255), 3)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                        break

        is_obstacle_emergency = False
        emergency_obstacle = None
        for obs in processed_obstacles:
            # 현재 나의 주행 경로 추정
            is_center_path = self.last_override_cmd == 'reset'
            is_left_path = self.last_override_cmd == 'go_left'
            is_right_path = self.last_override_cmd == 'go_right'
                
            # 충돌 경로에 있는지 확인
            # (예: 장애물이 왼쪽에 있는데, 내가 중앙이나 왼쪽으로 가려 할 때)
            collision_path = (obs['position'] == 'left' and (is_center_path or is_left_path)) or \
                                (obs['position'] == 'right' and (is_center_path or is_right_path))

            # E-Stop 조건: 1.거리가 가깝고, 2.내 주행 경로 중앙에 있고, 3.충돌 방향일 때
            if obs['distance'] < self.E_STOP_DISTANCE_M and obs['points_in_path_count'] > 0 and collision_path:
                is_obstacle_emergency = True
                emergency_obstacle = obs
                self.get_logger().error(f"🚨 E-STOP TRIGGER! Obstacle at {obs['distance']:.2f}m on a collision path.")
                break # 하나라도 감지되면 즉시 중단

        # ================================ E_STOP ================================


        # ================================ 추월 로직 시작 ================================

        obstacle_msg = ObstacleState()
        target_obstacle = None
        current_override_cmd = self.last_override_cmd
        num_detected_vehicles = len(detections) # YOLO가 탐지한 차량 총 수

        is_curve = curve_msg and curve_msg.state.lower() == 'curve'

        # [상태 1] 기본 (CENTER_DRIVING)
        if self.driving_state == "CENTER_DRIVING":
            # [전환 조건 1] 차량 2대 이상 감지 (직선/커브 무관)
            if num_detected_vehicles >= 2:
                self.driving_state = "Time_Gap"
                current_override_cmd = 'reset'
                self.get_logger().warn("STATE CHANGE: CENTER_DRIVING -> Time_Gap (2+ vehicles).")
            
            # [전환 조건 2] 차량 1대 감지
            elif num_detected_vehicles == 1 and processed_obstacles:
                obstacle = processed_obstacles[0]
                # (직선, 1): 거리가 가까우면 즉시 추월
                if not is_curve and obstacle['distance'] < self.OVERTAKE_THRESHOLD_M:
                    self.driving_state = "OVERTAKING"
                    self.is_passing_obstacle = False
                    self.pass_complete_timer = None
                    target_obstacle = obstacle
                    current_override_cmd = 'go_left' if target_obstacle['position'] == 'right' else 'go_right'
                    self.get_logger().warn(f"STATE CHANGE (Straight): CENTER_DRIVING -> OVERTAKING ({current_override_cmd}).")
                # (커브, 1): 추월하지 않고 Time_Gap으로 전환하여 추종 시
                elif is_curve:
                    self.driving_state = "Time_Gap"
                    current_override_cmd = 'reset'
                    self.get_logger().warn("STATE CHANGE (Curve): CENTER_DRIVING -> Time_Gap. Start following.")
                # 그 외 (직선에서 차가 멀리 있을 때)는 중앙 주행 유지
                else:
                    current_override_cmd = 'reset'
            
            # [전환 조건 3] 차량 없음: 중앙 주행 유지
            else:
                current_override_cmd = 'reset'

        # [상태 2] 추월 중 (OVERTAKING)
        elif self.driving_state == "OVERTAKING":
            if processed_obstacles:
                target_obstacle = min(processed_obstacles, key=lambda obs: obs['distance'])
            current_override_cmd = self.last_override_cmd # 추월 중에는 차선 변경 명령 유지

            if not self.is_passing_obstacle:
                is_side_detected = (self.last_override_cmd == 'go_left' and right_ultra < self.ULTAR_THRESHOLD) or \
                                   (self.last_override_cmd == 'go_right' and left_ultra < self.ULTAR_THRESHOLD)
                if is_side_detected:
                    self.is_passing_obstacle = True
                    self.get_logger().info("Side of obstacle detected. Now passing alongside.")
            else:
                is_side_cleared = (self.last_override_cmd == 'go_left' and right_ultra > self.ULTAR_THRESHOLD and right_back_ultra > self.ULTAR_BACK_THRESHOLD) or \
                                  (self.last_override_cmd == 'go_right' and left_ultra > self.ULTAR_THRESHOLD and left_back_ultra > self.ULTAR_BACK_THRESHOLD)
                if is_side_cleared and self.pass_complete_timer is None:
                    self.pass_complete_timer = self.get_clock().now()
                    self.get_logger().info(f"Side of obstacle cleared. Starting {self.PASS_TIMER_DURATION_S}s safety timer.")
                
                if self.pass_complete_timer is not None:
                    duration = self.get_clock().now() - self.pass_complete_timer
                    if duration.nanoseconds / 1e9 > self.PASS_TIMER_DURATION_S:
                        self.driving_state = "CENTER_DRIVING"
                        current_override_cmd = 'reset'
                        self.get_logger().warn("STATE CHANGE: OVERTAKING -> CENTER_DRIVING (Overtake complete).")
                        self.is_passing_obstacle = False
                        self.pass_complete_timer = None

        # [상태 3] 다중 차량 추종 또는 커브길 단일 차량 추종 (Time_Gap)
        elif self.driving_state == "Time_Gap":
            # [전환 조건 1] (커브, 1) 상황에서 추월 개시
            if is_curve and num_detected_vehicles == 1 and processed_obstacles:
                target_obstacle = processed_obstacles[0]
                if target_obstacle['distance'] <= self.CURVE_OVERTAKE_THRESHOLD_M:
                    self.driving_state = "OVERTAKING"
                    self.is_passing_obstacle = False
                    self.pass_complete_timer = None
                    current_override_cmd = 'go_left' if target_obstacle['position'] == 'right' else 'go_right'
                    self.get_logger().warn(f"STATE CHANGE (Curve): Time_Gap -> OVERTAKING. Target close ({target_obstacle['distance']:.2f}m).")
                else:
                    current_override_cmd = 'reset' # 아직 멀어서 계속 추종
                    self.get_logger().info(f"Time_Gap (Curve): Following target at {target_obstacle['distance']:.2f}m.")
            
            # [전환 조건 2] (직선+곡선, 2대 이상) 상황에서, 차간 거리가 충분하면 추월 개시
            elif num_detected_vehicles >= 2 and len(processed_obstacles) >= 2:

                sorted_obstacles = sorted(processed_obstacles, key=lambda obs: obs['distance'])
                closest_car = sorted_obstacles[0]
                second_closest_car = sorted_obstacles[1]
                
                distance_diff = second_closest_car['distance'] - closest_car['distance']

                if distance_diff >= self.MULTI_VEHICLE_OVERTAKE_GAP_M:
                    self.driving_state = "OVERTAKING"
                    self.is_passing_obstacle = False
                    self.pass_complete_timer = None
                    target_obstacle = closest_car
                    current_override_cmd = 'go_left' if target_obstacle['position'] == 'right' else 'go_right'
                    self.get_logger().warn(f"STATE CHANGE (Multi-Vehicle): Time_Gap -> OVERTAKING. Gap is sufficient ({distance_diff:.2f}m >= {self.MULTI_VEHICLE_OVERTAKE_GAP_M}m).")
                else:
                    current_override_cmd = 'reset'
                    target_obstacle = closest_car
                    self.get_logger().info(f"Time_Gap (Multi-Vehicle): Gap is tight ({distance_diff:.2f}m < {self.MULTI_VEHICLE_OVERTAKE_GAP_M}m). Following closest car.")
            # [전환 조건 3] Time_Gap 상태 탈출 조건
            elif (not is_curve and num_detected_vehicles < 2) or num_detected_vehicles == 0:
                self.driving_state = "CENTER_DRIVING"
                current_override_cmd = 'reset'
                self.get_logger().warn("STATE CHANGE: Time_Gap -> CENTER_DRIVING (Exit condition met).")
            
            # [상태 유지] 계속 Time_Gap 상태 유지
            else:
                current_override_cmd = 'reset'
                if processed_obstacles:
                    target_obstacle = min(processed_obstacles, key=lambda obs: obs['distance'])
                    self.get_logger().info(f"Time_Gap: Following closest vehicle at {target_obstacle['distance']:.1f}m.")
                else:
                    self.get_logger().info("Time_Gap: Vehicle(s) detected, but no distance info. Driving straight.")

        if current_override_cmd != self.last_override_cmd:
            cmd_msg = String(data=current_override_cmd)
            self.override_pub.publish(cmd_msg)
            self.get_logger().warn(f"💡 Lane command sent: '{current_override_cmd}'")
            self.last_override_cmd = current_override_cmd
            
        # ▼▼▼ 최종 메시지 발행 전, E-Stop 조건 확인 및 메시지 덮어쓰기 ▼▼▼
        if is_obstacle_emergency or is_pixel_emergency:
            obstacle_msg.distance_m = 0.0  # 컨트롤러에게 보내는 E-Stop 신호!
            
            if is_obstacle_emergency:
                obstacle_msg.position = emergency_obstacle['position']
            else: # is_pixel_emergency인 경우
                obstacle_msg.position = "pixel_danger" # E-Stop 원인을 명시

            if self.driving_state != "E_STOP":
                 self.get_logger().error("STATE CHANGE: ANY -> E_STOP")
                 self.driving_state = "E_STOP"
        else:
            if self.driving_state == "E_STOP":
                self.driving_state = "CENTER_DRIVING"
                self.get_logger().warn("STATE CHANGE: E_STOP -> CENTER_DRIVING (Obstacle Cleared).")

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
        #self.get_logger().info('--- 동기화 콜백 실행됨! ---')
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
                
        if curve_msg.points:
            curve_pts = np.array([[p.x, p.y] for p in curve_msg.points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [curve_pts], isClosed=False, color=(255, 0, 255), thickness=2)
        if curve_msg.state:
            cv2.putText(frame, f"Lane State: {curve_msg.state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        vehicle_count = sum(1 for det in yolo_msg.detections if det.class_name == 'obstacle_vehicle')
        for det in yolo_msg.detections:
            if det.class_name == 'obstacle_vehicle':
                x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 현재 주행 상태를 화면에 표시
        cv2.putText(frame, f"State: {self.driving_state}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Ultra: {left_ultra}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Back Ultra: {left_back_ultra}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right Ultra: {right_ultra}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right Back Ultra: {right_back_ultra}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        frame = self.project_lidar_on_image(frame, yolo_msg, points_3d, curve_msg,left_ultra,right_ultra,left_back_ultra,right_back_ultra,img_pts)
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