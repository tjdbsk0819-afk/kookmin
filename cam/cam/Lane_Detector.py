#! /usr/bin/env python
# -*- coding:utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import numpy as np
from collections import deque
from custom_interfaces.msg import XycarState, Detections
import message_filters
from std_srvs.srv import Trigger # <<< 1. Trigger 서비스 타입 임포트

# ===============================
# 13pixels = 2.5cm (차선 폭)
# 82.5cm (좌우차선 중앙을 끝점으로)
# 41.25cm = 214.5pixels
# ===============================
clicked_points = []



class Lane_Detector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # --- 기존 Subscriber 및 Publisher 설정 (변경 없음) ---
        self.trigger_sub = self.create_subscription(String,'/lane_override_cmd', self.trigger_callback, 10)
        self.img_sub = message_filters.Subscriber(self, Image, '/image_raw')
        self.detections_sub = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.detections_sub],
            queue_size=10, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.sync_callback)
        self.xycar_state_pub = self.create_publisher(XycarState, '/xycar_state', 10)

        # ### <<< 2. 서비스 서버 생성 및 상태 플래그 초기화
        self.is_active = False  # 메인 로직 실행 여부를 결정하는 플래그
        self.start_service = self.create_service(
            Trigger,
            '/start_track_driving',  # 서비스 이름
            self.start_service_callback  # 서비스 요청 시 호출될 콜백 함수
        )

        self.bridge = CvBridge()
        self.get_logger().info('🛣️  Lane_Detector Node Started Successfully❗')
        self.get_logger().info('Waiting for service call on /start_lane_detection to begin processing...')

        # 0. 초기값
        self.angle = 0.0
        self.lane = np.array([120.0, 320.0, 520.0])  # 초기 왼쪽, 중앙, 오른쪽 차선의 x좌표

        # 1. mask_detected_objects
        self.last_seen_map = {}       # Key: grid_cell, Value: {'last_frame', 'bbox'}
        self.grid_cell_size = 50      # 객체 중심점 기준 동일 객체로 판단할 격자 크기 (px)
        self.frame_counter = 0        # 프레임 카운터

        # 2. apply_canny
        self.canny_low = 50
        self.canny_high = 150

        # 3. detect_lines_hough 
        self.hough_threshold = 25
        self.min_gap = 1
        self.min_length = 10

        # 4. filter_lines_by_angle
        self.angle_tolerance = np.radians(30)       # y=60px에서 현재 차선 방향과의 허용 각도 차이 (degree)
        self.prev_angle = deque([0.0], maxlen=3)    # (6 에서도 사용됨)

        # 5. extract_lane_candidates_from_clusters
        self.cluster_threshold = 30                 # 위치값들을 클러스터로 묶을 때, 평균값과의 최대 허용 거리(임계값)

        # 6. predict_lane
        self.left_offset = -200
        self.right_offset = 200
        self.angle_correction_gain = 80             # 예측 차선 위치 보정 시 각도 변화량에 곱해질 보정 계수 
        self.min_cos_angle = 0.5                    # (7 에서도 사용됨) 차선 각도가 너무 기울어질 경우를 방지하기 위한 최소 cos(각도) 값
        self.angle_prev_weight = 0.7                # 이전 각도에 대한 가중치
        self.angle_new_weight  = 0.3                # 현재 프레임 각도에 대한 가중치

        # 7. refine_lane_with_candidates_and_prediction
        self.left_to_center_dist = 200
        self.right_to_center_dist = 200
        self.outer_lane_dist = self.left_to_center_dist + self.right_to_center_dist
        self.cluster_match_threshold = 70           ### 예상된 차선 위치와 실제 후보 간의 허용 거리
        self.lane_update_weight = 0.7               ### 최종 차선 위치 계산 시 현재 관측값에 대한 반영 비율
        self.prediction_weight = 0.3                ### 최종 차선 위치 계산 시 예측값에 대한 반영 비율
        self.max_lane_delta = 50                    # 한 프레임에서 허용되는 최대 차선 변화량 [픽셀]

        # 8. trigger_callback
        self.override_target_lane = None            # 'go_right', 'go_left' 등

    # ### <<< 3. 서비스 콜백 함수 추가
    def start_service_callback(self, request, response):
        """
        서비스 호출을 받으면 is_active 플래그를 True로 설정하고,
        메인 로직(sync_callback)이 실행되도록 합니다.
        """
        if not self.is_active:
            self.is_active = True
            self.get_logger().info('✅ Service called. Starting lane detection main logic!')
            response.success = True
            response.message = 'Lane detection started.'
        else:
            self.get_logger().warn('⚠️ Main logic is already running.')
            response.success = False
            response.message = 'Already active.'

        return response

    ### ===== 원근 변환을 통해 영상을 탑다운 시점의 Bird's Eye View로 변환한다 =====
    def get_birds_eye_view(self, image):
        height, width = image.shape[:2]
        output_width = 640
        output_height = 120
        src = np.float32([
            [width * 0.16, height * 0.64],      # TOP-LEFT
            [width * 0.84, height * 0.64],      # TOP-RIGHT
            [width * 1.00, height * 0.72],      # BOT-RIGHT
            [width * 0.00, height * 0.72]       # BOT-LEFT
        ])
        dst = np.float32([
            [output_width * 0.1, 0],
            [output_width * 0.9, 0],
            [output_width * 0.9, output_height],
            [output_width * 0.1, output_height]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        bev = cv2.warpPerspective(image, M, (output_width, output_height))
        self.bev_img_w = output_width
        self.bev_img_h = output_height
        self.bev_img_mid = output_height // 2
        return bev, src

    ### ===== 가우시안 블러 후 캐니 알고리즘으로 이미지의 에지(윤곽선)를 검출한다 =====
    def apply_canny(self, img, show=False):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7,7), 0)
        img = cv2.Canny(img, self.canny_low, self.canny_high)
        if show:
            cv2.imshow('canny', img)
        return img

    ### ===== 감지된 객체 정보를 바탕으로 Canny 이미지의 특정 영역을 조건부로 마스킹한다 =====
    def mask_detected_objects(self, canny_img, detections_msg, original_frame, show=False):
        self.frame_counter += 1
        modified_canny = canny_img.copy()
        
        if show:
            viz_img = original_frame.copy()
            final_mask_for_viz = np.zeros_like(modified_canny)

        # --- 1. 현재 프레임 detection 정보 분리 ---
        active_target_bboxes = set()
        current_grid_cells = set()
        
        current_target_detections = [d for d in detections_msg.detections if d.class_name in ["checkerboard", "obstacle_vehicle"]]
        lane_detections = [d for d in detections_msg.detections if d.class_name == "center_lane"]

        # --- 2. 현재 감지된 객체 처리 및 추적기 업데이트 ---
        for det in current_target_detections:
            bbox = (det.xmin, det.ymin, det.xmax, det.ymax)
            active_target_bboxes.add(bbox)
            
            center_x = (det.xmin + det.xmax) // 2
            center_y = (det.ymin + det.ymax) // 2
            grid_cell = (center_x // self.grid_cell_size, center_y // self.grid_cell_size)
            current_grid_cells.add(grid_cell)

            # 해당 격자의 마지막 감지 정보를 현재 프레임으로 갱신
            self.last_seen_map[grid_cell] = {
                'last_frame': self.frame_counter,
                'bbox': bbox
            }

        # --- 3. 직전 프레임 대비 사라진 객체 보간 및 추적기 정리 ---
        for cell, data in list(self.last_seen_map.items()):
            # 직전 프레임에 있었지만 현재 프레임에 없는 경우
            if self.frame_counter - 1 == data['last_frame'] and cell not in current_grid_cells:
                active_target_bboxes.add(data['bbox']) # 보간할 bbox 추가
            
            # 2프레임 이상 오래된 기록은 추적기에서 삭제
            if data['last_frame'] < self.frame_counter - 1:
                del self.last_seen_map[cell]

        # --- 4. 최종 마스킹 및 시각화 처리 ---
        final_bboxes_to_mask = list(active_target_bboxes)
        for t_bbox in final_bboxes_to_mask:
            t_xmin, t_ymin, t_xmax, t_ymax = t_bbox
            
            overlapping_lanes_bboxes = []
            for lane_det in lane_detections:
                l_xmin, l_ymin, l_xmax, l_ymax = lane_det.xmin, lane_det.ymin, lane_det.xmax, lane_det.ymax
                if not (t_xmax < l_xmin or t_xmin > l_xmax or t_ymax < l_ymin or t_ymin > l_ymax):
                    overlapping_lanes_bboxes.append((l_xmin, l_ymin, l_xmax, l_ymax))

            current_mask = np.zeros_like(modified_canny)
            if not overlapping_lanes_bboxes:
                cv2.rectangle(current_mask, (t_xmin, t_ymin), (t_xmax, t_ymax), 255, -1)
            else:
                cv2.rectangle(current_mask, (t_xmin, t_ymin), (t_xmax, t_ymax), 255, -1)
                for l_bbox in overlapping_lanes_bboxes:
                    cv2.rectangle(current_mask, (l_bbox[0], l_bbox[1]), (l_bbox[2], l_bbox[3]), 0, -1)

            modified_canny[current_mask == 255] = 0
            
            if show:
                final_mask_for_viz = cv2.bitwise_or(final_mask_for_viz, current_mask)

        if show:
            viz_img[final_mask_for_viz == 255] = (0, 0, 0)
            contours, _ = cv2.findContours(final_mask_for_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz_img, contours, -1, (0, 0, 255), 2)
            
            cv2.imshow("Object Masking Visualization", viz_img)
            cv2.imshow("Masked Canny Image", modified_canny)
            
        return modified_canny


    ### ===== 확률적 허프 변환을 이용해 이미지에서 직선(선분)들을 검출한다 =====
    def detect_lines_hough(self, img, show=False):
        lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
        if show:
            hough_img = np.zeros((img.shape[0], img.shape[1], 3))
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('hough', hough_img)
        return lines

    ### ===== 현재 차선 각도와 유사한 직선들만 필터링하여 차선 후보 위치(x좌표)를 추출한다 =====
    def filter_lines_by_angle(self, lines, show=True):
        thetas, positions = [], []
        if show:
            filter_img = np.zeros((self.bev_img_h, self.bev_img_w, 3))
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if y1 == y2:
                    continue
                flag = 1 if y1-y2 > 0 else -1
                theta = np.arctan2(flag * (x2-x1), flag * (y1-y2))      # 허프 변환으로 검출된 두 끝점으로부터 계산
                if abs(theta - self.angle) < self.angle_tolerance:      # 현재 선분의 방향(theta)과 과거 차선 방향(self.angle)의 차이
                    position = float((x2-x1)*(self.bev_img_mid-y1))/(y2-y1) + x1    # 60px에서 각 직선의 위치 x (단일)
                    thetas.append(theta)
                    positions.append(position)                                      # 60px에서 각 직선의 위치 x (리스트)
                    if show:
                        cv2.line(filter_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.prev_angle.append(self.angle)
        if thetas:
            new_angle = np.mean(thetas)                                 # 이전 프레임에서 필터링된 모든 선의 각도 리스트 theta의 평균
            self.angle = self.angle_prev_weight * self.angle + self.angle_new_weight * new_angle
        if show:
            cv2.imshow('filtered lines by angle', filter_img)
        return positions

    ### ===== 위치값들을 클러스터링하여 평균값 기준의 차선 후보 좌표들을 추출한다 =====
    def extract_lane_candidates_from_clusters(self, positions, show=False, base_img=None):
        clusters = []                               # 여러 개의 cluster를 담은 전체 리스트 (2차원 리스트)
        for position in positions:
            if 0 <= position < self.bev_img_w:
                for cluster in clusters:            # 유사한 x좌표들을 모은 하나의 그룹 (1차원 리스트)
                    cluster_mean = np.mean(cluster)
                    if abs(cluster_mean - position) < self.cluster_threshold:
                        cluster.append(position)
                        break
                else:
                    clusters.append([position])
        lane_candidates = [np.mean(cluster) for cluster in clusters]

        if show and base_img is not None:
            cluster_img = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR)
            np.random.seed(42)
            colors = [tuple(map(int, np.random.randint(0, 255, size=3))) for _ in range(len(clusters))]
            for idx, cluster in enumerate(clusters):
                color = colors[idx]
                for x in cluster:
                    cv2.circle(cluster_img, (int(x), self.bev_img_mid), 4, color, -1)
            cv2.imshow('Clustered Lane Positions', cluster_img)
        return lane_candidates

    ### ===== 이전 차선 위치와 각도를 기반으로 현재 프레임의 차선 위치를 예측한다 =====
    def predict_lane(self, show=False, base_img=None):
        denom = max(np.cos(self.angle), self.min_cos_angle)
        predicted_lane = self.lane[1] + np.array([
            self.left_offset / denom,
            0,
            self.right_offset / denom
        ])
        predicted_lane += (self.angle - np.mean(self.prev_angle)) * self.angle_correction_gain

        if show and base_img is not None:
            img = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR) if len(base_img.shape) == 2 else base_img.copy()
            y = self.bev_img_mid
            cv2.circle(img, (int(predicted_lane[0]), y), 4, (255, 0, 255), 2)       # 좌측 예측 (보라)
            cv2.circle(img, (int(predicted_lane[1]), y), 4, (128, 128, 128), 2)     # 중앙 예측 (회색)
            cv2.circle(img, (int(predicted_lane[2]), y), 4, (255, 255, 0), 2)       # 우측 예측 (노랑)
            cv2.putText(img, 'Pred_L', (int(predicted_lane[0]) - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(img, 'Pred_C', (int(predicted_lane[1]) - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            cv2.putText(img, 'Pred_R', (int(predicted_lane[2]) - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.imshow('Predicted Lane', img)
        return predicted_lane
        
    ### ===== 예측된 차선과 후보 차선을 비교하여 가장 유사한 조합을 선택하고 최종 차선 위치를 보정한다 =====
    def refine_lane_with_candidates_and_prediction(self, lane_candidates, predicted_lane):
            # Override 모드에서 기준 차선이 화면을 벗어났는지 확인하고 자동 복귀
            if self.override_target_lane == 'go_left':
                if predicted_lane[0] < 0 or predicted_lane[0] >= self.bev_img_w:
                    self.get_logger().warn('⚠️ Left lane out of view, switching back to normal mode')
                    self.override_target_lane = None
            elif self.override_target_lane == 'go_right':
                if predicted_lane[2] < 0 or predicted_lane[2] >= self.bev_img_w:
                    self.get_logger().warn('⚠️ Right lane out of view, switching back to normal mode')
                    self.override_target_lane = None

            # Override 모드일 때는 특정 차선만 기준으로 처리
            if self.override_target_lane == 'go_left':
                # 왼쪽 차선만 기준으로 계산
                cos_angle = max(np.cos(self.angle), self.min_cos_angle)
                left_candidates = [lc for lc in lane_candidates if abs(lc - predicted_lane[0]) < self.cluster_match_threshold]
                
                if left_candidates:
                    # 가장 가까운 왼쪽 차선 후보 선택
                    best_left = min(left_candidates, key=lambda x: abs(x - predicted_lane[0]))
                    new_lane = np.array([
                        best_left,
                        best_left + self.left_to_center_dist / cos_angle,
                        best_left + self.outer_lane_dist / cos_angle
                    ])
                    # 부드러운 업데이트
                    delta = new_lane - self.lane
                    delta_clipped = np.clip(delta, -self.max_lane_delta, self.max_lane_delta)
                    self.lane = self.lane + delta_clipped
                else:
                    # 왼쪽 차선을 찾지 못했을 경우 예측값만 사용
                    self.lane = predicted_lane
                return
                    
            elif self.override_target_lane == 'go_right':
                # 오른쪽 차선만 기준으로 계산
                cos_angle = max(np.cos(self.angle), self.min_cos_angle)
                right_candidates = [lc for lc in lane_candidates if abs(lc - predicted_lane[2]) < self.cluster_match_threshold]
                
                if right_candidates:
                    # 가장 가까운 오른쪽 차선 후보 선택
                    best_right = min(right_candidates, key=lambda x: abs(x - predicted_lane[2]))
                    new_lane = np.array([
                        best_right - self.outer_lane_dist / cos_angle,
                        best_right - self.right_to_center_dist / cos_angle,
                        best_right
                    ])
                    # 부드러운 업데이트
                    delta = new_lane - self.lane
                    delta_clipped = np.clip(delta, -self.max_lane_delta, self.max_lane_delta)
                    self.lane = self.lane + delta_clipped
                else:
                    # 오른쪽 차선을 찾지 못했을 경우 예측값만 사용
                    self.lane = predicted_lane
                return

            # 원래 방식 (reset 모드 또는 초기 상태 또는 자동 복귀된 상태)
            if not lane_candidates:
                self.lane = predicted_lane
                return
            
            possibles = []
            cos_angle = max(np.cos(self.angle), self.min_cos_angle)
            for lc in lane_candidates:
                idx = np.argmin(abs(self.lane - lc))
                if idx == 0:
                    estimated_lane = [
                        lc,
                        lc + self.left_to_center_dist / cos_angle,
                        lc + self.outer_lane_dist / cos_angle
                    ]
                    lc2_candidate = [c for c in lane_candidates if abs(c - estimated_lane[1]) < self.cluster_match_threshold]
                    lc3_candidate = [c for c in lane_candidates if abs(c - estimated_lane[2]) < self.cluster_match_threshold]
                    if not lc2_candidate:
                        lc2_candidate.append(estimated_lane[1])
                    if not lc3_candidate:
                        lc3_candidate.append(estimated_lane[2])
                    for lc2 in lc2_candidate:
                        for lc3 in lc3_candidate:
                            possibles.append([lc, lc2, lc3])
                elif idx == 1:
                    estimated_lane = [
                        lc - self.left_to_center_dist / cos_angle,
                        lc,
                        lc + self.right_to_center_dist / cos_angle
                    ]
                    lc1_candidate = [c for c in lane_candidates if abs(c - estimated_lane[0]) < self.cluster_match_threshold]
                    lc3_candidate = [c for c in lane_candidates if abs(c - estimated_lane[2]) < self.cluster_match_threshold]
                    if not lc1_candidate:
                        lc1_candidate.append(estimated_lane[0])
                    if not lc3_candidate:
                        lc3_candidate.append(estimated_lane[2])
                    for lc1 in lc1_candidate:
                        for lc3 in lc3_candidate:
                            possibles.append([lc1, lc, lc3])
                else:
                    estimated_lane = [
                        lc - self.outer_lane_dist / cos_angle,
                        lc - self.right_to_center_dist / cos_angle,
                        lc
                    ]
                    lc1_candidate = [c for c in lane_candidates if abs(c - estimated_lane[0]) < self.cluster_match_threshold]
                    lc2_candidate = [c for c in lane_candidates if abs(c - estimated_lane[1]) < self.cluster_match_threshold]
                    if not lc1_candidate:
                        lc1_candidate.append(estimated_lane[0])
                    if not lc2_candidate:
                        lc2_candidate.append(estimated_lane[1])
                    for lc1 in lc1_candidate:
                        for lc2 in lc2_candidate:
                            possibles.append([lc1, lc2, lc])
            
            possibles = np.array(possibles)
            error = np.sum((possibles - predicted_lane) ** 2, axis=1)
            best = possibles[np.argmin(error)]
            new_lane = self.lane_update_weight * best + self.prediction_weight * predicted_lane

            delta = new_lane - self.lane
            delta_clipped = np.clip(delta, -self.max_lane_delta, self.max_lane_delta)
            self.lane = self.lane + delta_clipped
    ### ===== 최종 결정된 차선 좌표들과 그 중앙값들을 이미지에 시각화하여 표시한다 =====
    def mark_lane(self, img, lane=None, show=False):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if lane is None:
            lane = self.lane
        l1, l2, l3 = lane
        cv2.circle(img, (int(l1), self.bev_img_mid), 3, (255, 0, 0), 5, cv2.FILLED)     # BLUE
        cv2.circle(img, (int(l2), self.bev_img_mid), 3, (0, 255, 255), 5, cv2.FILLED)   # YELLOW
        cv2.circle(img, (int(l3), self.bev_img_mid), 3, (0, 0, 255), 5, cv2.FILLED)     # RED

        mid_left = (l1 + l2) / 2
        mid_right = (l2 + l3) / 2
        cv2.circle(img, (int(mid_left), self.bev_img_mid), 4, (0, 255, 0), 4, cv2.FILLED)   # GREEN
        cv2.putText(img, 'go_L', (int(mid_left)-20, self.bev_img_mid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.circle(img, (int(mid_right), self.bev_img_mid), 4, (0, 255, 0), 4, cv2.FILLED)  # GREEN
        cv2.putText(img, 'go_R', (int(mid_right)-20, self.bev_img_mid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Override 모드 표시
        if self.override_target_lane:
            cv2.putText(img, f'Override: {self.override_target_lane}', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if show:
            cv2.imshow('marked_lane_point', img)

    ### ===== 원본 이미지에 BEV 변환 영역을 폴리라인으로 표시하여 시각화한다 =====
    def show_roi_region(self, img, src, show=False):
        if show:
            COLOR = (0,255,0)
            THICKNESS = 2
            cv2.polylines(img, [src.astype(int)], True, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.imshow('roi region on the undistorted frame',img)
        else :
            pass

    ### ===== 수신된 이미지에서 차선을 인식하고, 최종 중앙 차선 위치와 각도를 계산하여 퍼블리시한다 =====
    def sync_callback(self, img_msg, detections_msg):
        # ### <<< 4. 메인 로직 실행 전, 활성화 상태 체크
        if not self.is_active:
            return  # 비활성화 상태이면 함수를 즉시 종료
            
        frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        canny = self.apply_canny(frame, show=False)
        
        # Canny 이미지에서 감지된 객체 영역을 조건부로 마스킹 (시각화 활성화)
        canny = self.mask_detected_objects(canny, detections_msg, frame, show=False)
        
        bev, src = self.get_birds_eye_view(canny)
        self.show_roi_region(frame, src, show=False)
        lines = self.detect_lines_hough(bev, show=False)
        positions = self.filter_lines_by_angle(lines, show=False)
        lane_candidates = self.extract_lane_candidates_from_clusters(positions,show=False,base_img=bev)

        predicted_lane = self.predict_lane(show=False, base_img=bev)
        self.refine_lane_with_candidates_and_prediction(lane_candidates, predicted_lane)
        self.mark_lane(bev, show=True)

        state_msg = XycarState()

        if self.override_target_lane == 'go_right':
            state_msg.drive_mode = 'right'
            target_x = (self.lane[1] + self.lane[2]) / 2
        elif self.override_target_lane == 'go_left':
            state_msg.drive_mode = 'left'
            target_x = (self.lane[0] + self.lane[1]) / 2
        else:
            state_msg.drive_mode = 'center'
            target_x = self.lane[1]

        state_msg.target_point.x = float(target_x)
        state_msg.target_point.y = float(self.bev_img_mid)
        state_msg.target_point.z = self.angle

        self.xycar_state_pub.publish(state_msg)

        cv2.waitKey(1)
        

    ### ===== 외부 명령(go_right/go_left)에 따라 차선 선택을 강제로 지정하거나 초기화한다 =====
    def trigger_callback(self, msg):  # std_msgs/String
        command = msg.data.strip().lower()

        # 1) override_target_lane 먼저 갱신
        if command == 'go_right':
            self.override_target_lane = 'go_right'
        elif command == 'go_left':
            self.override_target_lane = 'go_left'
        elif command == 'reset':
            self.override_target_lane = None
        else:
            self.get_logger().warn(f'⚠️ [Command Received] unknown: {command}')
            return
    
def main(args=None):
    rclpy.init(args=args)
    node = Lane_Detector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()