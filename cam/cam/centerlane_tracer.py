#! /usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================================
#
#   FILE: centerlane_tracer.py
#
#   DESCRIPTION:
#       YOLOv8 Bounding Box 탐지 결과를 기반으로 중앙 차선을 추적하는 ROS 2 노드.
#       디버그 모드를 통해 실시간으로 파라미터를 조절하며 시각화하거나,
#       Headless 모드로 UI 없이 최적화된 파라미터로 커브 토픽만 발행할 수 있습니다.
#
#   LOGIC FLOW:
#       1. 이미지 토픽('/image_raw')과 YOLO 탐지 토픽('/yolo_detections')을 동기화하여 수신.
#       2. Debug 모드가 켜져 있으면, UI 트랙바에서 실시간으로 파라미터를 읽어와 업데이트.
#          - 꺼져 있으면, 코드에 설정된 기본 파라미터 값을 사용.
#       3. YOLO 탐지 결과 중 'center_line' 클래스만 필터링.
#       4. 각 BBox 내에서 이미지 처리(Adaptive Threshold -> Canny Edge -> Hough Line Transform)를
#          수행하여 가장 긴 선분을 찾고, 선분과 BBox의 교차점을 '대표점'으로 추출.
#          - 선분 탐지 실패 시 BBox의 중앙점을 fallback으로 사용.
#       5. 모든 대표점들을 화면 아래쪽부터 위쪽으로 Greedy 알고리즘을 통해 정렬.
#          - 이때 매우 가까운(2px 이내) 점들은 하나의 점으로 통합하여 노이즈를 제거.
#       6. 정렬된 대표점들을 기반으로 'spline' 또는 'linear' 모드로 커브 피팅.
#          - 양 끝점은 이미지 경계까지 연장하여 경로를 생성.
#       7. 최종 생성된 커브(점들의 배열)를 '/center_curve' 토픽으로 발행.
#       8. Debug 모드가 켜져 있으면, 모든 처리 과정과 최종 결과를 화면에 시각화하여 출력.
#
# ==================================================================================


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from custom_interfaces.msg import Detections, Curve, Point2D
import message_filters
import math

from scipy.interpolate import splprep, splev

class CenterlaneTracer(Node):
    """
    대표점들에 대해 적응형 곡선 피팅을 수행하고,
    실시간 파라미터 조정을 위한 트랙바와 디버그 창을 제공합니다.
    Debug 모드가 False일 경우, UI 없이 설정된 기본값으로 커브 토픽만 발행합니다.
    """
    def __init__(self):
        super().__init__('centerlane_tracer')

        # =======================================================
        # True로 설정 시 디버그 창 활성화, False로 설정 시 Headless 모드
        self.debug = True
        # =======================================================

        self.HEIGHT_THRESHOLD = 220
        self.bridge = CvBridge()
        self.current_frame = None

        self._initialize_parameters()
        
        # 차선 상태 유지를 위한 변수 초기화
        self.lane_state = "Straight"
        self.straight_frames_counter = 0

        self.image_subscription = message_filters.Subscriber(self, Image, '/image_raw')
        self.yolo_subscription = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.yolo_subscription],
            queue_size=10, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        self.curve_publisher = self.create_publisher(Curve, '/center_curve', 10)

        if self.debug:
            self.main_window_name = 'Centerlane Tracer'
            self.debug_window_names = ['Grayscale (AdaptiveThresh)', 'Canny Edges', 'Hough Lines']
            cv2.namedWindow(self.main_window_name, cv2.WINDOW_AUTOSIZE)
            for name in self.debug_window_names:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)

            cv2.setMouseCallback(self.main_window_name, self.mouse_event_callback)
            
            def nothing(x): pass
            cv2.createTrackbar('Adapt Method', self.main_window_name, self.params['adapt_method_val'], 1, nothing)
            cv2.createTrackbar('Block Size', self.main_window_name, self.params['block_size_val'], 20, nothing)
            cv2.createTrackbar('C Constant', self.main_window_name, self.params['c_constant_val'], 50, nothing)
            cv2.createTrackbar('Canny Thresh1', self.main_window_name, self.params['canny_thresh1'], 255, nothing)
            cv2.createTrackbar('Canny Thresh2', self.main_window_name, self.params['canny_thresh2'], 255, nothing)
            cv2.createTrackbar('Hough Threshold', self.main_window_name, self.params['hough_threshold'], 100, nothing)
            cv2.createTrackbar('Hough MinLineLength', self.main_window_name, self.params['hough_min_len'], 100, nothing)
            cv2.createTrackbar('Hough MaxLineGap', self.main_window_name, self.params['hough_max_gap'], 100, nothing)
            cv2.createTrackbar('Similarity Thresh', self.main_window_name, self.params['similarity_thresh'], 50, nothing)
            cv2.createTrackbar('Fit Mode', self.main_window_name, self.params['fit_mode_val'], 1, nothing)
            
            cv2.createTrackbar('Draw BBox', self.main_window_name, int(self.params['draw_bbox']), 1, nothing)
            cv2.createTrackbar('Draw Points', self.main_window_name, int(self.params['draw_points']), 1, nothing)
            cv2.createTrackbar('Draw Numbers', self.main_window_name, int(self.params['draw_numbers']), 1, nothing)
            cv2.createTrackbar('Draw Hough', self.main_window_name, int(self.params['draw_hough']), 1, nothing)
            cv2.createTrackbar('Draw Curve', self.main_window_name, int(self.params['draw_curve']), 1, nothing)
            
            cv2.createTrackbar('Curve Thresh', self.main_window_name, self.params['curve_thresh'], 100, nothing)
            cv2.createTrackbar('Straight Frame Thresh', self.main_window_name, self.params['straight_thresh_frame'], 20, nothing)

            self.get_logger().info("Debug mode ON. Trackbars are enabled.")
        else:
            self.get_logger().info("Debug mode OFF. Running in headless mode.")

        self.get_logger().info('CenterlaneTracer has been started.')

    def _initialize_parameters(self):
        """
        모든 파라미터의 기본값을 중앙에서 관리합니다.
        Headless 모드에서는 이 값들이 고정값으로 사용됩니다.
        """
        self.params = {
            # --- 이미지 처리 파라미터 ---
            'adapt_method_val': 1,
            'block_size_val': 5,
            'c_constant_val': 5,
            'canny_thresh1': 50,
            'canny_thresh2': 150,
            'hough_threshold': 20,
            'hough_min_len': 20,
            'hough_max_gap': 5,

            # --- 대표점 정렬 및 커브 피팅 파라미터 ---
            'similarity_thresh': 25,
            'fit_mode_val': 1,

            # --- 시각화 옵션 ---
            'draw_bbox': True,
            'draw_points': True,
            'draw_numbers': True,
            'draw_hough': True,
            'draw_curve': True,

            # --- 커브/직선 판단 파라미터 ---
            'curve_thresh': 98,
            'straight_thresh_frame': 10,
        }
    
    def _update_params_from_trackbars(self):
        """디버그 모드일 때 트랙바에서 파라미터를 읽어와 self.params를 업데이트합니다."""
        self.params['adapt_method_val'] = cv2.getTrackbarPos('Adapt Method', self.main_window_name)
        self.params['block_size_val'] = cv2.getTrackbarPos('Block Size', self.main_window_name)
        self.params['c_constant_val'] = cv2.getTrackbarPos('C Constant', self.main_window_name)
        self.params['canny_thresh1'] = cv2.getTrackbarPos('Canny Thresh1', self.main_window_name)
        self.params['canny_thresh2'] = cv2.getTrackbarPos('Canny Thresh2', self.main_window_name)
        self.params['hough_threshold'] = cv2.getTrackbarPos('Hough Threshold', self.main_window_name)
        self.params['hough_min_len'] = cv2.getTrackbarPos('Hough MinLineLength', self.main_window_name)
        self.params['hough_max_gap'] = cv2.getTrackbarPos('Hough MaxLineGap', self.main_window_name)
        self.params['similarity_thresh'] = cv2.getTrackbarPos('Similarity Thresh', self.main_window_name)
        self.params['fit_mode_val'] = cv2.getTrackbarPos('Fit Mode', self.main_window_name)
        self.params['draw_bbox'] = cv2.getTrackbarPos('Draw BBox', self.main_window_name) == 1
        self.params['draw_points'] = cv2.getTrackbarPos('Draw Points', self.main_window_name) == 1
        self.params['draw_numbers'] = cv2.getTrackbarPos('Draw Numbers', self.main_window_name) == 1
        self.params['draw_hough'] = cv2.getTrackbarPos('Draw Hough', self.main_window_name) == 1
        self.params['draw_curve'] = cv2.getTrackbarPos('Draw Curve', self.main_window_name) == 1
        self.params['curve_thresh'] = cv2.getTrackbarPos('Curve Thresh', self.main_window_name)
        self.params['straight_thresh_frame'] = cv2.getTrackbarPos('Straight Frame Thresh', self.main_window_name)


    def mouse_event_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_frame is not None:
            bgr_pixel = self.current_frame[y, x]
            hsv_pixel = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
            self.get_logger().info(f'Click at ({x}, {y}) | BGR: {bgr_pixel} -> HSV: {hsv_pixel}')

    def synchronized_callback(self, image_msg, yolo_msg):
        if self.debug:
            self._update_params_from_trackbars()
        
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            if self.debug:
                self.current_frame = frame.copy()
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        h, w, _ = frame.shape
        
        proc_params = {
            'adapt_method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C if self.params['adapt_method_val'] == 1 else cv2.ADAPTIVE_THRESH_MEAN_C,
            'block_size': self.params['block_size_val'] * 2 + 3,
            'c_constant': self.params['c_constant_val'] - 25,
            'canny_thresh1': self.params['canny_thresh1'],
            'canny_thresh2': self.params['canny_thresh2'],
            'hough_threshold': self.params['hough_threshold'],
            'hough_min_len': self.params['hough_min_len'],
            'hough_max_gap': self.params['hough_max_gap'],
            'similarity_thresh': self.params['similarity_thresh'],
        }
        fit_mode = 'linear' if self.params['fit_mode_val'] == 1 else 'spline'

        # --- 핵심 로직 (모드와 관계없이 항상 실행) ---
        filtered_detections = [
            det for det in yolo_msg.detections
            if det.class_name == 'center_line' and det.ymax > self.HEIGHT_THRESHOLD
        ]
        all_rep_points_unsorted, lines_to_draw, debug_images = self._extract_representative_points(frame, filtered_detections, proc_params)
        sorted_rep_points = self._sort_representative_points(all_rep_points_unsorted, proc_params)
        curve_points_to_draw = self.fit_curve_to_points(
            sorted_rep_points,
            mode=fit_mode, 
            frame_height=h,
            frame_width=w
        )

        ratio = 1.0 # ratio 변수 초기화
        
        # ======================= [수정된 부분 시작] =========================
        if curve_points_to_draw:
            current_threshold = self.params['curve_thresh'] / 100.0
            current_frame_decision, ratio = self._get_current_frame_decision(curve_points_to_draw, current_threshold)
            self._update_lane_state(current_frame_decision)
            
            curve_msg = Curve()
            curve_msg.header = image_msg.header
            curve_msg.points = [Point2D(x=float(p[0]), y=float(p[1])) for p in curve_points_to_draw]
            curve_msg.state = self.lane_state 
            self.curve_publisher.publish(curve_msg)
        else:
            # 'center_line'이 탐지되지 않았거나 곡선을 그리지 못한 경우, 이전 프레임의 상태를 유지합니다.
            # self.lane_state를 업데이트하지 않으므로 이전 상태가 유지됩니다.
            
            curve_msg = Curve()
            curve_msg.header = image_msg.header # 타임스탬프 동기화를 위해 헤더는 필수!
            curve_msg.points = [] # points 리스트를 비워서 보냄
            curve_msg.state = self.lane_state # 이전 상태(self.lane_state)를 그대로 발행
            self.curve_publisher.publish(curve_msg)
        # ======================= [수정된 부분 끝] ===========================
        
        # --- 시각화 로직 (Debug 모드일 때만 실행) ---
        if self.debug:
            final_decision_text = f"State: {self.lane_state}"
            if curve_points_to_draw:
                current_threshold = self.params['curve_thresh'] / 100.0
                final_decision_text = f"State: {self.lane_state} (Ratio: {ratio:.2f} / Thresh: {current_threshold:.2f})"
            
            self.visualize_results(
                frame, 
                filtered_detections, 
                sorted_rep_points, 
                curve_points_to_draw, 
                lines_to_draw, 
                self.params,
                final_decision_text
            )
            for name, img in zip(self.debug_window_names, debug_images):
                if img is not None:
                    cv2.imshow(name, img)
    
    def _get_current_frame_decision(self, curve_pts, threshold):
        """
        '현재 프레임'의 점들만을 기반으로 직선인지 커브인지 판단합니다. (영문 반환)
        """
        if len(curve_pts) < 2:
            return "Straight", 1.0

        start_point = curve_pts[0]
        end_point = curve_pts[-1]
        end_to_end_distance = math.dist(start_point, end_point)
        path_length = sum(math.dist(curve_pts[i], curve_pts[i-1]) for i in range(1, len(curve_pts)))

        if path_length == 0:
            return "Straight", 1.0

        curvature_ratio = end_to_end_distance / path_length

        decision = "Curve" if curvature_ratio < threshold else "Straight"
        return decision, curvature_ratio

    def _update_lane_state(self, current_frame_decision):
        """
        Hysteresis 로직을 적용하여 최종 차선 상태를 업데이트합니다.
        """
        if self.lane_state == "Straight":
            if current_frame_decision == "Curve":
                self.lane_state = "Curve"
                self.straight_frames_counter = 0
        
        elif self.lane_state == "Curve":
            if current_frame_decision == "Straight":
                self.straight_frames_counter += 1
                if self.straight_frames_counter >= self.params['straight_thresh_frame']:
                    self.lane_state = "Straight"
            else: 
                self.straight_frames_counter = 0


    def _extract_representative_points(self, frame, detections, params):
        all_rep_points = []
        lines_to_draw = []
        h, w, _ = frame.shape
        debug_gray, debug_canny, debug_hough = None, None, None
        if self.debug:
            debug_gray = np.zeros((h, w), dtype=np.uint8)
            debug_canny = np.zeros((h, w), dtype=np.uint8)
            debug_hough = np.zeros((h, w, 3), dtype=np.uint8)

        if not detections:
            return all_rep_points, lines_to_draw, [debug_gray, debug_canny, debug_hough]

        for det in detections:
            x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            if y1 >= y2 or x1 >= x2: continue
            roi = frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, params['adapt_method'], cv2.THRESH_BINARY, params['block_size'], params['c_constant'])
            edges = cv2.Canny(thresh_roi, params['canny_thresh1'], params['canny_thresh2'], apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=params['hough_threshold'], minLineLength=params['hough_min_len'], maxLineGap=params['hough_max_gap'])

            if self.debug:
                debug_gray[y1:y2, x1:x2] = thresh_roi
                debug_canny[y1:y2, x1:x2] = edges
                hough_roi_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                if lines is not None:
                    for line in lines:
                        lx1, ly1, lx2, ly2 = line[0]
                        cv2.line(hough_roi_vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                debug_hough[y1:y2, x1:x2] = hough_roi_vis

            found_hough_points = False
            if lines is not None:
                # ======================= [수정된 부분 시작] =========================
                # 가장 긴 직선 대신, 가장 오른쪽에 있는 직선을 대표선으로 선택
                rightmost_line, max_mid_x = None, -1
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    # 직선의 x축 중심점 계산
                    mid_x = (lx1 + lx2) / 2
                    if mid_x > max_mid_x:
                        max_mid_x = mid_x
                        # BBox 내 상대 좌표를 이미지 전체 좌표로 변환하여 저장
                        rightmost_line = (lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1)
                
                if rightmost_line:
                    intersections = self._get_line_bbox_intersections(rightmost_line, (x1, y1, x2, y2))
                    if len(intersections) == 2:
                        all_rep_points.extend(intersections)
                        lines_to_draw.append(intersections)
                        found_hough_points = True
                # ======================= [수정된 부분 끝] ===========================
                        
            if not found_hough_points:
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                all_rep_points.append((center_x, center_y))
        return all_rep_points, lines_to_draw, [debug_gray, debug_canny, debug_hough]

    def _sort_representative_points(self, points, params):
        if len(points) < 2: return points
        remaining_points = points.copy()
        start_point = max(remaining_points, key=lambda p: p[1])
        sorted_points = [start_point]
        remaining_points.remove(start_point)
        current_point = start_point
        while remaining_points:
            distances = [(p, math.dist(current_point, p)) for p in remaining_points]
            if not distances: break
            min_distance = min(d for p, d in distances)
            candidates = [p for p, d in distances if d <= min_distance + params['similarity_thresh']]
            best_next_point = max(candidates, key=lambda p: p[1])
            sorted_points.append(best_next_point)
            remaining_points.remove(best_next_point)
            remaining_points = [p for p in remaining_points if math.dist(p, best_next_point) > 2]
            current_point = best_next_point
        return sorted_points
    
    def _get_line_bbox_intersections(self, line_coords, bbox_coords):
        lx1, ly1, lx2, ly2 = line_coords
        xmin, ymin, xmax, ymax = bbox_coords
        if lx1 == lx2: return [(lx1, ymin), (lx1, ymax)]
        m = (ly2 - ly1) / (lx2 - lx1)
        c = ly1 - m * lx1
        points = []
        for y in [ymin, ymax]:
            if m != 0:
                x = (y - c) / m
                if xmin <= x <= xmax: points.append((x, y))
        for x in [xmin, xmax]:
            y = m * x + c
            if ymin <= y <= ymax: points.append((x, y))
        unique_points = sorted(list(set(tuple(map(int, p)) for p in points)))
        return unique_points if len(unique_points) >= 2 else []

    def _extend_path_to_boundaries(self, path_points, frame_height, frame_width):
        if len(path_points) < 2: return path_points
        result_points = list(path_points)
        p_start, p_next = path_points[0], path_points[1]
        if p_next[0] != p_start[0]:
            m = (p_next[1] - p_start[1]) / (p_next[0] - p_start[0])
            if m != 0:
                c = p_start[1] - m * p_start[0]
                y_new, x_new = frame_height - 1, (frame_height - 1 - c) / m
                result_points.insert(0, (x_new, y_new))
        p_end, p_prev = path_points[-1], path_points[-2]
        intersections, dx, dy = [], p_end[0] - p_prev[0], p_end[1] - p_prev[1]
        if dy != 0:
            m = dy / dx if dx != 0 else float('inf')
            c = p_end[1] - m * p_end[0] if dx != 0 else 0
            if dy < 0:
                x_top = (0 - c) / m if dx != 0 else p_end[0]
                if 0 <= x_top < frame_width: intersections.append((x_top, 0))
            if dy > 0:
                y_bot = frame_height - 1
                x_bot = (y_bot - c) / m if dx != 0 else p_end[0]
                if 0 <= x_bot < frame_width: intersections.append((x_bot, y_bot))
        if dx != 0:
            m, c = dy / dx, p_end[1] - (dy / dx) * p_end[0]
            if dx < 0 and 0 <= c < frame_height: intersections.append((0, c))
            if dx > 0:
                y_right = m * (frame_width - 1) + c
                if 0 <= y_right < frame_height: intersections.append((frame_width - 1, y_right))
        if intersections:
            closest_point = min(intersections, key=lambda p: math.dist(p, p_end))
            result_points.append(closest_point)
        return result_points

    def fit_curve_to_points(self, points, mode='spline', frame_height=None, frame_width=None):
        if len(points) < 2: return []
        base_path = []
        if mode == 'spline':
            try:
                points_np = np.array(points, dtype=np.float32)
                x_coords, y_coords = points_np[:, 0], points_np[:, 1]
                num_points = len(points)
                if num_points >= 3:
                    k = 3 if num_points >= 4 else 2
                    tck, u = splprep([x_coords, y_coords], s=30.0, k=k)
                    eval_points = np.linspace(u.min(), u.max(), 100)
                    x_new, y_new = splev(eval_points, tck)
                    base_path = list(zip(x_new, y_new))
                else: base_path = points
            except (ValueError, np.linalg.LinAlgError) as e:
                # self.get_logger().warn(f"Spline curve fitting failed: {e}. Falling back to linear.")
                base_path = points
        elif mode == 'linear': base_path = points
        else:
            self.get_logger().warn(f"Invalid fit mode '{mode}'. Defaulting to linear.")
            base_path = points
        if not base_path: return []
        if frame_height is not None and frame_width is not None:
            return self._extend_path_to_boundaries(base_path, frame_height, frame_width)
        else: return base_path

    def visualize_results(self, frame, detections, sorted_points, curve_points, lines_to_draw, params, final_decision_text):
        output_frame = frame.copy()
        
        if final_decision_text:
            cv2.putText(output_frame, final_decision_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
        if params['draw_bbox']:
            for det in detections:
                x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if params['draw_hough']:
            for line in lines_to_draw:
                pt1, pt2 = tuple(map(int, line[0])), tuple(map(int, line[1]))
                cv2.line(output_frame, pt1, pt2, (255, 0, 0), 2)
        for i, point in enumerate(sorted_points):
            x, y = map(int, point)
            if params['draw_points']:
                cv2.circle(output_frame, (x, y), 7, (0, 0, 255), -1)
            if params['draw_numbers']:
                cv2.putText(output_frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        if params['draw_curve'] and curve_points:
            curve_pts = np.array(curve_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(output_frame, [curve_pts], isClosed=False, color=(255, 0, 255), thickness=2)
        cv2.imshow(self.main_window_name, output_frame)
        cv2.waitKey(1)

    def destroy_node(self):
        if self.debug:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CenterlaneTracer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt detected, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()