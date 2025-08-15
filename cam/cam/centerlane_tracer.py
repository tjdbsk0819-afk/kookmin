#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from cam_interfaces.msg import Detections
import message_filters
import math

class CenterlaneTracer(Node):
    """
    대표점들을 이용해 조각별 3차 베지어 곡선(Piecewise Cubic Bezier)을 생성하여
    매끄러운 차선을 추정하고 시각화합니다.
    """
    def __init__(self):
        super().__init__('centerlane_tracer')

        # --- Configuration ---
        self.HEIGHT_THRESHOLD = 220
        self.debug = True
        # 트랙바 초기값 설정
        self.initial_block_size_track = 8
        self.initial_c_constant_track = 16
        self.initial_num_points_track = 4
        self.initial_min_spacing_track = 14
        self.initial_bezier_scale_track = 8 # <<-- [추가] 베지어 스케일
        # ---------------------

        self.bridge = CvBridge()
        self.adaptive_block_size = 19
        self.adaptive_c = -9
        self.num_points_to_extract = 5
        self.min_point_spacing = 14
        self.bezier_scale_factor = 8.0 # <<-- [추가]

        self.image_subscription = message_filters.Subscriber(self, Image, '/image_raw')
        self.yolo_subscription = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.yolo_subscription],
            queue_size=10, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        
        if self.debug:
            self.main_window_name = 'Centerlane Tracer'
            self.thresh_window_name = 'Thresholded Image'
            cv2.namedWindow(self.main_window_name, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(self.thresh_window_name, cv2.WINDOW_AUTOSIZE)

            cv2.createTrackbar('Block Size', self.thresh_window_name, self.initial_block_size_track, 25, self.on_params_change)
            cv2.createTrackbar('C Constant', self.thresh_window_name, self.initial_c_constant_track, 50, self.on_params_change)
            cv2.createTrackbar('Num Points', self.thresh_window_name, self.initial_num_points_track, 19, self.on_params_change)
            cv2.createTrackbar('Min Spacing', self.thresh_window_name, self.initial_min_spacing_track, 50, self.on_params_change)
            cv2.createTrackbar('Bezier Scale', self.thresh_window_name, self.initial_bezier_scale_track, 200, self.on_params_change) # <<-- [추가]
            self.on_params_change(0)
            self.get_logger().info("Debug mode ON. Using Piecewise Bezier Curve Model.")

        self.get_logger().info('CenterlaneTracer has been started.')
    
    def on_params_change(self, _):
        block_size_track = cv2.getTrackbarPos('Block Size', self.thresh_window_name)
        self.adaptive_block_size = 2 * block_size_track + 3
        c_constant_track = cv2.getTrackbarPos('C Constant', self.thresh_window_name)
        self.adaptive_c = c_constant_track - 25
        num_points_track = cv2.getTrackbarPos('Num Points', self.thresh_window_name)
        self.num_points_to_extract = num_points_track + 1
        self.min_point_spacing = cv2.getTrackbarPos('Min Spacing', self.thresh_window_name)
        self.bezier_scale_factor = cv2.getTrackbarPos('Bezier Scale', self.thresh_window_name) # <<-- [추가]

    def draw_points(self, image, points, radius=3, color=(0, 0, 255)):
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)

    def draw_curve(self, image, curve_points, color=(255, 0, 255), thickness=2):
        if curve_points is None or len(curve_points) == 0:
            return
        pts = np.array(curve_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)

    # <<-- [추가] 베지어 곡선 관련 헬퍼 함수 ---
    def get_tangents(self, points):
        """각 점에서 부드럽게 이어지는 접선 벡터들을 계산합니다."""
        tangents = []
        n = len(points)
        for i in range(n):
            if i == 0: # 시작점
                tangent = (points[i+1] - points[i])
            elif i == n - 1: # 끝점
                tangent = (points[i] - points[i-1])
            else: # 중간점
                tangent = (points[i+1] - points[i-1])
            
            # 벡터 정규화
            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangents.append(tangent / norm)
            else:
                tangents.append(np.array([0.0, 0.0]))
        return tangents

    def get_bezier_segment(self, p0, p1, t0, t1, scale_factor):
        """두 점과 양 끝의 접선 벡터로 3차 베지어 곡선 점들을 계산합니다."""
        # 제어점 계산
        cp1 = p0 + t0 * scale_factor
        cp2 = p1 - t1 * scale_factor
        
        points = []
        for t in np.linspace(0, 1, 25): # 한 세그먼트당 25개 점 생성
            pt = (1-t)**3 * p0 + 3*(1-t)**2 * t * cp1 + 3*(1-t) * t**2 * cp2 + t**3 * p1
            points.append(pt)
        return points
    # ---------------------------------------------

    def synchronized_callback(self, image_msg, yolo_msg):
        if not self.debug: return
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.adaptiveThreshold(
            gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, self.adaptive_block_size, self.adaptive_c)

        filtered_detections = [det for det in yolo_msg.detections if det.class_name == 'center_line' and det.ymax > self.HEIGHT_THRESHOLD]
        
        all_rep_points = []
        # ... (대표점 추출 및 필터링 로직은 이전과 동일)
        for det in filtered_detections:
            x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
            box_w, box_h = x2 - x1, y2 - y1
            roi = thresh_frame[y1:y2, x1:x2]
            points_in_box = []
            if cv2.countNonZero(roi) > 0:
                num_pts = self.num_points_to_extract
                if box_w >= box_h:
                    for i in range(1, num_pts + 1):
                        x_local = int(box_w * i / (num_pts + 1))
                        ys_in_col = np.where(roi[:, x_local] == 255)[0]
                        if len(ys_in_col) > 0:
                            points_in_box.append((x1 + x_local, y1 + np.mean(ys_in_col)))
                else:
                    for i in range(1, num_pts + 1):
                        y_local = int(box_h * i / (num_pts + 1))
                        xs_in_row = np.where(roi[y_local, :] == 255)[0]
                        if len(xs_in_row) > 0:
                            points_in_box.append((x1 + np.mean(xs_in_row), y1 + y_local))
            else:
                points_in_box.append(((x1 + x2) / 2, (y1 + y2) / 2))

            if len(points_in_box) > 1:
                points_in_box.sort(key=lambda p: p[1])
                filtered_points = [points_in_box[0]]
                last_kept_point = points_in_box[0]
                for i in range(1, len(points_in_box)):
                    current_point = points_in_box[i]
                    dist = math.sqrt((current_point[0] - last_kept_point[0])**2 + (current_point[1] - last_kept_point[1])**2)
                    if dist > self.min_point_spacing:
                        filtered_points.append(current_point)
                        last_kept_point = current_point
                points_in_box = filtered_points
            all_rep_points.extend(points_in_box)

        # --- 베지어 곡선 피팅 로직 ---
        curve_points_to_draw = []
        if len(all_rep_points) >= 2:
            # y좌표 기준 오름차순 정렬 (위->아래)
            all_rep_points.sort(key=lambda p: p[1])
            points_np = np.array(all_rep_points)
            
            # 1. 각 점에서의 접선 벡터 계산
            tangents = self.get_tangents(points_np)
            
            # 2. 각 점 구간마다 베지어 곡선 생성하여 이어붙이기
            for i in range(len(points_np) - 1):
                p0 = points_np[i]
                p1 = points_np[i+1]
                t0 = tangents[i]
                t1 = tangents[i+1]
                segment_points = self.get_bezier_segment(p0, p1, t0, t1, self.bezier_scale_factor)
                curve_points_to_draw.extend(segment_points)
        # ---------------------------

        # 시각화
        output_frame = frame.copy()
        masked_thresh_frame = np.zeros_like(thresh_frame)
        for det in filtered_detections:
            x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
            masked_thresh_frame[y1:y2, x1:x2] = thresh_frame[y1:y2, x1:x2]
        masked_thresh_frame_bgr = cv2.cvtColor(masked_thresh_frame, cv2.COLOR_GRAY2BGR)
        
        self.draw_curve(output_frame, curve_points_to_draw)
        self.draw_curve(masked_thresh_frame_bgr, curve_points_to_draw)
        self.draw_points(output_frame, all_rep_points)
        self.draw_points(masked_thresh_frame_bgr, all_rep_points, color=(0, 0, 255))

        for det in filtered_detections:
            x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0, 50), 1)
        
        param_text = f"B:{self.adaptive_block_size}, C:{self.adaptive_c}, Pts:{self.num_points_to_extract}, Spacing:{self.min_point_spacing}, BezierScale:{self.bezier_scale_factor}"
        cv2.putText(output_frame, param_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(output_frame, param_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(self.main_window_name, output_frame)
        cv2.imshow(self.thresh_window_name, masked_thresh_frame_bgr)
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