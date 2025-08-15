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

from scipy.interpolate import splprep, splev

class CenterlaneTracer(Node):
    """
    대표점들에 대해 적응형 곡선 피팅을 수행하고,
    실시간 파라미터 조정을 위한 트랙바와 디버그 창을 제공합니다.
    """
    def __init__(self):
        super().__init__('centerlane_tracer')

        self.HEIGHT_THRESHOLD = 220
        self.debug = True
        self.bridge = CvBridge()
        self.current_frame = None

        self.image_subscription = message_filters.Subscriber(self, Image, '/image_raw')
        self.yolo_subscription = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.yolo_subscription],
            queue_size=10, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        if self.debug:
            self.main_window_name = 'Centerlane Tracer'
            self.debug_window_names = ['Grayscale (AdaptiveThresh)', 'Canny Edges', 'Hough Lines']
            cv2.namedWindow(self.main_window_name, cv2.WINDOW_AUTOSIZE)
            for name in self.debug_window_names:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)

            cv2.setMouseCallback(self.main_window_name, self.mouse_event_callback)

            def nothing(x): pass
            cv2.createTrackbar('Adapt Method', self.main_window_name, 1, 1, nothing)
            cv2.createTrackbar('Block Size', self.main_window_name, 5, 20, nothing)
            cv2.createTrackbar('C Constant', self.main_window_name, 5, 50, nothing)
            cv2.createTrackbar('Canny Thresh1', self.main_window_name, 50, 255, nothing)
            cv2.createTrackbar('Canny Thresh2', self.main_window_name, 150, 255, nothing)
            cv2.createTrackbar('Hough Threshold', self.main_window_name, 20, 100, nothing)
            cv2.createTrackbar('Hough MinLineLength', self.main_window_name, 20, 100, nothing)
            cv2.createTrackbar('Hough MaxLineGap', self.main_window_name, 5, 100, nothing)
            # [수정] 거리 유사성 임계값 트랙바 추가 (0~50 픽셀)
            cv2.createTrackbar('Similarity Thresh', self.main_window_name, 25, 50, nothing)

            self.get_logger().info("Debug mode ON. Using Adaptive Curve Fitting Model.")

        self.get_logger().info('CenterlaneTracer has been started.')

    def mouse_event_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_frame is not None:
            bgr_pixel = self.current_frame[y, x]
            hsv_pixel = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
            self.get_logger().info(f'Click at ({x}, {y}) | BGR: {bgr_pixel} -> HSV: {hsv_pixel}')

    def synchronized_callback(self, image_msg, yolo_msg):
        if not self.debug: return
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            self.current_frame = frame.copy()
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        adapt_method_val = cv2.getTrackbarPos('Adapt Method', self.main_window_name)
        block_size_val = cv2.getTrackbarPos('Block Size', self.main_window_name)
        c_constant_val = cv2.getTrackbarPos('C Constant', self.main_window_name)

        params = {
            'adapt_method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adapt_method_val == 1 else cv2.ADAPTIVE_THRESH_MEAN_C,
            'block_size': block_size_val * 2 + 3,
            'c_constant': c_constant_val - 25,
            'canny_thresh1': cv2.getTrackbarPos('Canny Thresh1', self.main_window_name),
            'canny_thresh2': cv2.getTrackbarPos('Canny Thresh2', self.main_window_name),
            'hough_threshold': cv2.getTrackbarPos('Hough Threshold', self.main_window_name),
            'hough_min_len': cv2.getTrackbarPos('Hough MinLineLength', self.main_window_name),
            'hough_max_gap': cv2.getTrackbarPos('Hough MaxLineGap', self.main_window_name),
            # [수정] 트랙바에서 거리 유사성 임계값 읽기
            'similarity_thresh': cv2.getTrackbarPos('Similarity Thresh', self.main_window_name),
        }

        filtered_detections = [
            det for det in yolo_msg.detections
            if det.class_name == 'center_line' and det.ymax > self.HEIGHT_THRESHOLD
        ]

        all_rep_points_unsorted, lines_to_draw, debug_images = self._extract_representative_points(frame, filtered_detections, params)

        # [수정] 정렬 함수에 params 전달
        sorted_rep_points = self._sort_representative_points(all_rep_points_unsorted, params)
        curve_points_to_draw = self.fit_adaptive_curve(sorted_rep_points)
        self.visualize_results(frame, filtered_detections, sorted_rep_points, curve_points_to_draw, lines_to_draw)

        for name, img in zip(self.debug_window_names, debug_images):
            if img is not None:
                cv2.imshow(name, img)

    def _extract_representative_points(self, frame, detections, params):
        all_rep_points = []
        lines_to_draw = []
        h, w, _ = frame.shape
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

            roi = self.current_frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            thresh_roi = cv2.adaptiveThreshold(gray_roi, 255,
                                               params['adapt_method'],
                                               cv2.THRESH_BINARY,
                                               params['block_size'],
                                               params['c_constant'])
            edges = cv2.Canny(thresh_roi, params['canny_thresh1'], params['canny_thresh2'], apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=params['hough_threshold'], minLineLength=params['hough_min_len'], maxLineGap=params['hough_max_gap'])

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
                longest_line, max_len_sq = None, 0
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    current_len_sq = (lx2 - lx1)**2 + (ly2 - ly1)**2
                    if current_len_sq > max_len_sq:
                        max_len_sq = current_len_sq
                        longest_line = (lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1)

                if longest_line:
                    intersections = self._get_line_bbox_intersections(longest_line, (x1, y1, x2, y2))
                    if len(intersections) == 2:
                        all_rep_points.extend(intersections)
                        lines_to_draw.append(intersections)
                        found_hough_points = True

            if not found_hough_points:
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                all_rep_points.append((center_x, center_y))
                self.get_logger().warn(f"Hough failed for box at ({x1},{y1}). Using center point as fallback.")

        return all_rep_points, lines_to_draw, [debug_gray, debug_canny, debug_hough]

    # [수정] params 딕셔너리를 인자로 받도록 변경
    def _sort_representative_points(self, points, params):
        if len(points) < 2:
            return points

        remaining_points = points.copy()
        
        start_point = max(remaining_points, key=lambda p: p[1])
        
        sorted_points = [start_point]
        remaining_points.remove(start_point)
        current_point = start_point

        while remaining_points:
            distances = [(p, math.dist(current_point, p)) for p in remaining_points]
            if not distances:
                break
            
            min_distance = min(d for p, d in distances)
            
            # [수정] self 변수 대신 params 에서 읽어온 임계값 사용
            candidates = [p for p, d in distances if d <= min_distance + params['similarity_thresh']]
            
            best_next_point = max(candidates, key=lambda p: p[1])
            
            sorted_points.append(best_next_point)
            remaining_points.remove(best_next_point)
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

    def fit_adaptive_curve(self, points):
        if len(points) < 2: return []
        
        try:
            points_np = np.array(points, dtype=np.float32)
            x_coords = points_np[:, 0]
            y_coords = points_np[:, 1]
            
            num_points = len(points)
            if num_points >= 3:
                k = 3 if num_points >= 4 else 2
                tck, u = splprep([x_coords, y_coords], s=30.0, k=k)
                eval_points = np.linspace(u.min(), u.max(), 100)
                x_new, y_new = splev(eval_points, tck)
                return list(zip(x_new, y_new))
            elif num_points == 2:
                return points

        except (ValueError, np.linalg.LinAlgError) as e:
            self.get_logger().warn(f"Curve fitting failed: {e}")
        return []

    def visualize_results(self, frame, detections, sorted_points, curve_points, lines_to_draw):
        output_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        for line in lines_to_draw:
            pt1 = tuple(map(int, line[0]))
            pt2 = tuple(map(int, line[1]))
            cv2.line(output_frame, pt1, pt2, (255, 0, 0), 2)

        for i, point in enumerate(sorted_points):
            x, y = map(int, point)
            cv2.circle(output_frame, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(output_frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        if curve_points:
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