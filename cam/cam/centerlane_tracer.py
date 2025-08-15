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
            # 파라미터 트랙바
            cv2.createTrackbar('Adapt Method', self.main_window_name, 1, 1, nothing)
            cv2.createTrackbar('Block Size', self.main_window_name, 5, 20, nothing)
            cv2.createTrackbar('C Constant', self.main_window_name, 5, 50, nothing)
            cv2.createTrackbar('Canny Thresh1', self.main_window_name, 50, 255, nothing)
            cv2.createTrackbar('Canny Thresh2', self.main_window_name, 150, 255, nothing)
            cv2.createTrackbar('Hough Threshold', self.main_window_name, 20, 100, nothing)
            cv2.createTrackbar('Hough MinLineLength', self.main_window_name, 20, 100, nothing)
            cv2.createTrackbar('Hough MaxLineGap', self.main_window_name, 5, 100, nothing)
            cv2.createTrackbar('Similarity Thresh', self.main_window_name, 25, 50, nothing)
            # [변경] Fit Mode의 기본값을 1로 설정
            cv2.createTrackbar('Fit Mode', self.main_window_name, 1, 1, nothing)
            
            # 그리기 옵션 토글을 위한 트랙바
            cv2.createTrackbar('Draw BBox', self.main_window_name, 1, 1, nothing)
            cv2.createTrackbar('Draw Points', self.main_window_name, 1, 1, nothing)
            cv2.createTrackbar('Draw Numbers', self.main_window_name, 1, 1, nothing)
            cv2.createTrackbar('Draw Hough', self.main_window_name, 1, 1, nothing)
            cv2.createTrackbar('Draw Curve', self.main_window_name, 1, 1, nothing)

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

        h, w, _ = frame.shape

        params = {
            'adapt_method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C if cv2.getTrackbarPos('Adapt Method', self.main_window_name) == 1 else cv2.ADAPTIVE_THRESH_MEAN_C,
            'block_size': cv2.getTrackbarPos('Block Size', self.main_window_name) * 2 + 3,
            'c_constant': cv2.getTrackbarPos('C Constant', self.main_window_name) - 25,
            'canny_thresh1': cv2.getTrackbarPos('Canny Thresh1', self.main_window_name),
            'canny_thresh2': cv2.getTrackbarPos('Canny Thresh2', self.main_window_name),
            'hough_threshold': cv2.getTrackbarPos('Hough Threshold', self.main_window_name),
            'hough_min_len': cv2.getTrackbarPos('Hough MinLineLength', self.main_window_name),
            'hough_max_gap': cv2.getTrackbarPos('Hough MaxLineGap', self.main_window_name),
            'similarity_thresh': cv2.getTrackbarPos('Similarity Thresh', self.main_window_name),
            'draw_bbox': cv2.getTrackbarPos('Draw BBox', self.main_window_name) == 1,
            'draw_points': cv2.getTrackbarPos('Draw Points', self.main_window_name) == 1,
            'draw_numbers': cv2.getTrackbarPos('Draw Numbers', self.main_window_name) == 1,
            'draw_hough': cv2.getTrackbarPos('Draw Hough', self.main_window_name) == 1,
            'draw_curve': cv2.getTrackbarPos('Draw Curve', self.main_window_name) == 1,
        }

        fit_mode_val = cv2.getTrackbarPos('Fit Mode', self.main_window_name)
        fit_mode = 'linear' if fit_mode_val == 1 else 'spline'

        filtered_detections = [
            det for det in yolo_msg.detections
            if det.class_name == 'center_line' and det.ymax > self.HEIGHT_THRESHOLD
        ]

        all_rep_points_unsorted, lines_to_draw, debug_images = self._extract_representative_points(frame, filtered_detections, params)
        sorted_rep_points = self._sort_representative_points(all_rep_points_unsorted, params)
        
        curve_points_to_draw = self.fit_curve_to_points(
            sorted_rep_points,
            mode=fit_mode, 
            frame_height=h,
            frame_width=w
        )
        
        self.visualize_results(frame, filtered_detections, sorted_rep_points, curve_points_to_draw, lines_to_draw, params)

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
                # [제거] fallback 사용 시 출력되던 경고 메시지 제거
                # self.get_logger().warn(f"Hough failed for box at ({x1},{y1}). Using center point as fallback.")

        return all_rep_points, lines_to_draw, [debug_gray, debug_canny, debug_hough]

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
        if len(path_points) < 2:
            return path_points

        result_points = list(path_points)
        
        p_start = path_points[0]
        p_next = path_points[1]
        if p_next[0] != p_start[0]:
            m = (p_next[1] - p_start[1]) / (p_next[0] - p_start[0])
            if m != 0:
                c = p_start[1] - m * p_start[0]
                y_new = frame_height - 1
                x_new = (y_new - c) / m
                result_points.insert(0, (x_new, y_new))

        p_end = path_points[-1]
        p_prev = path_points[-2]
        
        intersections = []
        
        dx = p_end[0] - p_prev[0]
        dy = p_end[1] - p_prev[1]

        if dy != 0:
            m = dy / dx if dx != 0 else float('inf')
            c = p_end[1] - m * p_end[0] if dx != 0 else 0

            if dy < 0:
                x_top = (0 - c) / m if dx != 0 else p_end[0]
                if 0 <= x_top < frame_width:
                    intersections.append((x_top, 0))
            if dy > 0:
                y_bot = frame_height - 1
                x_bot = (y_bot - c) / m if dx != 0 else p_end[0]
                if 0 <= x_bot < frame_width:
                    intersections.append((x_bot, y_bot))
        
        if dx != 0:
            m = dy / dx
            c = p_end[1] - m * p_end[0]
            
            if dx < 0:
                y_left = c
                if 0 <= y_left < frame_height:
                    intersections.append((0, y_left))
            if dx > 0:
                y_right = m * (frame_width - 1) + c
                if 0 <= y_right < frame_height:
                    intersections.append((frame_width - 1, y_right))
        
        if intersections:
            closest_point = min(intersections, key=lambda p: math.dist(p, p_end))
            result_points.append(closest_point)
        
        return result_points

    def fit_curve_to_points(self, points, mode='spline', frame_height=None, frame_width=None):
        if len(points) < 2:
            return []

        base_path = []
        if mode == 'spline':
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
                    base_path = list(zip(x_new, y_new))
                else:
                    base_path = points
            except (ValueError, np.linalg.LinAlgError) as e:
                self.get_logger().warn(f"Spline curve fitting failed: {e}. Falling back to linear.")
                base_path = points
        
        elif mode == 'linear':
            base_path = points
        
        else:
            self.get_logger().warn(f"Invalid fit mode '{mode}'. Defaulting to linear.")
            base_path = points

        if not base_path:
            return []

        if frame_height is not None and frame_width is not None:
            return self._extend_path_to_boundaries(base_path, frame_height, frame_width)
        else:
            return base_path

    def visualize_results(self, frame, detections, sorted_points, curve_points, lines_to_draw, params):
        output_frame = frame.copy()
        
        if params['draw_bbox']:
            for det in detections:
                x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        if params['draw_hough']:
            for line in lines_to_draw:
                pt1 = tuple(map(int, line[0]))
                pt2 = tuple(map(int, line[1]))
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