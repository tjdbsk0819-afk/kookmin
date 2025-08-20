#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray
from collections import deque
from custom_interfaces.msg import ObstacleState, XycarState, Curve
from std_srvs.srv import Trigger

### 추가된 부분 ###
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

class IntegratedStanleyController(Node):
    def __init__(self):
        super().__init__('integrated_stanley_controller')

        # ==============================================================
        #              ### 추가된 부분 ### : 콜백 그룹 생성
        # ==============================================================
        # ReentrantCallbackGroup: 그룹 내의 콜백들이 서로를 차단하지 않고 병렬 실행되도록 허용
        self.control_callback_group = ReentrantCallbackGroup()
        self.state_update_callback_group = ReentrantCallbackGroup()

        # ==============================================================
        #                       ROS2 인터페이스
        # ==============================================================
        # 구독자들 (callback_group 인자 추가)
        self.xycar_state_sub = self.create_subscription(
            XycarState,
            '/xycar_state',
            self.xycar_state_callback,
            10,
            callback_group=self.control_callback_group  # 제어 그룹 할당
        )
        self.obstacle_state_sub = self.create_subscription(
            ObstacleState,
            '/obstacle_state',
            self.obstacle_state_callback,
            10,
            callback_group=self.state_update_callback_group  # 상태 업데이트 그룹 할당
        )
        self.curve_subscription = self.create_subscription(
            Curve, 
            '/center_curve',
            self.curve_callback,
            10,
            callback_group=self.state_update_callback_group  # 상태 업데이트 그룹 할당
        )

        # 퍼블리셔
        self.motor_pub = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.last_time = self.get_clock().now()

        # 서비스 서버
        self.is_active = False
        self.start_service = self.create_service(
            Trigger,
            'start_track_driving',
            self.start_service_callback,
            callback_group=self.state_update_callback_group  # 상태 업데이트 그룹 할당
        )

        self.get_logger().info('🎯 Integrated Stanley Controller Node Successfully Started❗')

        # ==============================================================
        #                   Stanley Controller 파라미터
        # ==============================================================
        self.image_center_x = 320
        self.pixel_to_meter = 1.9 / 650.0
        self.heading_weight = 0.3
        self.deg_to_servo_scale = 5.0 / 3.0
        self.deg_to_servo_offset = 0.0
        self.max_delta_angle = 15.0
        self.window = 1
        self.steering_threshold = 3.0
        self.k_straight = 1.2
        self.k_curve = 1.2
        self.k_obstacle = 1.2
        self.speed_straight = 35.0
        self.speed_curve = 20.0

        # ==============================================================
        #                   Obstacle Follower 파라미터
        # ==============================================================
        self.TARGET_TIME_GAP = 5.0
        self.OBSTACLE_DETECTION_THRESHOLD = 2.0
        self.KP_SPEED_CONTROL = 0.5
        self.MIN_SPEED = 0.0
        self.ACCEL_STEP = 0.4
        self.DECEL_STEP = 1.0

        # ==============================================================
        #                       상태 변수들
        # ==============================================================
        self.last_steering_deg = 0.0
        self.angle_buffer = deque(maxlen=self.window)
        self.current_k = self.k_straight
        self.current_mode = 'center'
        self.relative_distance = self.OBSTACLE_DETECTION_THRESHOLD + 1.0
        self.is_obstacle_detected = False
        self.current_speed = self.MIN_SPEED
        self.obstacle_position = 'none'
        self.detected_vehicle_count = 0
        self.curve_state = "Straight"
        self.min_allowed_speed = min(self.speed_curve, self.speed_straight)
        self.max_allowed_speed = max(self.speed_curve, self.speed_straight)

        # ### 추가된 부분 ### : 로깅 빈도 제어를 위한 변수
        self.log_counter = 0
        self.log_interval = 1  # 10번의 콜백 당 1번만 로그 출력
    
    def start_service_callback(self, request, response):
        if not self.is_active:
            self.is_active = True
            self.get_logger().info('✅ Service called. Starting Integrated Stanley Controller main logic!')
            response.success = True
            response.message = 'Integrated Stanley Controller started.'
        else:
            self.get_logger().warn('⚠️ Main logic is already running.')
            response.success = False
            response.message = 'Already active.'
        return response

    # ==============================================================
    #                   Stanley Controller 메서드들
    # ==============================================================
    @staticmethod
    def compute_steering_angle_stanley(lane_center_x, lane_center_y, lane_angle_rad,
                                       image_center_x, pixel_to_meter, heading_weight, 
                                       deg_to_servo_scale, deg_to_servo_offset, k):
        cte_pixel = lane_center_x - image_center_x
        cte = cte_pixel * pixel_to_meter
        heading_error = lane_angle_rad
        cte_term = np.arctan2(k * cte, 3.0)
        steering_angle_rad = heading_weight * heading_error + cte_term
        servo_angle = np.degrees(steering_angle_rad) * deg_to_servo_scale + deg_to_servo_offset
        return servo_angle, heading_error, cte_term

    @staticmethod
    def clamp_angle(angle):
        return max(min(angle, 100.0), -100.0)

    def limit_angle_change(self, current, last, max_delta_angle):
        delta = current - last
        delta = max(-max_delta_angle, min(delta, max_delta_angle))
        return last + delta

    def smooth_angle(self, new_angle):
        self.angle_buffer.append(new_angle)
        return sum(self.angle_buffer) / len(self.angle_buffer)

    def get_target_speed_and_k(self, servo_angle):
        if self.is_obstacle_detected:
            self.current_k = self.k_obstacle
            if abs(servo_angle) > self.steering_threshold:
                target_speed = self.speed_curve
            else:
                target_speed = self.speed_straight
        else:
            if abs(servo_angle) > self.steering_threshold:
                target_speed = self.speed_curve
                self.current_k = self.k_curve
            else:
                target_speed = self.speed_straight
                self.current_k = self.k_straight
        return target_speed, self.current_k

    # ==============================================================
    #                   Obstacle Follower 메서드들
    # ==============================================================
    def calculate_final_speed(self, target_speed):
            is_in_curve = self.curve_state.lower() == 'curve'
            condition1 = is_in_curve and self.is_obstacle_detected 
            condition2 = self.detected_vehicle_count >= 2
            condition_to_follow = condition1 or condition2

            if condition_to_follow:
                current_speed_mps = self.current_speed / 20.0 
                safe_speed_mps = max(current_speed_mps, 0.1)
                time_gap = self.relative_distance / safe_speed_mps
                error = time_gap - self.TARGET_TIME_GAP
                speed_adjustment = self.KP_SPEED_CONTROL * error
                adjusted_speed = self.current_speed + speed_adjustment
                final_speed = np.clip(adjusted_speed, self.MIN_SPEED, target_speed)
                return final_speed, f"🚧 FOLLOWING (V:{self.detected_vehicle_count})"
            else:
                speed_diff = target_speed - self.current_speed
                if speed_diff > 0:
                    return self.current_speed + self.ACCEL_STEP, "🔼 ACCELERATING"
                elif speed_diff < 0:
                    return self.current_speed - self.DECEL_STEP, "🔽 DECELERATING"
                else:
                    return self.current_speed, "➡️ MAINTAINING"

    # ==============================================================
    #                           콜백 함수들
    # ==============================================================
    def curve_callback(self, msg: Curve):
        self.curve_state = msg.state

    def obstacle_state_callback(self, msg: ObstacleState):
        self.relative_distance = msg.distance_m
        self.obstacle_position = msg.position.strip().lower()
        self.detected_vehicle_count = msg.vehicle_count
        self.is_obstacle_detected = self.relative_distance < self.OBSTACLE_DETECTION_THRESHOLD

    def xycar_state_callback(self, msg: XycarState):
        if not self.is_active:
            return

        self.current_mode = msg.drive_mode.strip().lower()
        lane_center_x = msg.target_point.x
        lane_center_y = msg.target_point.y
        lane_angle_rad = msg.target_point.z

        servo_angle, heading_error_rad, cte_term = self.compute_steering_angle_stanley(
            lane_center_x=lane_center_x,
            lane_center_y=lane_center_y,
            lane_angle_rad=lane_angle_rad,
            image_center_x=self.image_center_x,
            pixel_to_meter=self.pixel_to_meter,
            heading_weight=self.heading_weight,
            deg_to_servo_scale=self.deg_to_servo_scale,
            deg_to_servo_offset=self.deg_to_servo_offset,
            k=self.current_k
        )

        raw_delta = servo_angle - self.last_steering_deg
        rate_limited = abs(raw_delta) > self.max_delta_angle
        limited_angle = self.limit_angle_change(servo_angle, self.last_steering_deg, self.max_delta_angle)
        smoothed_angle = self.smooth_angle(limited_angle)
        final_angle = self.clamp_angle(smoothed_angle)
        self.last_steering_deg = final_angle

        target_speed, current_k = self.get_target_speed_and_k(smoothed_angle)

        final_speed, speed_status = self.calculate_final_speed(target_speed)
        self.current_speed = final_speed

        motor_msg = Float32MultiArray()
        motor_msg.data = [final_angle, float(self.current_speed)]
        self.motor_pub.publish(motor_msg)

        # ### 변경된 부분 ### : 로깅을 주기적으로 실행
        self.log_counter += 1
        if self.log_counter % self.log_interval == 0:
            self.log_status(heading_error_rad, cte_term, current_k, final_angle, 
                           target_speed, final_speed, speed_status, rate_limited)
        
    def log_status(self, heading_error_rad, cte_term, current_k, final_angle, 
                   target_speed, final_speed, speed_status, rate_limited):
        if self.current_mode == 'center':
            lane_target_str = "🟡 CENTER"
        elif self.current_mode == 'left':
            lane_target_str = "🔵 LEFT"
        elif self.current_mode == 'right':
            lane_target_str = "🔴 RIGHT"
        else:
            lane_target_str = "⚪ UNKNOWN"

        if self.is_obstacle_detected:
            driving_mode = "🚧 OBSTACLE MODE"
        else:
            driving_mode = "🔄 CURVE MODE" if abs(final_angle) > self.steering_threshold else "➡️ STRAIGHT MODE"

        rate_limit_marker = "⛔" if rate_limited else ""
        obstacle_status = f"🚗 Dist: {self.relative_distance:.2f}m" if self.is_obstacle_detected else "🟢 Clear"

        self.get_logger().info(
            "\n📐 [Integrated Stanley Controller Log] ---------------\n"
            f"  • Lane Target       : {lane_target_str}\n"
            f"  • Driving Mode      : {driving_mode}\n"
            f"  • Obstacle Status   : {obstacle_status}\n"
            f"  • Stanley K Value   : {current_k:>6.2f}\n"
            f"  • Heading Error     : {np.degrees(heading_error_rad):>6.2f}°\n"
            f"  • Cross Track Term  : {np.degrees(cte_term):>6.2f}°\n"
            f"  • Servo Angle       : {final_angle:>6.2f} {rate_limit_marker}\n"
            f"  • Target Speed      : {target_speed:>6.2f}\n"
            f"  • Final Speed       : {final_speed:>6.2f} {speed_status}\n"
            "--------------------------------------------------------"
        )

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedStanleyController()
    
    ### 변경된 부분 ###
    # 멀티스레드 실행기 생성 (CPU 코어 수를 고려하여 스레드 수 조절 가능)
    # num_threads를 지정하지 않으면 CPU 코어 수만큼 스레드를 생성합니다.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        # executor.spin()을 사용하여 노드를 실행
        executor.spin()
    finally:
        # 노드와 실행기를 깔끔하게 종료
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()