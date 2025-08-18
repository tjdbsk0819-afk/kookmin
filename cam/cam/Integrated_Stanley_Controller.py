#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray, String
from collections import deque
from custom_interfaces.msg import ObstacleState, XycarState, Curve  # Curve 구독

class IntegratedStanleyController(Node):
    def __init__(self):
        super().__init__('integrated_stanley_controller')

        # ==============================================================
        #                       ROS2 인터페이스
        # ==============================================================
        # 구독자들
        self.xycar_state_sub = self.create_subscription(XycarState, '/xycar_state', self.xycar_state_callback, 10)
        self.obstacle_state_sub = self.create_subscription(ObstacleState, '/obstacle_state', self.obstacle_state_callback, 10)
        self.curve_state_sub = self.create_subscription(Curve, '/center_curve', self.curve_callback, 10)  # 커브/직선 상태

        # 퍼블리셔
        self.motor_pub = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.override_pub = self.create_publisher(String, '/lane_override_cmd', 10)

        self.last_time = self.get_clock().now()
        self.get_logger().info('🎯 Integrated Stanley Controller Node Successfully Started❗')

        # ==============================================================
        #                   Stanley Controller 파라미터
        # ==============================================================
        # 1) Stanley 계산용
        self.image_center_x = 320                   # 이미지 중앙 x좌표
        self.pixel_to_meter = 1.9 / 650.0           # 픽셀 → 미터 환산계수
        self.heading_weight = 0.4                   # 헤딩오차 반영 비율
        self.deg_to_servo_scale = 5.0 / 3.0
        self.deg_to_servo_offset = 0.0

        # 2) 조향각 후처리
        self.max_delta_angle = 10.0
        self.window = 3

        # 3) 속도 & 게인
        self.steering_threshold = 3.0               # curve 판단 임계 조향각(안전 캡용)

        self.k_straight = 1.0
        self.k_curve = 1.2
        self.k_obstacle = 1.2

        self.speed_straight = 35.0
        self.speed_curve = 20.0

        # ==============================================================
        #                   Obstacle Follower 파라미터
        # ==============================================================
        self.TARGET_TIME_GAP = 4.0
        self.OBSTACLE_DETECTION_THRESHOLD = 2.0     # 감지 거리(m)
        self.KP_SPEED_CONTROL = 0.5                  # 제어 게인
        self.MIN_SPEED = 1.0
        # 부드러운 가감속
        self.ACCEL_STEP = 0.4
        self.DECEL_STEP = 0.7

        # ==============================================================
        #                       상태 변수들
        # ==============================================================
        # Stanley 상태
        self.last_steering_deg = 0.0
        self.angle_buffer = deque(maxlen=self.window)
        self.current_k = self.k_straight
        self.current_mode = 'center'                # 초기 타겟 포인트

        # 커브/직선 상태(외부 구독)
        self.lane_state_from_curve = 'straight'     # 'straight' | 'curve'

        # Obstacle Follower 상태
        self.relative_distance = self.OBSTACLE_DETECTION_THRESHOLD + 1.0
        self.is_obstacle_detected = False
        self.current_speed = self.MIN_SPEED
        self.obstacle_position = 'none'
        self.detected_vehicle_count = 0

        # 기타
        self.min_allowed_speed = min(self.speed_curve, self.speed_straight)
        self.max_allowed_speed = max(self.speed_curve, self.speed_straight)
        
        # 토픽 READY 상태 플래그
        self.topic_ready = {
            "xycar_state": False,
            "obstacle_state": False
        }

		# ==============================================================
    #                   필요한 토픽 전체 READY 확인
    # ==============================================================

    def wait_for_message(self):
        """
        xycar_state, obstacle_state, image_raw, yolo_detections 토픽이
        모두 수신될 때까지 대기하는 함수입니다.
        """
        self.get_logger().info('⏳ Waiting for required topics to become available...')
        
        while not all(self.topic_ready.values()) and rclpy.ok():
            # 어떤 토픽을 기다리고 있는지 1초에 한 번씩 로그로 출력
            waiting_for = [topic for topic, is_ready in self.topic_ready.items() if not is_ready]
            self.get_logger().info(f"  - Waiting for: {', '.join(waiting_for)}", throttle_duration_sec=1.0)
            
            # 콜백이 호출될 수 있도록 spin_once를 사용
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('✅ All topics are ready. Starting the integrated controller.')

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
        """
        (안전상한용) 조향각을 기준으로 목표 속도와 k를 계산.
        최종 k는 아래에서 상태 동기화로 덮어쓸 수 있음.
        """
        if self.is_obstacle_detected:
            self.current_k = self.k_obstacle
            target_speed = self.speed_curve if abs(servo_angle) > self.steering_threshold else self.speed_straight
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
        # 장애물 차량이 2대 이상일 때만 Follow 로직 실행
        condition_to_follow = (
            self.is_obstacle_detected and
            self.detected_vehicle_count >= 2
        )

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
            # 평상시 가감속 스무딩
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
        """CenterlaneTracer의 커브/직선 상태 구독"""
        state = (msg.state or '').strip().lower()
        if state in ('straight', 'curve'):
            self.lane_state_from_curve = state
        else:
            self.lane_state_from_curve = 'straight'  # 보수적 기본값

    def obstacle_state_callback(self, msg: ObstacleState):
        """장애물 상태(거리 및 위치) 콜백"""
        self.topic_ready['obstacle_state'] = True
        
        # 메시지에서 거리와 위치 정보를 한 번에 업데이트
        self.relative_distance = msg.distance_m
        self.obstacle_position = msg.position.strip().lower()
        self.detected_vehicle_count = msg.vehicle_count
        self.is_obstacle_detected = self.relative_distance < self.OBSTACLE_DETECTION_THRESHOLD

        # 회피 명령(차선 override) 발행
        if self.is_obstacle_detected and self.detected_vehicle_count == 1:
            if self.obstacle_position == 'right':
                self.get_logger().warn("🚨 Condition met! Sending 'go_left' command to Lane Detector.")
                cmd_msg = String()
                cmd_msg.data = 'go_left'
                self.override_pub.publish(cmd_msg)
            elif self.obstacle_position == 'left':
                self.get_logger().warn("🚨 Condition met! Sending 'go_right' command to Lane Detector.")
                cmd_msg = String()
                cmd_msg.data = 'go_right'
                self.override_pub.publish(cmd_msg)

    def xycar_state_callback(self, msg: XycarState):
        """차선/주행모드 콜백 - 메인 제어"""
        self.topic_ready['xycar_state'] = True
        
        # 모든 토픽이 준비되기 전에는 제어 로직을 실행하지 않음
        if not all(self.topic_ready.values()):
            return
            
        # 1) 주행 모드 업데이트
        self.current_mode = msg.drive_mode.strip().lower()

        lane_center_x = msg.target_point.x
        lane_center_y = msg.target_point.y
        lane_angle_rad = msg.target_point.z

        # 2) Stanley 조향각 계산 (현재 self.current_k 사용)
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

        # 3) 조향각 후처리(리미트/스무딩/클램프)
        raw_delta = servo_angle - self.last_steering_deg
        rate_limited = abs(raw_delta) > self.max_delta_angle
        limited_angle = self.limit_angle_change(servo_angle, self.last_steering_deg, self.max_delta_angle)
        smoothed_angle = self.smooth_angle(limited_angle)
        final_angle = self.clamp_angle(smoothed_angle)
        self.last_steering_deg = final_angle

        # 4) 목표 속도 결정
        # 4-1) (세이프티 캡) 조향각 기반 속도
        target_speed_servo, current_k_servo = self.get_target_speed_and_k(smoothed_angle)

        # 4-2) (주속도) 커브/직선 상태 기반 속도
        target_speed_state = self.speed_curve if self.lane_state_from_curve == 'curve' else self.speed_straight

        # 4-3) 최종 목표속도: 상태 기반이 우선, 조향각 기반은 안전 상한
        target_speed = min(target_speed_state, target_speed_servo)

        # 🔴 k 동기화: 장애물 없으면 커브/직선 상태로 k를 강제로 동기화
        # (다음 프레임의 Stanley 계산부터 반영됨. 필요하면 계산 전에 옮겨 선반영도 가능)
        if self.is_obstacle_detected:
            self.current_k = self.k_obstacle
        else:
            self.current_k = self.k_curve if self.lane_state_from_curve == 'curve' else self.k_straight

        # 5) 최종 적용 속도(장애물/스무딩 포함)
        final_speed, speed_status = self.calculate_final_speed(target_speed)
        self.current_speed = final_speed

        # 6) 모터 명령 발행
        motor_msg = Float32MultiArray()
        motor_msg.data = [float(final_angle), float(self.current_speed)]
        self.motor_pub.publish(motor_msg)

        # 7) 로깅
        self.log_status(heading_error_rad, cte_term, self.current_k, final_angle,
                        target_speed, final_speed, speed_status, rate_limited,
                        self.lane_state_from_curve, target_speed_state, target_speed_servo)

    # ==============================================================
    #                           로깅
    # ==============================================================
    def log_status(self, heading_error_rad, cte_term, current_k, final_angle,
                   target_speed, final_speed, speed_status, rate_limited,
                   curve_state='straight', target_speed_state=None, target_speed_servo=None):

        # 차선 타겟 모드 표시
        if self.current_mode == 'center':
            lane_target_str = "🟡 CENTER"
        elif self.current_mode == 'left':
            lane_target_str = "🔵 LEFT"
        elif self.current_mode == 'right':
            lane_target_str = "🔴 RIGHT"
        else:
            lane_target_str = "⚪ UNKNOWN"

        # 주행 모드 표시 (조향각 기준)
        if self.is_obstacle_detected:
            driving_mode = "🚧 OBSTACLE MODE"
        else:
            driving_mode = "🔄 CURVE MODE" if abs(final_angle) > self.steering_threshold else "➡️ STRAIGHT MODE"

        rate_limit_marker = "⛔" if rate_limited else ""
        obstacle_status = f"🚗 Dist: {self.relative_distance:.2f}m" if self.is_obstacle_detected else "🟢 Clear"

        curve_state_str = "Curve" if curve_state == 'curve' else "Straight"
        state_speed_line = f"  • Target Speed (State) : {target_speed_state:>6.2f}  [{curve_state_str}]\n" if target_speed_state is not None else ""
        servo_speed_line = f"  • Target Speed (Servo) : {target_speed_servo:>6.2f}\n" if target_speed_servo is not None else ""

        self.get_logger().info(
            "\n📐 [Integrated Stanley Controller Log] ---------------\n"
            f"  • Lane Target            : {lane_target_str}\n"
            f"  • Driving Mode           : {driving_mode}\n"
            f"  • Obstacle Status        : {obstacle_status}\n"
            f"  • Stanley K Value        : {current_k:>6.2f}\n"
            f"  • Heading Error          : {np.degrees(heading_error_rad):>6.2f}°\n"
            f"  • Cross Track Term       : {np.degrees(cte_term):>6.2f}°\n"
            f"  • Servo Angle            : {final_angle:>6.2f} {rate_limit_marker}\n"
            + state_speed_line
            + servo_speed_line +
            f"  • Target Speed (Final)   : {target_speed:>6.2f}\n"
            f"  • Final Speed (Applied)  : {final_speed:>6.2f} {speed_status}\n"
            "--------------------------------------------------------"
        )

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedStanleyController()
    try:
		    # 노드를 생성한 후, 메인 스핀 루프에 들어가기 전에 모든 토픽이 준비될 때까지 대기합니다.
        node.wait_for_message()
        
        # 메인 스핀 루프 진입
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
