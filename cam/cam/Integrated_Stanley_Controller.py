#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray, String
from collections import deque
from custom_interfaces.msg import ObstacleState, XycarState, Curve
from std_srvs.srv import Trigger # <<< 1. Trigger 서비스 타입 임포트

class IntegratedStanleyController(Node):
    def __init__(self):
        super().__init__('integrated_stanley_controller')

        # ==============================================================
        #                       ROS2 인터페이스
        # ==============================================================
        # 구독자들
        self.xycar_state_sub = self.create_subscription(XycarState,'/xycar_state',self.xycar_state_callback,10)
        self.obstacle_state_sub = self.create_subscription(ObstacleState,'/obstacle_state',self.obstacle_state_callback,10)
        self.curve_subscription = self.create_subscription(Curve, '/center_curve',self.curve_callback,10)
        # 퍼블리셔
        self.motor_pub = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.last_time = self.get_clock().now()
        
        # <<< 2. 서비스 서버 생성 및 상태 플래그 초기화
        self.is_active = False # 메인 제어 로직 실행 여부를 결정하는 플래그
        self.start_service = self.create_service(
            Trigger,
            'start_track_driving', # 서비스 이름
            self.start_service_callback # 서비스 요청 시 호출될 콜백 함수
        )
        
        self.get_logger().info('🎯 Integrated Stanley Controller Node Successfully Started❗')
        self.get_logger().info('Waiting for service call on /start_track_driving to begin processing...')


        # ==============================================================
        #                   Stanley Controller 파라미터
        # ==============================================================
        # 1. compute_steering_angle_stanley
        self.image_center_x = 320                   # 이미지 중앙 x좌표
        self.pixel_to_meter = 1.9 / 650.0           # 픽셀 → 미터 환산계수
        self.heading_weight = 0.4                   # 헤딩오차 반영 비율
        self.deg_to_servo_scale = 5.0 / 3.0
        self.deg_to_servo_offset = 0.0

        # 2. steering angle utilities
        self.max_delta_angle = 10.0
        self.window = 3

        # 3. speed and k(gain) utilities
        self.steering_threshold = 3.0              # curve구간 판단 임계 조향각

        self.k_straight = 1.0                      
        self.k_curve = 1.2              
        self.k_obstacle = 1.5          

        self.speed_straight = 30.0                      
        self.speed_curve = 20.0     

        # ==============================================================
        #                   Obstacle Follower 파라미터
        # ==============================================================
        self.TARGET_TIME_GAP = 10.0
        self.OBSTACLE_DETECTION_THRESHOLD = 2.0    # 감지 거리
        self.KP_SPEED_CONTROL = 0.5                 # 제어 게인
        self.MIN_SPEED = 4.0
        # 부드러운 가감속을 위한 파라미터
        self.ACCEL_STEP = 0.4
        self.DECEL_STEP = 0.7

        # ==============================================================
        #                       상태 변수들
        # ==============================================================
        # Stanley Controller 변수들
        self.last_steering_deg = 0.0
        self.angle_buffer = deque(maxlen=self.window)
        self.current_k = self.k_straight                       
        self.current_mode = 'center'               # 초기 타겟 포인트

        # Obstacle Follower 변수들
        self.relative_distance = self.OBSTACLE_DETECTION_THRESHOLD + 1.0
        self.is_obstacle_detected = False
        self.current_speed = self.MIN_SPEED        # 현재 실제 속도
        self.obstacle_position = 'none'
        self.detected_vehicle_count = 0
        self.curve_state = 'straight'
        
        # 기타
        self.min_allowed_speed = min(self.speed_curve, self.speed_straight)
        self.max_allowed_speed = max(self.speed_curve, self.speed_straight)

    # <<< 3. 서비스 콜백 함수 추가
    def start_service_callback(self, request, response):
        """
        서비스 호출을 받으면 is_active 플래그를 True로 설정하고,
        메인 제어 로직(xycar_state_callback)이 실행되도록 합니다.
        """
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
        """
        Stanley Controller의 목표 속도와 k값을 계산합니다.
        조향각이 임계값을 넘는 경우에도 커브로 간주하는 안전 로직이 추가되었습니다.
        """
        # 조건 1: 차선 정보가 커브라고 명시 ('Curve')
        # 조건 2: 최종 조향각의 절대값이 설정된 임계값보다 큼
        is_curve_situation = (abs(servo_angle) > self.steering_threshold)

        # 장애물 감지 시에는 k값만 설정하고 속도는 동적으로 제어
        if self.is_obstacle_detected:
            self.current_k = self.k_obstacle
            # 장애물 모드에서도 커브/직선 상황에 따라 기본 목표 속도는 설정
            if is_curve_situation:
                target_speed = self.speed_curve
            else:
                target_speed = self.speed_straight
        # 장애물이 없을 경우
        else:
            if is_curve_situation:
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
            condition1 = is_in_curve and self.is_obstacle_detected and self.current_mode == 'center'
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
                
                # 로그에 차량 개수도 표시
                return final_speed, f"🚧 FOLLOWING (V:{self.detected_vehicle_count})"
            
            else: # 장애물이 없거나, 다른 차선에 있거나, 1대일 경우
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
        """중앙 차선의 커브 여부 콜백"""
        self.curve_state = msg.state

    def obstacle_state_callback(self, msg: ObstacleState):
        """장애물 상태(거리 및 위치) 정보 콜백"""
        # 메시지에서 거리와 위치 정보를 한 번에 업데이트
        self.relative_distance = msg.distance_m
        self.obstacle_position = msg.position.strip().lower()
        self.detected_vehicle_count = msg.vehicle_count
        # 장애물 감지 여부 판단
        self.is_obstacle_detected = self.relative_distance < self.OBSTACLE_DETECTION_THRESHOLD

    def xycar_state_callback(self, msg: XycarState):
        """차선/주행모드 정보 콜백 - 메인 제어 로직"""
        # <<< 4. 메인 로직 실행 전, 활성화 상태 체크
        if not self.is_active:
            return # 비활성화 상태이면 함수를 즉시 종료

        # 1. 메시지로부터 차선 정보와 주행 모드를 한 번에 업데이트
        self.current_mode = msg.drive_mode.strip().lower()
        
        lane_center_x = msg.target_point.x
        lane_center_y = msg.target_point.y
        lane_angle_rad = msg.target_point.z

        # 2. Stanley Controller로 조향각 계산 (기존 lane_callback 로직과 동일)
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

        # 3. 조향각 후처리 (기존과 동일)
        raw_delta = servo_angle - self.last_steering_deg
        rate_limited = abs(raw_delta) > self.max_delta_angle
        limited_angle = self.limit_angle_change(servo_angle, self.last_steering_deg, self.max_delta_angle)
        smoothed_angle = self.smooth_angle(limited_angle)
        final_angle = self.clamp_angle(smoothed_angle)
        self.last_steering_deg = final_angle

        # 4. 목표 속도 계산 (기존과 동일)
        target_speed, current_k = self.get_target_speed_and_k(smoothed_angle)

        # 5. 최종 속도 결정 (기존과 동일)
        final_speed, speed_status = self.calculate_final_speed(target_speed)
        self.current_speed = final_speed

        # 6. 모터 명령 발행 (기존과 동일)
        motor_msg = Float32MultiArray()
        motor_msg.data = [final_angle, float(self.current_speed)]
        self.motor_pub.publish(motor_msg)

        # 7. 로깅 (기존과 동일)
        self.log_status(heading_error_rad, cte_term, current_k, final_angle, 
                       target_speed, final_speed, speed_status, rate_limited)
        
    def log_status(self, heading_error_rad, cte_term, current_k, final_angle, 
                    target_speed, final_speed, speed_status, rate_limited):
            """상태 로깅"""
            # 차선 타겟 모드 표시
            if self.current_mode == 'center':
                lane_target_str = "🟡 CENTER"
            elif self.current_mode == 'left':
                lane_target_str = "🔵 LEFT"
            elif self.current_mode == 'right':
                lane_target_str = "🔴 RIGHT"
            else:
                lane_target_str = "⚪ UNKNOWN"

            # 주행 모드 표시 (장애물 감지 및 조향각 기반)
            if self.is_obstacle_detected:
                driving_mode = "🚧 OBSTACLE MODE"
            else:
                # 차선 정보가 'Curve'이거나 조향각이 임계값을 넘으면 CURVE MODE로 판단
                is_curve_situation = (self.curve_state.lower() == "curve") or (abs(final_angle) > self.steering_threshold)
                if is_curve_situation:
                    driving_mode = "🔄 CURVE MODE"
                else:
                    driving_mode = "➡️ STRAIGHT MODE"
            # Rate limiting 표시
            rate_limit_marker = "⛔" if rate_limited else ""
            
            # 장애물 상태 표시
            obstacle_status = f"🚗 Dist: {self.relative_distance:.2f}m" if self.is_obstacle_detected else "🟢 Clear"

            # 로그 출력 (매 프레임)
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
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()