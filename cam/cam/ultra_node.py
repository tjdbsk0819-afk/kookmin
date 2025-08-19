#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Header
from custom_interfaces.msg import Ultrasonic
from std_srvs.srv import Trigger # <<< 1. Trigger 서비스 타입 임포트

class UltraNode(Node):
    def __init__(self):
        super().__init__('ultra_node')

        # 기존 xycar_ultrasonic 구독
        self.subscription = self.create_subscription(
            Int32MultiArray,
            'xycar_ultrasonic',
            self.ultra_callback,
            10
        )

        # 커스텀 토픽으로 publish
        self.publisher_ = self.create_publisher(Ultrasonic, 'ultrasonic', 10)

        # <<< 2. 서비스 서버 생성 및 상태 플래그 초기화
        self.is_active = False # 메인 로직 실행 여부를 결정하는 플래그
        self.start_service = self.create_service(
            Trigger,
            'start_track_driving', # 서비스 이름
            self.start_service_callback # 서비스 요청 시 호출될 콜백 함수
        )
        
        self.ultra_msg = None
        self.timer = None # 타이머를 미리 생성하지 않음

        self.get_logger().info("Waiting for ultrasonic data...")
        self.wait_for_message()
        self.get_logger().info("Ultrasonic Ready ----------")
        self.get_logger().info('Waiting for service call on /start_track_driving to begin publishing...')


    def ultra_callback(self, msg):
        self.ultra_msg = msg.data

    def wait_for_message(self):
        while self.ultra_msg is None and rclpy.ok():
            rclpy.spin_once(self)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

    # <<< 3. 서비스 콜백 함수 추가 (타이머 시작 로직 포함)
    def start_service_callback(self, request, response):
        """
        서비스 호출을 받으면 is_active 플래그를 True로 설정하고,
        주기적으로 데이터를 발행하는 타이머를 시작합니다.
        """
        if not self.is_active:
            self.is_active = True
            # 서비스 호출 시 타이머를 생성하고 시작
            if self.timer is None:
                self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
                self.get_logger().info('✅ Service called. Starting Ultrasonic publisher timer!')
                response.success = True
                response.message = 'Ultrasonic publisher started.'
            else: # 이미 타이머가 어떠한 이유로든 생성된 경우
                self.get_logger().warn('⚠️ Timer is already running.')
                response.success = False
                response.message = 'Timer already active.'
        else:
            self.get_logger().warn('⚠️ Main logic is already running.')
            response.success = False
            response.message = 'Already active.'

        return response
        
    def timer_callback(self):
        # 이 콜백은 is_active가 True가 된 후에만 실행되므로 별도 체크 불필요
        if self.ultra_msg is not None:
            # 커스텀 메시지 생성
            ultra_msg = Ultrasonic()
            ultra_msg.header.stamp = self.get_clock().now().to_msg()
            ultra_msg.header.frame_id = 'ultrasonic'
            ultra_msg.data = self.ultra_msg

            # publish
            self.publisher_.publish(ultra_msg)

def main(args=None):
    rclpy.init(args=args)
    node = UltraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()