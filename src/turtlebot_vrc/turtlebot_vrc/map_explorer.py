#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time
import subprocess
import os
import math


class MapExplorer(Node):

    WALL_DIST     = 0.5
    FORWARD_SPEED = 0.15
    TURN_SPEED    = 0.5

    def __init__(self):
        super().__init__('map_explorer')
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.ranges     = []
        self.start_time = None
        self.explore_time = 180
        self.state      = 'wait'
        self.turn_dir   = 1.0
        self.map_saved  = False
        self.turn_start = None
        self.scan_count = 0

        self.timer = self.create_timer(0.1, self.explore_loop)
        self.get_logger().info('Map Explorer ready — waiting for laser scan...')

    def scan_callback(self, msg):
        self.ranges = [r if (not math.isinf(r) and not math.isnan(r) and r > 0.05)
                       else 3.5 for r in msg.ranges]
        self.scan_count += 1
        if self.state == 'wait' and self.scan_count > 5:
            self.state      = 'forward'
            self.start_time = time.time()
            self.get_logger().info('Laser ready! Starting exploration!')

    def get_sector(self, angle_start, angle_end):
        if not self.ranges:
            return 3.5
        n     = len(self.ranges)
        start = int(angle_start * n / 360) % n
        end   = int(angle_end   * n / 360) % n
        if start <= end:
            sector = self.ranges[start:end]
        else:
            sector = self.ranges[start:] + self.ranges[:end]
        valid = [r for r in sector if 0.05 < r < 3.5]
        return min(valid) if valid else 3.5

    def explore_loop(self):
        if self.state == 'wait':
            return

        elapsed = time.time() - self.start_time

        if elapsed > self.explore_time and not self.map_saved:
            self.get_logger().info('Exploration complete! Saving map...')
            self._pub(0.0, 0.0)
            self.save_map()
            self.map_saved = True
            return

        if self.map_saved:
            return

        front       = self.get_sector(350, 10)
        front_left  = self.get_sector(10,  45)
        front_right = self.get_sector(315, 350)
        left        = self.get_sector(60,  120)
        right       = self.get_sector(240, 300)

        self.get_logger().info(
            f't={elapsed:.0f}s state={self.state} '
            f'F={front:.2f} FL={front_left:.2f} FR={front_right:.2f} '
            f'L={left:.2f} R={right:.2f}'
        )

        if self.state == 'forward':
            if front < self.WALL_DIST or front_left < 0.3 or front_right < 0.3:
                self.turn_dir   = 1.0 if left > right else -1.0
                self.state      = 'turn'
                self.turn_start = time.time()
                self.get_logger().info(
                    f'Wall! Turning {"left" if self.turn_dir>0 else "right"}'
                )
                self._pub(0.0, self.TURN_SPEED * self.turn_dir)
            else:
                az = 0.3 if right < 0.3 else (-0.3 if left < 0.3 else 0.0)
                self._pub(self.FORWARD_SPEED, az)

        elif self.state == 'turn':
            turn_elapsed = time.time() - self.turn_start
            if turn_elapsed > 1.0 and front > self.WALL_DIST * 1.5:
                self.state = 'forward'
                self.get_logger().info('Path clear — going forward')
            elif turn_elapsed > 4.0:
                self.turn_dir  *= -1.0
                self.turn_start = time.time()
                self.get_logger().info('Still stuck — reversing turn direction')
            else:
                self._pub(0.0, self.TURN_SPEED * self.turn_dir)

    def save_map(self):
        map_path = os.path.expanduser('~/hospital_map')
        try:
            result = subprocess.run(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                 '-f', map_path, '--ros-args', '-p', 'use_sim_time:=true'],
                capture_output=True, text=True, timeout=20
            )
            self.get_logger().info(f'Map saved! {result.stdout}')
        except Exception as e:
            self.get_logger().error(f'Save error: {e}')

    def _pub(self, lx, az):
        msg = Twist()
        msg.linear.x  = float(lx)
        msg.angular.z = float(az)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = MapExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
