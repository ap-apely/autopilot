import ev3dev2.motor as motor
import ev3dev2.sensor.lego as sensors
from ev3dev2.motor import SpeedPercent
from typing import Tuple
import numpy as np

class EV3Controller:
    def __init__(self):
        # Initialize motors
        self.left_motor = motor.LargeMotor('outB')
        self.right_motor = motor.LargeMotor('outC')
        self.tank_drive = motor.MoveTank('outB', 'outC')
        
        # Initialize sensors
        self.ultrasonic = sensors.UltrasonicSensor()
        
        # Control parameters
        self.base_speed = 30  # Base speed percentage
        self.max_turn_offset = 50  # Maximum turn adjustment
        self.min_safe_distance = 20  # Minimum safe distance in cm
        
    def stop(self):
        """Stop all motors"""
        self.tank_drive.off()
        
    def adjust_steering(self, lane_offset: float, curve_radius: float = None):
        """
        Adjust steering based on lane offset and curve radius
        lane_offset: Normalized offset from center (-1 to 1)
        curve_radius: Radius of road curve (if detected)
        """
        # Basic proportional control
        turn_adjustment = lane_offset * self.max_turn_offset
        
        # Adjust speeds for each motor
        left_speed = self.base_speed - turn_adjustment
        right_speed = self.base_speed + turn_adjustment
        
        # Clamp speeds to valid range (-100 to 100)
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))
        
        # Apply speeds to motors
        self.tank_drive.on(SpeedPercent(left_speed), SpeedPercent(right_speed))
        
    def check_obstacles(self) -> bool:
        """Check for obstacles using ultrasonic sensor"""
        distance = self.ultrasonic.distance_centimeters
        return distance < self.min_safe_distance
        
    def emergency_stop(self):
        """Emergency stop procedure"""
        self.stop()
        # Add any additional emergency procedures here
        
    def process_lane_detection(self, lane_points: np.ndarray) -> Tuple[float, float]:
        """
        Process lane detection points to get steering parameters
        Returns: (lane_offset, curve_radius)
        """
        if lane_points is None or len(lane_points) == 0:
            return 0.0, None
            
        # Calculate center offset
        image_center = 0.5  # Assuming normalized coordinates
        lane_center = np.mean(lane_points[:, 0])  # Average x-coordinate
        offset = (lane_center - image_center) * 2  # Scale to [-1, 1]
        
        # Simple curve detection
        if len(lane_points) > 2:
            # Fit a polynomial to estimate curve
            coeffs = np.polyfit(lane_points[:, 1], lane_points[:, 0], 2)
            curve_radius = 1 / coeffs[0] if abs(coeffs[0]) > 1e-6 else None
        else:
            curve_radius = None
            
        return offset, curve_radius
