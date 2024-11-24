#!/usr/bin/env python3
from ev3dev2.motor import LargeMotor, MoveTank, OUTPUT_A, OUTPUT_B
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3, INPUT_4
from ev3dev2.sensor.lego import ColorSensor, GyroSensor
from ev3dev2.sound import Sound
import time

class EV3Controller:
    def __init__(self):
        # Initialize motors
        self.tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
        self.left_motor = LargeMotor(OUTPUT_A)
        self.right_motor = LargeMotor(OUTPUT_B)
        
        # Initialize sensors
        try:
            self.color_sensor = ColorSensor(INPUT_1)
            self.gyro_sensor = GyroSensor(INPUT_2)
            self.has_sensors = True
        except:
            print("Warning: Some sensors not found. Running in basic mode.")
            self.has_sensors = False
        
        # Initialize sound
        self.sound = Sound()
        self.sound.beep()
        
        # Control parameters
        self.base_speed = 30  # Base speed percentage
        self.turn_speed = 20  # Turn speed percentage
        
    def move_forward(self, duration=None):
        """Move robot forward"""
        self.tank_drive.on(self.base_speed, self.base_speed)
        if duration:
            time.sleep(duration)
            self.stop()
            
    def move_backward(self, duration=None):
        """Move robot backward"""
        self.tank_drive.on(-self.base_speed, -self.base_speed)
        if duration:
            time.sleep(duration)
            self.stop()
    
    def turn_left(self, duration=None):
        """Turn robot left"""
        self.tank_drive.on(-self.turn_speed, self.turn_speed)
        if duration:
            time.sleep(duration)
            self.stop()
    
    def turn_right(self, duration=None):
        """Turn robot right"""
        self.tank_drive.on(self.turn_speed, -self.turn_speed)
        if duration:
            time.sleep(duration)
            self.stop()
    
    def stop(self):
        """Stop all motors"""
        self.tank_drive.off()
    
    def follow_line(self, duration=5):
        """Basic line following using color sensor"""
        if not self.has_sensors:
            print("Error: Color sensor not found")
            return
            
        start_time = time.time()
        while time.time() - start_time < duration:
            reflected_light = self.color_sensor.reflected_light_intensity
            
            if reflected_light < 30:  # Dark surface (line)
                self.tank_drive.on(self.base_speed - 10, self.base_speed + 10)
            else:  # Light surface
                self.tank_drive.on(self.base_speed + 10, self.base_speed - 10)
        
        self.stop()
    
    def square_dance(self):
        """Make robot perform a square dance pattern"""
        for _ in range(4):
            self.move_forward(2)
            self.turn_right(1)
        self.sound.beep()

def main():
    # Create robot controller
    robot = EV3Controller()
    
    try:
        # Example sequence
        print("Starting demo sequence...")
        robot.sound.speak("Starting demo")
        
        # Move in a square
        robot.square_dance()
        
        # Try line following if sensors available
        if robot.has_sensors:
            print("Starting line following...")
            robot.follow_line(10)
        
        # Final beep
        robot.sound.beep()
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        robot.stop()

if __name__ == "__main__":
    main()
