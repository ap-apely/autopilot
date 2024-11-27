import cv2
import numpy as np
import os
import asyncio
from Lines.ufldDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
from Lines.ufldDetector.utils import LaneModelType
from rich.console import Console
from rich.panel import Panel
from rich.table import Table, box
from rich.traceback import install
from rich.syntax import Syntax
from rich.progress import track
from datetime import datetime
import traceback
from utils.error_handlers import (
    AutopilotError, ModelError, CameraError, ProcessingError,
    create_error_panel, handle_camera_error, handle_runtime_error,
    handle_fatal_error, print_warning, console
)

# Install rich traceback handler
install(show_locals=True)

class SimulatedController:
    def __init__(self):
        # Control parameters
        self.base_speed = 30  # Base speed percentage
        self.max_turn_offset = 50  # Maximum turn adjustment
        
    def process_lane_detection(self, lane_points):
        """
        Process lane detection points to get steering parameters
        Returns: (lane_offset, curve_radius, has_road)
        """
        if lane_points is None or len(lane_points) == 0:
            return 0.0, None, False
            
        # Extract ego lanes (left-ego and right-ego)
        left_ego = lane_points[1]  # Index 1 is left-ego lane
        right_ego = lane_points[2]  # Index 2 is right-ego lane
        
        # If no ego lanes detected, return no offset and no road
        if len(left_ego) == 0 and len(right_ego) == 0:
            return 0.0, None, False
            
        # Calculate center between ego lanes
        if len(left_ego) > 0 and len(right_ego) > 0:
            # Both lanes detected - use average
            left_x = np.mean([pt[0] for pt in left_ego])
            right_x = np.mean([pt[0] for pt in right_ego])
            center_x = (left_x + right_x) / 2
        elif len(left_ego) > 0:
            # Only left ego lane - estimate center
            left_x = np.mean([pt[0] for pt in left_ego])
            center_x = left_x + 320  # Assume standard lane width
        elif len(right_ego) > 0:
            # Only right ego lane - estimate center
            right_x = np.mean([pt[0] for pt in right_ego])
            center_x = right_x - 320  # Assume standard lane width
        else:
            return 0.0, None, False
            
        # Calculate normalized offset (-1 to 1)
        image_center = 640  # Half of 1280 width
        offset = (center_x - image_center) / image_center
        
        # Calculate curve radius if enough points
        curve_radius = None
        if len(left_ego) > 2 or len(right_ego) > 2:
            # Use the lane with more points
            points = left_ego if len(left_ego) > len(right_ego) else right_ego
            if len(points) > 2:
                # Fit a polynomial to estimate curve
                x_coords = [pt[0] for pt in points]
                y_coords = [pt[1] for pt in points]
                coeffs = np.polyfit(y_coords, x_coords, 2)
                curve_radius = 1 / coeffs[0] if abs(coeffs[0]) > 1e-6 else None
                
        return offset, curve_radius, True

    def get_status_table(self, lane_offset, curve_radius, has_road, left_speed, right_speed):
        """Create a Rich table with current status"""
        table = Table(title="Vehicle Status", box=box.ROUNDED)
        
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Road Detected", "✅ Yes" if has_road else "❌ No")
        table.add_row("Lane Offset", f"{lane_offset:.2f}")
        if curve_radius:
            table.add_row("Curve Radius", f"{curve_radius:.2f}")
        table.add_row("Left Motor", f"{left_speed:.1f}%")
        table.add_row("Right Motor", f"{right_speed:.1f}%")
        
        direction = "Straight ahead" if abs(lane_offset) < 0.1 else \
                   f"Turning right ({abs(lane_offset):.2f})" if lane_offset > 0 else \
                   f"Turning left ({abs(lane_offset):.2f})"
        table.add_row("Direction", direction)
        
        return table

    def simulate_control(self, lane_offset, curve_radius=None):
        """Simulate control commands based on lane detection"""
        # Basic proportional control
        turn_adjustment = lane_offset * self.max_turn_offset
        
        # Calculate motor speeds
        left_speed = self.base_speed - turn_adjustment
        right_speed = self.base_speed + turn_adjustment
        
        # Clamp speeds
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))
        
        return left_speed, right_speed

async def process_frame(frame, lane_detector, controller):
    """Process a single frame asynchronously"""
    try:
        # Ensure frame is in the correct format
        if frame is None:
            raise CameraError("Frame is None - Camera may be disconnected")

        # Print frame info for debugging
        console.print(f"[cyan]Frame Info[/cyan]\n"
                      f"Shape: {frame.shape}\n"
                      f"Type: {frame.dtype}\n"
                      f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        # Ensure frame is in BGR format and correct size
        if frame.shape[2] != 3:
            raise ProcessingError(f"Invalid frame format: expected 3 channels, got {frame.shape[2]}")
        
        # Resize frame if needed (model expects specific size)
        target_size = (1280, 720)
        if frame.shape[:2] != target_size[::-1]:
            console.print(f"[yellow]⚠️ Resizing frame from {frame.shape[:2]} to {target_size[::-1]}[/yellow]")
            frame = cv2.resize(frame, target_size)

        # Run lane detection
        try:
            lane_detector.DetectFrame(frame)
            lane_points = lane_detector.lane_info.lanes_points
        except Exception as e:
            raise ModelError(f"Lane detection failed: {str(e)}")
        
        # Create lane detection stats table
        lane_table = Table(title="Lane Detection Stats", box=box.ROUNDED)
        lane_table.add_column("Lane", style="cyan")
        lane_table.add_column("Points", style="green")
        lane_table.add_column("Status", style="yellow")
        
        if lane_points is not None:
            for i, lane in enumerate(lane_points):
                status = "✅ Good" if len(lane) > 5 else "⚠️ Poor" if len(lane) > 0 else "❌ None"
                lane_table.add_row(
                    f"Lane {i}",
                    str(len(lane)),
                    status
                )
        console.print(lane_table)
        
    except Exception as e:
        error_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        error_panel = create_error_panel(
            f"[red bold]Error at {error_time}[/red bold]\n"
            f"[red]{str(e)}[/red]\n\n"
            f"[yellow]Stack Trace:[/yellow]\n"
            f"{Syntax(traceback.format_exc(), 'python', background_color='default')}",
            title="🚨 Error Details",
            border_style="red"
        )
        console.print(error_panel)
        return frame

    # Process lane detection results
    offset, curve_radius, has_road = controller.process_lane_detection(lane_points)
    
    # Simulate control commands
    left_speed, right_speed = controller.simulate_control(offset, curve_radius)
    
    # Update status table
    status_table = controller.get_status_table(offset, curve_radius, has_road, left_speed, right_speed)
    console.print(status_table)
    
    # Create debug visualization
    debug_frame = frame.copy()
    
    # Draw filled road area if we have lanes
    try:
        if lane_points is not None and len(lane_points) >= 2:
            H, W = debug_frame.shape[:2]
            lane_segment_img = debug_frame.copy()
            
            # Create polygon points for the area between lanes
            left_lane = lane_points[0]
            right_lane = lane_points[1]
            
            if len(left_lane) > 0 and len(right_lane) > 0:
                polygon_points = np.vstack((left_lane, np.flipud(right_lane)))
                cv2.fillPoly(lane_segment_img, pts=[polygon_points], color=(0, 255, 0))
                debug_frame[:H,:W,:] = cv2.addWeighted(debug_frame, 0.7, lane_segment_img, 0.3, 0)
    except Exception as e:
        print_warning(f"Failed to draw road area: {str(e)}")

    # Draw lane points
    if lane_points is not None:
        try:
            for lane_points_set in lane_points:
                color = (0, 255, 0)  # Green for all lanes
                for point in lane_points_set:
                    if len(point) == 2:  # Check if point is valid
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(debug_frame, (x, y), 5, color, -1)
        except Exception as e:
            print_warning(f"Failed to draw lane points: {str(e)}")
    
    return debug_frame

async def main():
    try:
        # Check if model file exists
        model_path = "./Lines/models/tusimple32.trt"
        if not os.path.exists(model_path):
            raise ModelError(f"Model file not found: {model_path}")
        
        console.print(Panel(
            f"[cyan]Model:[/cyan] {model_path}\n"
            f"[cyan]Status:[/cyan] Initializing lane detection system...",
            title="Initialization",
            border_style="blue"
        ))
        
        line_config = {
            "model_path": model_path,
            "model_type": LaneModelType.UFLDV2_TUSIMPLE
        }
        
        lane_detector = UltrafastLaneDetectorV2(
            model_path=line_config["model_path"],
            model_type=line_config["model_type"]
        )
        
        # Initialize simulated controller
        controller = SimulatedController()
        
        # Initialize camera
        try:
            console.print("[cyan]🎥 Initializing camera...[/cyan]")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                raise CameraError("Could not open camera")
                
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Verify camera settings
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if actual_width != 1280 or actual_height != 720:
                print_warning(f"Camera resolution mismatch")
                print_warning(f"Requested: 1280x720, Got: {actual_width}x{actual_height}")
            else:
                console.print(f"[green]✅ Camera initialized: {actual_width}x{actual_height}[/green]")
            
        except Exception as e:
            handle_camera_error(e)
            return
        
        try:
            console.print("[green]🚀 Starting main loop...[/green]")
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    raise CameraError("Failed to grab frame")
                
                # Process frame asynchronously
                debug_frame = await process_frame(frame, lane_detector, controller)
                
                if debug_frame is not None and debug_frame.size > 0:
                    cv2.imshow('Lane Detection Simulation', debug_frame)
                
                # Check for exit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    console.print("[cyan]👋 Exiting...[/cyan]")
                    break
                    
        except Exception as e:
            handle_runtime_error(e)
            
        finally:
            # Clean up
            console.print("[cyan]🧹 Cleaning up...[/cyan]")
            cap.release()
            cv2.destroyAllWindows()

    except Exception as e:
        handle_fatal_error(e)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("[cyan]👋 Gracefully shutting down...[/cyan]")
    except Exception as e:
        handle_fatal_error(e)
