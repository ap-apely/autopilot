import cv2, time
import numpy as np
import logging
import pycuda.driver as drv
import rich
from rich.console import Console
from rich.table import Table
import time

"""
Object imports:
 -ObjectTRACKER
 -TaskConditions
 -YOLOv8/YOLOv10
"""
from Object.ObjectTracker.byteTrack.byteTracker import BYTETracker
from Road.taskConditions import TaskConditions, Logger
from Object.ObjectDetector.yoloDetector import YoloDetector
from Object.ObjectDetector.utils import ObjectModelType,  CollisionType
from Object.ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
from Object.PathPlanning.path import path_plan

"""
Lines imports:
 -UFLDv2
 -BirdView
"""
from Lines import UltrafastLaneDetectorV2
from Lines.ufldDetector.perspectiveTransformation import PerspectiveTransformation
from Lines.ufldDetector.utils import LaneModelType, OffsetType, CurvatureType

LOGGER = Logger(None, logging.INFO, logging.INFO )

class ControlPanel(object):
	CollisionDict = {
						CollisionType.UNKNOWN : (0, 255, 255),
						CollisionType.NORMAL : (0, 255, 0),
						CollisionType.PROMPT : (0, 102, 255),
						CollisionType.WARNING : (0, 0, 255)
	 				}

	OffsetDict = { 
					OffsetType.UNKNOWN : (0, 255, 255), 
					OffsetType.RIGHT :  (0, 0, 255), 
					OffsetType.LEFT : (0, 0, 255), 
					OffsetType.CENTER : (0, 255, 0)
				 }

	CurvatureDict = { 
						CurvatureType.UNKNOWN : (0, 255, 255),
						CurvatureType.STRAIGHT : (0, 255, 0),
						CurvatureType.EASY_LEFT : (0, 102, 255),
						CurvatureType.EASY_RIGHT : (0, 102, 255),
						CurvatureType.HARD_LEFT : (0, 0, 255),
						CurvatureType.HARD_RIGHT : (0, 0, 255)
					}

	def __init__(self):
		collision_warning_img = cv2.imread('./App/assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
		self.collision_warning_img = cv2.resize(collision_warning_img, (100, 100))
		collision_prompt_img = cv2.imread('./App/assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
		self.collision_prompt_img = cv2.resize(collision_prompt_img, (100, 100))
		collision_normal_img = cv2.imread('./App/assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
		self.collision_normal_img = cv2.resize(collision_normal_img, (100, 100))
		left_curve_img = cv2.imread('./App/assets/left_turn.png', cv2.IMREAD_UNCHANGED)
		self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
		right_curve_img = cv2.imread('./App/assets/right_turn.png', cv2.IMREAD_UNCHANGED)
		self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
		keep_straight_img = cv2.imread('./App/assets/straight.png', cv2.IMREAD_UNCHANGED)
		self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
		determined_img = cv2.imread('./App/assets/warn.png', cv2.IMREAD_UNCHANGED)
		self.determined_img = cv2.resize(determined_img, (200, 200))
		left_lanes_img = cv2.imread('./App/assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
		self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
		right_lanes_img = cv2.imread('./App/assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
		self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))

		# Path planning visualization
		self.path_window = np.zeros((500, 500, 3), dtype=np.uint8)
		cv2.namedWindow("Path Planning")

		# FPS
		self.fps = 0
		self.frame_count = 0
		self.start = time.time()

		self.curve_status = None

	def _updateFPS(self) :
		"""
		Update FPS.

		Args:
			None

		Returns:
			None
		"""
		self.frame_count += 1
		if self.frame_count >= 30:
			self.end = time.time()
			self.fps = self.frame_count / (self.end - self.start)
			self.frame_count = 0
			self.start = time.time()

	def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25) :
		"""
		Display BirdView Panel on image.

		Args:
			main_show: video image.
			min_show: bird view image.
			show_ratio: display scale of bird view image.

		Returns:
			main_show: Draw bird view on frame.
		"""
		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		min_birdview_show = cv2.resize(min_show, (W, H))
		min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # 添加边框
		main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]: ] = min_birdview_show

	def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
		"""
		Display Signs Panel on image.

		Args:
			main_show: image.
			offset_type: offset status by OffsetType. (UNKNOWN/CENTER/RIGHT/LEFT)
			curvature_type: curature status by CurvatureType. (UNKNOWN/STRAIGHT/HARD_LEFT/EASY_LEFT/HARD_RIGHT/EASY_RIGHT)

		Returns:
			main_show: Draw sings info on frame.
		"""

		W = 400
		H = 365
		widget = np.copy(main_show[:H, :W])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,0:3] = [0, 0, 255]  #left
		widget[:,-3:-1] = [0, 0, 255] # right
		main_show[:H, :W] = widget

		if curvature_type == CurvatureType.UNKNOWN and offset_type in { OffsetType.UNKNOWN, OffsetType.CENTER } :
			y, x = self.determined_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.determined_img[y, x, :3]
			self.curve_status = None

		elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status== "Left") and \
			(curvature_type not in { CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT }) :
			y, x = self.left_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.left_curve_img[y, x, :3]
			self.curve_status = "Left"

		elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status== "Right") and \
			(curvature_type not in { CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT }) :
			y, x = self.right_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.right_curve_img[y, x, :3]
			self.curve_status = "Right"
		
		
		if ( offset_type == OffsetType.RIGHT ) :
			y, x = self.left_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.left_lanes_img[y, x, :3]
		elif ( offset_type == OffsetType.LEFT ) :
			y, x = self.right_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.right_lanes_img[y, x, :3]
		elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight" :
			y, x = self.keep_straight_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.keep_straight_img[y, x, :3]
			self.curve_status = "Straight"

		self._updateFPS()
		cv2.putText(main_show, "LDWS : " + offset_type.value, (10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.OffsetDict[offset_type], thickness=2)
		cv2.putText(main_show, "LKAS : " + curvature_type.value, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.CurvatureDict[curvature_type], thickness=2)
		cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

	def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.25) :
		"""
		Display Collision Panel on image.

		Args:
			main_show: image.
			collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
			obect_infer_time: object detection time -> float.
			lane_infer_time:  lane detection time -> float.

		Returns:
			main_show: Draw collision info on frame.
		"""

		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		widget = np.copy(main_show[H+20:2*H, -W-20:])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,-3:-1] = [0, 0, 255] #left
		widget[:,0:3] = [0, 0, 255]  # right
		main_show[H+20:2*H, -W-20:] = widget

		if (collision_type == CollisionType.WARNING) :
			y, x = self.collision_warning_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_warning_img[y, x, :3]
		elif (collision_type == CollisionType.PROMPT) :
			y, x =self.collision_prompt_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_prompt_img[y, x, :3]
		elif (collision_type == CollisionType.NORMAL) :
			y, x = self.collision_normal_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_normal_img[y, x, :3]

		cv2.putText(main_show, "FCWS : " + collision_type.value, ( main_show.shape[1]- int(W) + 100 , 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=self.CollisionDict[collision_type], thickness=2)
		cv2.putText(main_show, "object-infer : %.2f s" % obect_infer_time, ( main_show.shape[1]- int(W) + 100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
		cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, ( main_show.shape[1]- int(W) + 100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
		
	def DisplayPathPlanning(self, frame, objectDetector):
		"""
		Display path planning visualization in a separate window
		"""
		if hasattr(objectDetector, '_object_info') and objectDetector._object_info:
			# Convert detections to format expected by path_plan
			det_list = []
			for obj in objectDetector._object_info:
				x1, y1, x2, y2 = obj.tolist()
				conf = obj.conf
				cls = objectDetector.class_names.index(obj.label) if obj.label in objectDetector.class_names else -1
				det_list.append([x1, y1, x2, y2, conf, cls])
			
			# Convert colors dict to list format matching class indices
			colors_list = [objectDetector.colors_dict[name] for name in objectDetector.class_names]
			
			# Call path planning and get visualization
			path_window = path_plan(det_list, None, True, objectDetector.class_names, colors_list, frame)
			
			# Show path planning window
			if path_window is not None:
				cv2.imshow("Path Planning", path_window)

def app_run(object_config, line_config, video_path):
	console = Console()
	
	# Create a nice header
	console.print("\n[bold cyan]🚗 ADAS Simulation System[/bold cyan]", justify="center")
	console.print("[dim]Advanced Driver Assistance System[/dim]\n", justify="center")
	
	with console.status("[bold green]Loading system components...") as status:
		start_time_loading = time.time()
		
		# Initialize read and save video 
		console.print("[yellow]• Loading video source...[/yellow]")
		cap = cv2.VideoCapture(video_path)
		if (not cap.isOpened()):
			console.print("[red bold]❌ Error: Video path is invalid. Please check it.[/red bold]")
			raise Exception("video path is error. please check it.")
			
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		vout = cv2.VideoWriter('./output.mp4', fourcc, fps, (width, height))
		
		console.print("[yellow]• Initializing detection models...[/yellow]")
		# Initialize logger
		LOGGER = logging.getLogger("YOLO")
		LOGGER.setLevel(logging.INFO)
		
		# Initialize models
		if ( "UFLDV2" in line_config["model_type"].name) :
			UltrafastLaneDetectorV2.set_defaults(line_config)
			laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
		transformView = PerspectiveTransformation( (width, height) , logger=LOGGER)
		YoloDetector.set_defaults(object_config)
		objectDetector = YoloDetector(logger=LOGGER)
		distanceDetector = SingleCamDistanceMeasure()
		objectTracker = BYTETracker(names=objectDetector.colors_dict)
		# display panel
		displayPanel = ControlPanel()
		analyzeMsg = TaskConditions()
		
		loading_time = round(time.time() - start_time_loading, 2)
		console.print(f"[bold green]✓ System loaded successfully in {loading_time}s![/bold green]\n")
	
	# Create stats table
	table = Table(show_header=True, header_style="bold magenta")
	table.add_column("Component", style="cyan")
	table.add_column("Status", justify="right", style="green")
	table.add_column("Time", justify="right", style="yellow")
	
	table.add_row("Video Source", "Ready", f"{width}x{height}@{fps}fps")
	table.add_row("Object Detection", "Active", f"YOLOv8 ({object_config['model_type'].name})")
	table.add_row("Distance Detection", "Active", "Single Camera")
	table.add_row("Object Tracking", "Active", "BYTE Tracker")
	console.print(table)
	console.print("\n[bold cyan]Press 'Q' to quit simulation[/bold cyan]\n")
	
	frame_count = 0
	total_fps = 0
	
	while cap.isOpened():
		frame_start = time.time()
		ret, frame = cap.read() # Read frame from the video
		if ret:
			frame_show = frame.copy()
			
			#========================== Detect Model =========================
			obect_time = time.time()
			objectDetector.DetectFrame(frame)
			obect_infer_time = round(time.time() - obect_time, 2)

			if objectTracker :
				box   = [obj.tolist(format_type= "xyxy") for obj in objectDetector.object_info]
				score = [obj.conf for obj in objectDetector.object_info]
				id    = [obj.label for  obj in objectDetector.object_info]
				objectTracker.update(box, score, id, frame)

			lane_time = time.time()
			laneDetector.DetectFrame(frame)
			lane_infer_time = round(time.time() - lane_time, 4)
			
			#========================= Analyze Status ========================
			distanceDetector.updateDistance(objectDetector.object_info)
			vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.lane_info.area_points)

			if (analyzeMsg.CheckStatus() and laneDetector.lane_info.area_status ) :
				transformView.updateTransformParams(*laneDetector.lane_info.lanes_points[1:3], analyzeMsg.transform_status)
			birdview_show = transformView.transformToBirdView(frame_show)

			birdview_lanes_points = [transformView.transformToBirdViewPoints(lanes_point) for lanes_point in laneDetector.lane_info.lanes_points]
			(vehicle_direction, vehicle_curvature) , vehicle_offset = transformView.calcCurveAndOffset(birdview_show, *birdview_lanes_points[1:3])

			analyzeMsg.UpdateCollisionStatus(vehicle_distance, laneDetector.lane_info.area_status)
			analyzeMsg.UpdateOffsetStatus(vehicle_offset)
			analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

			#========================== Draw Results =========================
			transformView.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, analyzeMsg.offset_msg)
			transformView.DrawTransformFrontalViewArea(frame_show)
			laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
			laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
			objectDetector.DrawDetectedOnFrame(frame_show)
			objectTracker.DrawTrackedOnFrame(frame_show, False)
			distanceDetector.DrawDetectedOnFrame(frame_show)

			displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
			displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)	
			displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg, obect_infer_time, lane_infer_time )
			#displayPanel.DisplayPathPlanning(frame_show, objectDetector)  
			
			# Calculate and display FPS
			frame_time = time.time() - frame_start
			fps = 1 / frame_time
			frame_count += 1
			total_fps += fps
			
			if frame_count % 30 == 0:  # Update stats every 30 frames
				avg_fps = total_fps / frame_count
				console.print(f"[cyan]Frame {frame_count}[/cyan] | [yellow]FPS: {fps:.1f}[/yellow] | [green]Avg FPS: {avg_fps:.1f}[/green]", end="\r")
			
			cv2.imshow("ADAS Simulation", frame_show)
		else:
			break
			
		vout.write(frame_show)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	# Cleanup
	console.print("\n\n[bold green]✓ Simulation completed![/bold green]")
	console.print(f"[yellow]• Processed {frame_count} frames[/yellow]")
	console.print(f"[yellow]• Average FPS: {total_fps/frame_count:.1f}[/yellow]")
	console.print("[yellow]• Output saved to: output.mp4[/yellow]")
	
	cap.release()
	vout.release()
	cv2.destroyAllWindows()