from .runmidas import run
from .midas.model_loader import load_model, default_models
import torch
from .astar import path_planning
from .utils.plots import plot_one_box
import random
import cv2
import numpy as np

def midad_preprocess(img: str, model_path: str) -> None:
    """
    Run MiDaS model on a given image.
    
    Parameters:
        img (str): Path to the input image.
        model_path (str): Path to the pre-trained MiDaS model checkpoint.
    """
    
    # Optimization flags
    midas_optimize: bool = True  # Whether to enable optimization in the model
    midas_square: bool = False   # Whether to resize the input image to a square shape
    midas_side: bool = True      # Whether to consider the side length of the input image
    midas_grayscale: bool = True  # Whether to convert the input image to grayscale

    # Model and output configuration
    midas_height: int = None       # Desired height of the output image (optional)
    midas_input_path: str = None   # Path to save the input image (optional)
    midas_output_path: str = None  # Path to save the output image (optional)

    # Model type and configuration
    midas_model_type: str = "dpt_beit_large_512"  # Type of pre-trained MiDaS model to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_midas, transform, net_w, net_h = load_model(device, model_path, midas_model_type, midas_optimize, midas_height, midas_square)
    print("MiDaS model loaded")

    midas_img = run(img, model_midas, transform, net_w, net_h, device,
        midas_input_path, midas_output_path, midas_model_type, midas_optimize,
        midas_side, midas_height, midas_square, midas_grayscale)

    midas_names = model_midas.module.names if hasattr(model_midas, 'module') else model_midas.names
    midas_colors = [[random.randint(0, 255) for _ in range(3)] for _ in midas_names]

    return midas_img, midas_names, midas_colors

def path_plan(det, midas_img, draw_boxes, midas_names, midas_colors, midas_frame) -> None:
    # Create visualization window with grid lines pre-drawn
    background = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw grid more efficiently using array operations
    background[:, ::50] = (50, 50, 50)
    background[::50, :] = (50, 50, 50)
    
    # Convert detections to obstacle points more efficiently
    external_points_set = []
    for *xyxy, conf, cls in reversed(det):
        if draw_boxes:
            label = f'{midas_names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, midas_frame, label=label, color=midas_colors[int(cls)], line_thickness=2)
        
        # Get bottom center point and scale in one operation
        x_center = int((int(xyxy[0]) + int(xyxy[2])) / 2 * 500 / midas_frame.shape[1])
        y_bottom = int(int(xyxy[3]) * 500 / midas_frame.shape[0])
        
        external_points_set.append((y_bottom, x_center))
        cv2.circle(background, (x_center, y_bottom), 10, (0, 0, 255), -1)
    
    # Calculate path using A* algorithm
    if external_points_set:
        path = path_planning(external_points_set)
        if path:
            # Draw path more efficiently
            path_points = np.array(path)
            for i in range(len(path)-1):
                pt1 = (path_points[i][1], path_points[i][0])
                pt2 = (path_points[i+1][1], path_points[i+1][0])
                cv2.line(background, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(background, pt1, 3, (0, 255, 255), -1)
            
            # Draw start and goal points
            cv2.circle(background, (path_points[0][1], path_points[0][0]), 5, (255, 0, 0), -1)
            cv2.circle(background, (path_points[-1][1], path_points[-1][0]), 5, (0, 255, 0), -1)
    
    # Add text labels
    cv2.putText(background, "Path Planning Visualization", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(background, "Red: Obstacles", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(background, "Green: Path", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(background, "Blue: Start", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return background