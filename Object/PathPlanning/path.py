from runmidas import run
from midas.model_loader import load_model, default_models
import torch
from astar import path_planning
from utils.plots import plot_one_box
import random
import cv2

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
    external_points_set = []
    for *xyxy, conf, cls in reversed(det):  
        if draw_boxes:  # Add bbox to image
            label = f'{midas_names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, midas_frame, label=label, color=midas_colors[int(cls)], line_thickness=2)
            plot_one_box(xyxy, midas_img, label=label, color=None, line_thickness=1)

        external_points_set.append((int((int(xyxy[0])+int(xyxy[2]))/2), int(xyxy[3])))

    if external_points_set != None:
        for external_point in external_points_set.copy():
            height, width, channel = car.shape
            # print(f"height {height}, width {width}")
            # external point should be center but due to opencv its top left corner

            x, y = external_point
            # opencv point is shift of origin
            opencv_point = (x-int(height/2), y-int(width/2))
            x, y = opencv_point

            # error checking for out of region points
            if x <= 0 or y <= 0:
                # print(f"point {external_point} is not displayed on top view beacause its out of region")
                external_points_set.remove(external_point)
                continue
            # print(f"opencv points {x} and  {y}")

            # I want to put logo on opencv point, So I create a ROI
            rows,cols,channels = car.shape
            # print(f"rows {rows}, columns {cols}")

            # error checking for out of region points
            if rows+x >= background_height or cols+y >= background_width:
                # print(f"point {external_point} is not displayed on top view beacause its out of region")
                external_points_set.remove(external_point)
                continue

            roi = background[x:rows+x, y:cols+y ]

            # Now create a mask of logo and create its inverse mask also
            img2gray = cv2.cvtColor(car,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(car,car,mask = mask)


            # Put logo in ROI and modify the main image
            dst = cv2.add(img1_bg,img2_fg)
            background[x:rows+x, y:cols+y ] = dst

        path=path_planning(external_points_set)
        if path!=None:        
            for i in path:
                background= cv2.circle(background,(i[1],i[0]), radius=0, color=(188, 145, 42), thickness=-1)
        else:
            pass