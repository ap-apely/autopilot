from Lines.ufldDetector.utils import LaneModelType
from Object.ObjectDetector.utils import ObjectModelType

from App.control_panel import app_run

import asyncio

async def main():
    object_config = {
		"model_path": './Object/ObjectDetector/models/yolov8n.trt',
		"model_type" : ObjectModelType.YOLOV8,
		"classes_path" : './Object/ObjectDetector/models/coco_label.txt',
		"box_score" : 0.4,
		"box_nms_iou" : 0.5
	}
    line_config = {
		"model_path": "./Lines/models/culane.trt",
		"model_type" : LaneModelType.UFLDV2_CULANE
	}
    video_path = "video.mov"
    await app_run(object_config, line_config, video_path=None,ev3_controller=None)

asyncio.run(main())