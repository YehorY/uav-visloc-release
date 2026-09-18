import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class GlobalMapExtractor:
    @staticmethod
    def extract_anchors(image_path: str, model_path: str) -> tuple[np.ndarray, np.ndarray]:
        print("Processing 4K Global Map via SAHI...")

        # LEVEL 0 — device allocation: use CUDA when a GPU-enabled torch build is present,
        # otherwise fall back to CPU. (Was hardcoded to "cpu", which forced the 40-slice
        # boot-time map extraction onto the CPU even on a CUDA-capable machine.)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] SAHI map extraction device: {device}")

        # Load YOLOv8 model via SAHI
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.3,
            device=device
        )
        
        # Perform sliced inference
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        anchors = []
        classes = []
        for obj in result.object_prediction_list:
            cls_id = int(obj.category.id)
            # Assuming classes [0, 1, 2] represent our static infrastructure like buildings/roads
            # as seen in the main pipeline. Exclude dynamic objects.
            if cls_id in [0, 1, 2]:
                bbox = obj.bbox
                # Calculate centroid (cx, cy)
                cx = bbox.minx + (bbox.maxx - bbox.minx) / 2.0
                cy = bbox.miny + (bbox.maxy - bbox.miny) / 2.0
                anchors.append([cx, cy])
                classes.append(cls_id)
                
        anchors_array = np.array(anchors)
        classes_array = np.array(classes)
        print(f"Found {len(anchors_array)} anchors on the global map.")
        
        return anchors_array, classes_array
