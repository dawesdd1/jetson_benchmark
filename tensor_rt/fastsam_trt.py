"""
Convert FastSAM to TensorRT using Ultralytics' built-in export functionality. 
Since FastSAM is based on YOLOv8-seg, it supports the same export formats including TensorRT engine format
"""

import os
from ultralytics import YOLO

# Load FastSAM model
model = YOLO('FastSAM-s.pt')

# TensorRT FP16 export (recommended for Jetson)
exported_path = model.export(format='engine', device='0', imgsz=640, half=True)

# Or TensorRT FP32 export
# exported_path = model.export(format='engine', device='0', imgsz=640)

# Or TensorRT INT8 export (for maximum optimization)
# exported_path = model.export(format='engine', device='0', imgsz=640, int8=True)

# --- Model Export ------------------------------------------

# Move/rename to your desired location
output_dir = "/path/to/your/output/directory"
custom_name = "FastSAM-s_fp16_640.engine"
os.makedirs(output_dir, exist_ok=True)

final_path = os.path.join(output_dir, custom_name)
os.rename(exported_path, final_path)
print(f"Model saved to: {final_path}")