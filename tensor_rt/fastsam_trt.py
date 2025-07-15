"""
Convert FastSAM to TensorRT using Ultralytics' built-in export functionality. 
Since FastSAM is based on YOLOv8-seg, it supports the same export formats including TensorRT engine format

# Pip install
pip install nvidia-pyindex
pip install nvidia-tensorrt

Usage:
python3 /home/copter/jetson_benchmark/tensor_rt/fastsam_trt.py
"""

import os
from ultralytics import YOLO

# Load FastSAM model
model = YOLO('FastSAM-s.pt')

# TensorRT FP16 export (recommended for Jetson)
exported_paths = model.export( # Changed variable name to plural to reflect it's a list
    format='engine', 
    device='0', 
    imgsz=640, 
    half=True,  # FP16
    verbose=True
)

# --- Model Export ------------------------------------------

# Ensure exported_paths is not empty and get the first element
if exported_paths and isinstance(exported_paths, list):
    exported_path = exported_paths[0] # Get the first path from the list
else:
    print("Error: Model export did not return a valid path.")
    exit()

output_dir = "/home/copter/jetson_benchmark/trt_models"  
custom_name = "FastSAM-s_fp16_640.engine"
os.makedirs(output_dir, exist_ok=True)

final_path = os.path.join(output_dir, custom_name)
os.rename(exported_path, final_path)
print(f"Model saved to: {final_path}")