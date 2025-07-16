"""
Convert FastSAM to TensorRT using Ultralytics' built-in export functionality. 
Since FastSAM is based on YOLOv8-seg, it supports the same export formats including TensorRT engine format

Usage:
conda create -n tensorrt python=3.10
conda activate tensorrt
pip install tensorrt
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
python3 tensor_rt/fastsam_trt_convert.py
"""

import os
from ultralytics import YOLO

# Define the models and precisions to export
model_configs = {
    'FastSAM-x': ['fp32', 'fp16'],
    'FastSAM-s': ['fp32', 'fp16']
}

output_dir = "/home/dawesdd1/repos/jetson_benchmark/tensorrt_engines"
os.makedirs(output_dir, exist_ok=True)

print("--- 🚀 Starting FastSAM to TensorRT Conversion ---")

for model_name, precisions in model_configs.items():
    print(f"\nProcessing {model_name}...")
    # Load FastSAM model
    model = YOLO(f'/home/dawesdd1/repos/FastSAM/weights/{model_name}.pt')

    for precision in precisions:
        half_precision = (precision == 'fp16')
        custom_name = f"{model_name}_{precision}_640.engine"
        final_path = os.path.join(output_dir, custom_name)

        print(f"Exporting {model_name} to TensorRT {precision.upper()} engine as {custom_name}...")

        try:
            exported_path = model.export(
                format='engine',
                device='0',  # Assuming GPU device '0'. Adjust if necessary.
                imgsz=640,
                half=half_precision,
                verbose=True
            )

            # Ultralytics export typically returns the path to the exported file
            # If the file is not directly in the output_dir, we move it.
            if os.path.exists(exported_path):
                os.rename(exported_path, final_path)
                print(f"Model successfully saved to: {final_path}")
            else:
                print(f"Error: Exported file not found at {exported_path}. Check Ultralytics export process.")

        except Exception as e:
            print(f"An error occurred during export of {model_name} {precision.upper()}: {e}")

print("\n--- ✅ FastSAM TensorRT Conversion Complete ---")