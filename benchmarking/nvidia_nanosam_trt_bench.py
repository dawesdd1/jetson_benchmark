"""
Requiremnts:
TensorRT (10.3.0) engine
pip install pycuda


Usage:
conda activate NanoSAM
export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

  
### NVIDIA Benches
# FP16
python /home/copter/jetson_benchmark/nvidia_nanosam_trt_bench.py \
  --image_encoder_path "/home/copter/engine_models/nvidia_nanosam_resnet18_image_encoder_fp16.engine" \
  --mask_decoder_path "/home/copter/engine_models/nvidia_nanosam_mask_decoder_fp16.engine" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --device cuda \
  --num_runs 50 \
  --output_csv "/home/copter/jetson_benchmark/output/nvidia_nanosam_direct_trt_bench_fp16.csv"

#FP32
python /home/copter/jetson_benchmark/nvidia_nanosam_trt_bench.py \
  --image_encoder_path "/home/copter/engine_models/nvidia_nanosam_resnet18_image_encoder_fp32.engine" \
  --mask_decoder_path "/home/copter/engine_models/nvidia_nanosam_mask_decoder_fp32.engine" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --device cuda \
  --num_runs 50 \
  --output_csv "/home/copter/jetson_benchmark/output/nvidia_nanosam_direct_trt_bench_fp32.csv"
"""

import argparse
import os
import glob
import time
import torch
import numpy as np
import csv
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from tqdm import tqdm 
from datetime import datetime
import cv2

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert bytes to MB
    return 0

class DirectTensorRTPredictor:
    """Direct TensorRT predictor that bypasses torch2trt"""
    
    def __init__(self, image_encoder_path, mask_decoder_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load image encoder
        print("🔧 Loading image encoder engine...")
        self.encoder_engine = self._load_engine(image_encoder_path)
        self.encoder_context = self.encoder_engine.create_execution_context()
        
        # Load mask decoder  
        print("🔧 Loading mask decoder engine...")
        self.decoder_engine = self._load_engine(mask_decoder_path)
        self.decoder_context = self.decoder_engine.create_execution_context()
        
        # Setup memory allocations
        self._setup_encoder_memory()
        self._setup_decoder_memory()
        
        print("✅ Direct TensorRT predictor initialized")
    
    def _load_engine(self, engine_path):
        """Load a TensorRT engine from file"""
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(self.logger)
        return runtime.deserialize_cuda_engine(engine_data)
    
    def _setup_encoder_memory(self):
        """Setup memory allocations for image encoder"""
        self.encoder_inputs = {}
        self.encoder_outputs = {}
        self.encoder_bindings = []
        self.encoder_stream = cuda.Stream()
        
        for i in range(self.encoder_engine.num_io_tensors):
            name = self.encoder_engine.get_tensor_name(i)
            shape = self.encoder_engine.get_tensor_shape(name)
            dtype = self.encoder_engine.get_tensor_dtype(name)
            mode = self.encoder_engine.get_tensor_mode(name)
            
            # Convert TensorRT dtype to numpy
            if dtype == trt.DataType.FLOAT:
                np_dtype = np.float32
            elif dtype == trt.DataType.HALF:
                np_dtype = np.float16
            else:
                np_dtype = np.float32
            
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.encoder_bindings.append(int(device_mem))
            
            if mode == trt.TensorIOMode.INPUT:
                self.encoder_inputs[name] = {
                    'host': host_mem, 'device': device_mem, 'shape': shape
                }
            else:
                self.encoder_outputs[name] = {
                    'host': host_mem, 'device': device_mem, 'shape': shape
                }
    
    def _setup_decoder_memory(self):
        """Setup memory allocations for mask decoder"""
        self.decoder_inputs = {}
        self.decoder_outputs = {}
        self.decoder_bindings = []
        self.decoder_stream = cuda.Stream()
        
        for i in range(self.decoder_engine.num_io_tensors):
            name = self.decoder_engine.get_tensor_name(i)
            shape = self.decoder_engine.get_tensor_shape(name)
            dtype = self.decoder_engine.get_tensor_dtype(name)
            mode = self.decoder_engine.get_tensor_mode(name)
            
            # Handle dynamic shapes
            if -1 in shape:
                if name == 'point_coords':
                    shape = (1, 1, 2)  # Single point
                elif name == 'point_labels':
                    shape = (1, 1)     # Single label
            
            # Convert TensorRT dtype to numpy
            if dtype == trt.DataType.FLOAT:
                np_dtype = np.float32
            elif dtype == trt.DataType.HALF:
                np_dtype = np.float16
            else:
                np_dtype = np.float32
            
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.decoder_bindings.append(int(device_mem))
            
            if mode == trt.TensorIOMode.INPUT:
                self.decoder_inputs[name] = {
                    'host': host_mem, 'device': device_mem, 'shape': shape
                }
            else:
                self.decoder_outputs[name] = {
                    'host': host_mem, 'device': device_mem, 'shape': shape
                }
    
    def preprocess_image(self, image):
        """Preprocess image for NanoSAM"""
        # Resize to 1024x1024
        image = image.resize((1024, 1024), Image.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image).astype(np.float32)
        
        # Normalize to [0, 1]
        image_np = image_np / 255.0
        
        # Convert HWC to CHW
        image_np = image_np.transpose(2, 0, 1)
        
        # Add batch dimension
        image_np = image_np[np.newaxis, ...]
        
        return image_np
    
    def encode_image(self, image):
        """Encode image using the image encoder"""
        # Preprocess image
        image_array = self.preprocess_image(image)
        
        # Copy input data to host memory
        np.copyto(self.encoder_inputs['image']['host'], image_array.ravel())
        
        # Transfer input data to device
        cuda.memcpy_htod_async(
            self.encoder_inputs['image']['device'], 
            self.encoder_inputs['image']['host'], 
            self.encoder_stream
        )
        
        # Set tensor addresses
        for name, data in self.encoder_inputs.items():
            self.encoder_context.set_tensor_address(name, data['device'])
        for name, data in self.encoder_outputs.items():
            self.encoder_context.set_tensor_address(name, data['device'])
        
        # Run inference
        self.encoder_context.execute_async_v3(stream_handle=self.encoder_stream.handle)
        
        # Transfer output data back to host
        cuda.memcpy_dtoh_async(
            self.encoder_outputs['image_embeddings']['host'], 
            self.encoder_outputs['image_embeddings']['device'], 
            self.encoder_stream
        )
        
        # Synchronize stream
        self.encoder_stream.synchronize()
        
        # Reshape output
        embeddings = self.encoder_outputs['image_embeddings']['host'].reshape(
            self.encoder_outputs['image_embeddings']['shape']
        )
        
        return embeddings
    
    def predict_mask(self, image_embeddings, point_coords, point_labels):
        """Predict mask using the mask decoder"""
        # Prepare inputs
        point_coords = np.array(point_coords, dtype=np.float32).reshape(1, 1, 2)
        point_labels = np.array(point_labels, dtype=np.float32).reshape(1, 1)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)
        
        # Copy input data
        np.copyto(self.decoder_inputs['image_embeddings']['host'], image_embeddings.ravel())
        np.copyto(self.decoder_inputs['point_coords']['host'], point_coords.ravel())
        np.copyto(self.decoder_inputs['point_labels']['host'], point_labels.ravel())
        np.copyto(self.decoder_inputs['mask_input']['host'], mask_input.ravel())
        np.copyto(self.decoder_inputs['has_mask_input']['host'], has_mask_input.ravel())
        
        # Transfer inputs to device
        for name, data in self.decoder_inputs.items():
            cuda.memcpy_htod_async(data['device'], data['host'], self.decoder_stream)
        
        # Set tensor addresses
        for name, data in self.decoder_inputs.items():
            self.decoder_context.set_tensor_address(name, data['device'])
        for name, data in self.decoder_outputs.items():
            self.decoder_context.set_tensor_address(name, data['device'])
        
        # Run inference
        self.decoder_context.execute_async_v3(stream_handle=self.decoder_stream.handle)
        
        # Transfer outputs back
        for name, data in self.decoder_outputs.items():
            cuda.memcpy_dtoh_async(data['host'], data['device'], self.decoder_stream)
        
        # Synchronize
        self.decoder_stream.synchronize()
        
        # Reshape outputs
        iou_predictions = self.decoder_outputs['iou_predictions']['host'].reshape(
            self.decoder_outputs['iou_predictions']['shape']
        )
        low_res_masks = self.decoder_outputs['low_res_masks']['host'].reshape(
            self.decoder_outputs['low_res_masks']['shape']
        )
        
        return low_res_masks, iou_predictions

def main():
    parser = argparse.ArgumentParser(description="Benchmark NanoSAM inference with direct TensorRT.")
    parser.add_argument("--image_encoder_path", type=str, required=True,
                        help="Path to the NanoSAM image encoder .engine file")
    parser.add_argument("--mask_decoder_path", type=str, required=True,
                        help="Path to the NanoSAM mask decoder .engine file")
    parser.add_argument("--img_folder", type=str, required=True,
                        help="Path to the folder containing images for benchmarking")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the benchmark on")
    parser.add_argument("--num_warmup", type=int, default=5,
                        help="Number of warmup runs")
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of runs to average for performance measurement")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save benchmark results as CSV")
    
    args = parser.parse_args()

    # Add timestamp to output CSV
    if args.output_csv:
        now = datetime.now()
        timestamp = now.strftime("%m%dT%H%M") 
        base_name, ext = os.path.splitext(args.output_csv)
        args.output_csv = f"{base_name}_{timestamp}{ext}"

    print(f"🔧 Starting NanoSAM Direct TensorRT Benchmark...")
    print(f"📁 Image Encoder: {args.image_encoder_path}")
    print(f"📁 Mask Decoder: {args.mask_decoder_path}")
    print(f"🖼️  Images: {args.img_folder}")
    print(f"📍 TensorRT version: {trt.__version__}")

    # Check CUDA
    memory_tracking_enabled = torch.cuda.is_available()
    if memory_tracking_enabled:
        print(f"🔥 CUDA available: True")
        baseline_memory = get_gpu_memory_mb()
        print(f"📊 Baseline GPU memory: {baseline_memory:.1f} MB")

    # Get images
    imgs = sorted(glob.glob(args.img_folder))
    if not imgs:
        raise FileNotFoundError(f"No images found with pattern {args.img_folder}")
    print(f"📸 Found {len(imgs)} images")

    # Load predictor
    print(f"🧠 Loading Direct TensorRT predictor...")
    try:
        predictor = DirectTensorRTPredictor(args.image_encoder_path, args.mask_decoder_path)
        
        if memory_tracking_enabled:
            model_memory = get_gpu_memory_mb()
            print(f"📊 GPU memory after model load: {model_memory:.1f} MB (+{model_memory - baseline_memory:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")
        return

    # Prepare for benchmarking
    sample_img_path = imgs[0]
    sample_image = Image.open(sample_img_path).convert("RGB")
    print(f"\n🚀 Starting benchmark using: {os.path.basename(sample_img_path)}")

    # Dummy point for consistent testing
    dummy_point = [sample_image.width // 2, sample_image.height // 2]
    dummy_label = [1]

    # Benchmark variables
    image_encoder_times = []
    full_pipeline_times = []
    peak_memory_usage = []

    total_runs = args.num_warmup + args.num_runs
    with tqdm(total=total_runs, desc="Benchmarking NanoSAM", unit="run") as pbar:
        for i in range(total_runs):
            if memory_tracking_enabled:
                torch.cuda.empty_cache()
                pre_run_memory = get_gpu_memory_mb()

            # Benchmark image encoding
            start_time = time.perf_counter()
            embeddings = predictor.encode_image(sample_image)
            end_time = time.perf_counter()
            encoder_time = (end_time - start_time) * 1000

            if memory_tracking_enabled:
                post_encoder_memory = get_gpu_memory_mb()

            # Benchmark full pipeline
            start_time = time.perf_counter()
            masks, scores = predictor.predict_mask(embeddings, dummy_point, dummy_label)
            end_time = time.perf_counter()
            pipeline_time = (end_time - start_time) * 1000

            if memory_tracking_enabled:
                post_predict_memory = get_gpu_memory_mb()
                run_peak_memory = max(post_encoder_memory, post_predict_memory) - pre_run_memory
            else:
                run_peak_memory = 0

            if i >= args.num_warmup:
                image_encoder_times.append(encoder_time)
                full_pipeline_times.append(pipeline_time)
                peak_memory_usage.append(run_peak_memory)

            pbar.update(1)
            if i == args.num_warmup - 1:
                pbar.write(f"--- Warmup complete, starting timed runs ({args.num_runs} runs) ---")

    # Calculate results
    avg_encoder_time = np.mean(image_encoder_times)
    avg_pipeline_time = np.mean(full_pipeline_times)
    avg_peak_memory = np.mean(peak_memory_usage) if memory_tracking_enabled else 0
    max_peak_memory = np.max(peak_memory_usage) if memory_tracking_enabled else 0

    print("\n--- Benchmark Results ---")
    print(f"Image Encoder Average Time: {avg_encoder_time:.2f} ms")
    print(f"Full Pipeline Average Time: {avg_pipeline_time:.2f} ms")
    if memory_tracking_enabled:
        print(f"Average Peak GPU Memory Usage: {avg_peak_memory:.1f} MB")
        print(f"Maximum Peak GPU Memory Usage: {max_peak_memory:.1f} MB")
    print("-------------------------")

    # Save to CSV
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(args.output_csv, 'w', newline='') as csvfile:
                fieldnames = ['Metric', 'Value', 'Unit']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({'Metric': 'Image Encoder Average Time', 'Value': f'{avg_encoder_time:.2f}', 'Unit': 'ms'})
                writer.writerow({'Metric': 'Full Pipeline Average Time', 'Value': f'{avg_pipeline_time:.2f}', 'Unit': 'ms'})
                
                if memory_tracking_enabled:
                    writer.writerow({'Metric': 'Average Peak GPU Memory Usage', 'Value': f'{avg_peak_memory:.1f}', 'Unit': 'MB'})
                    writer.writerow({'Metric': 'Maximum Peak GPU Memory Usage', 'Value': f'{max_peak_memory:.1f}', 'Unit': 'MB'})
                else:
                    writer.writerow({'Metric': 'Average Peak GPU Memory Usage', 'Value': 'N/A', 'Unit': 'CPU mode'})

            print(f"✅ Results saved to: {args.output_csv}")
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")

if __name__ == "__main__":
    main()
