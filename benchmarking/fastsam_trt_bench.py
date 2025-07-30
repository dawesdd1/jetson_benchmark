"""
Usage:

conda activate FastSAM_py310
export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH # If needed for TensorRT

# Example for benchmarking FastSAM-s.engine
python ./fastsam_trt_bench.py \
  --engine_path "/home/copter/engine_models/FastSAM-s_fp16.engine" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --num_runs 50 \
  --output_csv "/home/copter/jetson_benchmark/output/fastsam-s_fp16_bench_s.csv" \
  --retina_masks \
  --conf 0.1 \
  --iou 0.25 \
  --agnostic_nms

# Example for benchmarking FastSAM-x.engine
python ./fastsam_trt_bench.py \
  --engine_path "/home/copter/engine_models/FastSAM-x_fp16.engine \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --num_runs 50 \
  --output_csv "results/fastsam_bench_x.csv" \
  --retina_masks \
  --conf 0.1 \
  --iou 0.25 \
  --agnostic_nms
"""

import argparse
import os
import glob
import time
import cv2
import numpy as np
import csv
from datetime import datetime
from tqdm import tqdm
import torch

# Import necessary components from fastsam_trt_infer.py and local utilities
# Ensure common.py and fastsam_utils.py are correctly placed alongside this script,
# or that their parent directory is added to your PYTHONPATH.
try:
    import common # This expects a local common.py file
    from fastsam_utils import overlay # Using fastsam_utils.py as per your setup
    from ultralytics.engine.results import Results
    from ultralytics.utils import ops
    import tensorrt as trt
    from cuda import cudart # Still needed for cuda_call in common.py
    from random import randint
except ImportError as e:
    print(f"Error importing FastSAM dependencies: {e}")
    print("Please ensure 'common.py' and 'fastsam_utils.py' are correctly placed alongside this script,")
    print("or that their parent directory is added to your PYTHONPATH.")
    print("Also, ensure 'ultralytics', 'tensorrt', and 'cuda-python' are installed in your active environment.")
    exit(1)


class TensorRTInfer:
    """
    Implements inference for TensorRT engine.
    Refactored to use common.allocate_buffers and common.do_inference_v2 for modern TensorRT API.
    """

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Use common.allocate_buffers to handle input/output allocations
        # For benchmarking, we assume profile_idx=0 for dynamic shapes if any.
        profile_idx = 0 if self.engine.num_optimization_profiles > 0 else None
        
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine, profile_idx)

        # Map tensor names to their HostDeviceMem objects for easier access
        self.input_hdm_map = {}
        self.output_hdm_map = {}
        
        input_hdm_idx = 0
        output_hdm_idx = 0
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                # This is an input tensor
                current_hdm = self.inputs[input_hdm_idx]
                self.input_hdm_map[tensor_name] = current_hdm

                # Handle dynamic shapes for this input tensor
                # Check if any dimension is dynamic (indicated by -1)
                if -1 in self.engine.get_tensor_shape(tensor_name):
                    # Get the optimal shape from the first optimization profile (index 0)
                    optimal_shape = list(self.engine.get_tensor_profile_shape(tensor_name, 0)[1])
                    # If the batch dimension is dynamic (usually the first dimension), set it to 1
                    if optimal_shape[0] == -1:
                        optimal_shape[0] = 1
                    # Set the input shape for the execution context
                    self.context.set_input_shape(tensor_name, optimal_shape)
                    # Update the host buffer's shape to reflect the set input shape
                    # This is crucial for `input_hdm.host = batch` to work correctly.
                    current_hdm.host = np.zeros(optimal_shape, dtype=current_hdm.host.dtype)

                input_hdm_idx += 1
            else:
                # This is an output tensor
                current_hdm = self.outputs[output_hdm_idx]
                self.output_hdm_map[tensor_name] = current_hdm
                output_hdm_idx += 1
        
        # Store the actual input dimensions (H, W) from the engine's first input tensor
        # Assuming the input shape is NCHW or CHW, and H, W are the last two dimensions.
        if self.inputs:
            input_tensor_name = self.engine.get_tensor_name(0) # Get the name of the first input tensor
            input_shape = self.context.get_tensor_shape(input_tensor_name) # Get the shape from the context
            # If dynamic batching, the actual batch size for inference is 1, so we take the optimal shape
            if -1 in input_shape:
                 input_shape = self.engine.get_tensor_profile_shape(input_tensor_name, 0)[1] # Optimal shape
            
            # Extract H and W, assuming they are the last two dimensions (e.g., NCHW -> [N, C, H, W])
            self.input_height = input_shape[-2]
            self.input_width = input_shape[-1]
            print(f"Detected engine input dimensions: {self.input_width}x{self.input_height}")
        else:
            self.input_height = 1024 # Fallback, though an engine should have inputs
            self.input_width = 1024
            print(f"Warning: Could not determine engine input dimensions. Using default {self.input_width}x{self.input_height}.")

        # For explicit batch, batch_size is part of the input shape.
        # Assuming the first input's first dimension is the batch size for inference.
        if self.inputs:
            # Get batch size from the actual allocated host buffer shape after dynamic shape adjustments
            self.batch_size = self.inputs[0].host.shape[0] 
        else:
            self.batch_size = 1 # Default if no inputs (unlikely for inference)

        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        # self.allocations is now implicitly managed by common.allocate_buffers via HostDeviceMem objects
        assert len(self.bindings) == self.engine.num_io_tensors # Ensure all tensors have a binding

        # Debug: Print tensor information
        print(f"Engine has {self.engine.num_io_tensors} tensors:")
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_mode = "INPUT" if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT else "OUTPUT"
            print(f"  {i}: {tensor_name} - {tensor_mode} - Shape: {tensor_shape}")

    def infer(self, batch):
        # Use the first input tensor name to get its HostDeviceMem object
        # Assuming the model has at least one input.
        first_input_name = next(iter(self.input_hdm_map)) # Get the first key (tensor name)
        input_hdm = self.input_hdm_map[first_input_name]
        
        # Copy data to the host buffer of the HostDeviceMem object
        input_hdm.host = batch 

        # Set tensor addresses for all tensors before inference (required for execute_async_v3)
        self.context.set_tensor_address(first_input_name, input_hdm.device)
        
        # Set output tensor addresses
        for tensor_name, output_hdm in self.output_hdm_map.items():
            self.context.set_tensor_address(tensor_name, output_hdm.device)

        # Execute inference using common.do_inference_v2
        # This function handles host-to-device data transfer, execution, and device-to-host transfer.
        results = common.do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        
        return results

    def __del__(self):
        # Ensure resources are freed when the object is deleted
        if hasattr(self, 'inputs') and hasattr(self, 'outputs') and hasattr(self, 'stream'):
            common.free_buffers(self.inputs, self.outputs, self.stream)


def postprocess(preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=True):
    """
    Post-processing function from fastsam_trt_infer.py
    """
    p = ops.non_max_suppression(preds[0],
                                conf,
                                iou,
                                agnostic_nms,
                                max_det=100,
                                nc=1)
    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        img_path = "bench_image" # Dummy path for benchmarking
        if not len(pred):
            results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
            continue
        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(orig_img=orig_img, path=img_path, names="1213", boxes=pred[:, :6], masks=masks))
    return results


def pre_processing(img_origin, target_height, target_width):
    """
    Pre-processing function from fastsam_trt_infer.py, modified to handle
    resizing and padding to specific target_height and target_width.
    """
    h_orig, w_orig = img_origin.shape[:2]

    # Calculate scaling factors to maintain aspect ratio
    scale = min(target_width / w_orig, target_height / h_orig)

    # Calculate new dimensions after scaling
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)

    # Resize image
    resized_img_rgb = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (new_w, new_h))

    # Create a blank canvas with target dimensions and pad the resized image
    padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate padding offsets to center the image
    pad_x = (target_width - new_w) // 2
    pad_y = (target_height - new_h) // 2

    padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w, :] = resized_img_rgb

    # Convert to float32 and transpose for NCHW format (Batch, Channels, Height, Width)
    rgb = np.array([padded_img], dtype=np.float32) / 255.0
    rgb = np.transpose(rgb, (0, 3, 1, 2))
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    return rgb


class FastSamPredictor(object):
    """
    Adapted FastSam class from fastsam_trt_infer.py for benchmarking purposes.
    It focuses on the core inference logic without saving images.
    """
    def __init__(self, model_weights='fast_sam_1024.trt', max_size=1024):
        # max_size is now a placeholder; actual size comes from the engine
        self.model = TensorRTInfer(model_weights)
        # Overwrite imgsz with the actual input dimensions from the loaded engine
        self.imgsz = (self.model.input_height, self.model.input_width)
        print(f"FastSamPredictor using image size from engine: {self.imgsz[1]}x{self.imgsz[0]} (Width x Height)")


    def predict(self, bgr_img, retina_masks, conf, iou, agnostic_nms):
        """
        Performs a single inference run.
        Returns the raw masks and scores, similar to NanoSAM's predict.
        """
        # Pass the dynamically determined imgsz (height, width) to pre_processing
        inp = pre_processing(bgr_img, self.imgsz[0], self.imgsz[1]) # Pass height and width
        preds = self.model.infer(inp)

        # Debug: Print the structure of preds to understand the output format
        print(f"Debug: Number of outputs: {len(preds)}")
        for i, pred in enumerate(preds):
            print(f"  Output {i}: shape {pred.shape}, dtype {pred.dtype}")

        # Reconstruct preds into a format suitable for postprocess
        # Based on the engine tensor info:
        # output0: (1, 37, 8400) -> detection output
        # output1: (1, 32, 160, 160) -> segmentation masks
        
        try:
            # Reshape the flattened outputs to their expected dimensions
            # output0: (1, 37, 8400) = 310800 elements
            # output1: (1, 32, 160, 160) = 819200 elements
            
            # First output: detection results
            if len(preds) >= 1 and preds[0].size == 310800:
                detection_output = preds[0].reshape(1, 37, 8400)
                data_0 = torch.from_numpy(detection_output)
            else:
                print(f"Warning: Unexpected detection output size: {preds[0].size}, expected 310800")
                data_0 = torch.from_numpy(preds[0])
            
            # Second output: segmentation masks  
            if len(preds) >= 2 and preds[1].size == 819200:
                mask_output = preds[1].reshape(1, 32, 160, 160)
                data_1 = torch.from_numpy(mask_output)
            else:
                print(f"Warning: Unexpected mask output size: {preds[1].size}, expected 819200")
                data_1 = torch.from_numpy(preds[1])
                
            preds_for_postprocess = [data_0, data_1]
            
            print(f"Debug: Reshaped data_0 shape: {data_0.shape}")
            print(f"Debug: Reshaped data_1 shape: {data_1.shape}")
            
            results = postprocess(preds_for_postprocess, inp, bgr_img, retina_masks, conf, iou, agnostic_nms)
            
            # For benchmarking, we just need to ensure the operation completes and potentially get mask data
            if results and results[0].masks is not None:
                return results[0].masks.data, results[0].boxes.conf, None # Return masks, scores, and None for logits (not directly available/needed for FastSAM bench)
            return None, None, None
            
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            print("Falling back to simple benchmark completion")
            # For benchmarking purposes, just return dummy values to measure inference time
            return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Benchmark FastSAM TensorRT inference.")
    parser.add_argument("--engine_path", type=str, required=True,
                        help="Path to the FastSAM TensorRT .engine file (e.g., FastSAM-s.engine).")
    parser.add_argument("--img_folder", type=str, required=True,
                        help="Path to the folder containing images for benchmarking (e.g., './data/*.jpg').")
    parser.add_argument("--num_warmup", type=int, default=5,
                        help="Number of warmup runs before measuring performance.")
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of runs to average for performance measurement.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional: Path to save benchmark results as a CSV file (e.g., 'results/fastsam_bench.csv').")
    parser.add_argument("--retina_masks", action="store_true",
                        help="Use retina masks for post-processing.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for object detection.")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for non-maximum suppression.")
    parser.add_argument("--agnostic_nms", action="store_true",
                        help="Perform class-agnostic NMS.")
    
    args = parser.parse_args()

    # Add timestamp to output CSV path
    if args.output_csv:
        now = datetime.now()
        timestamp = now.strftime("%m%dT%H%M") 
        base_name, ext = os.path.splitext(args.output_csv)
        args.output_csv = f"{base_name}_{timestamp}{ext}"

    print(f"🔧 Starting FastSAM benchmark...")
    print(f"📁 Engine path: {args.engine_path}")
    print(f"🖼️  Image folder: {args.img_folder}")
    print(f"⚙️  Retina Masks: {args.retina_masks}")
    print(f"⚙️  Confidence: {args.conf}")
    print(f"⚙️  IoU: {args.iou}")
    print(f"⚙️  Agnostic NMS: {args.agnostic_nms}")

    # Check device availability (FastSAM TRT engine is typically CUDA-only)
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. FastSAM TensorRT models require CUDA.")
        exit(1)
    
    gpu_props = torch.cuda.get_device_properties(0)
    total_memory = gpu_props.total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
    free_memory = total_memory - reserved_memory
    
    print(f"🔥 GPU: {gpu_props.name}")
    print(f"🔥 Total memory: {total_memory:.1f} GB")
    print(f"🔥 Allocated: {allocated_memory:.1f} GB")
    print(f"🔥 Reserved: {reserved_memory:.1f} GB") 
    print(f"🔥 Free: {free_memory:.1f} GB")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used_mb, total_mb = map(int, result.stdout.strip().split(', '))
            print(f"🔥 nvidia-smi: {used_mb}MB used / {total_mb}MB total ({(total_mb-used_mb)}MB free)")
    except Exception as e:
        print(f"Warning: Could not run nvidia-smi ({e}). Memory details might be incomplete.")

    # Gather images
    print(f"🔍 Searching for images with pattern: {args.img_folder}")
    img_paths = sorted(glob.glob(args.img_folder))
    if not img_paths:
        raise FileNotFoundError(f"No images found with pattern {args.img_folder}")
    print(f"📸 Found {len(img_paths)} images")
    for i, img_path in enumerate(img_paths[:3]):
        print(f"   {i+1}. {os.path.basename(img_path)}")
    if len(img_paths) > 3:
        print(f"   ... and {len(img_paths) - 3} more")

    # Load Model
    print(f"🧠 Loading FastSAM model from {args.engine_path}...")
    try:
        predictor = FastSamPredictor(model_weights=args.engine_path)
        print("✅ FastSAM model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading FastSAM model: {e}")
        return

    # Benchmarking
    full_pipeline_times = []
    
    # Use a single sample image for consistent benchmarking
    sample_img_path = img_paths[0]
    sample_image_bgr = cv2.imread(sample_img_path)
    if sample_image_bgr is None:
        print(f"❌ Error: Could not load sample image at {sample_img_path}. Please check the path and image integrity.")
        return

    print(f"\n🚀 Starting benchmark runs using image: {os.path.basename(sample_img_path)}")

    total_runs = args.num_warmup + args.num_runs
    with tqdm(total=total_runs, desc="Benchmarking FastSAM", unit="run") as pbar:
        for i in range(total_runs):
            start_time_full_pipeline = time.perf_counter()
            _masks, _scores, _logits = predictor.predict(
                sample_image_bgr, 
                args.retina_masks, 
                args.conf, 
                args.iou, 
                args.agnostic_nms
            )
            end_time_full_pipeline = time.perf_counter()

            if i >= args.num_warmup:
                full_pipeline_times.append((end_time_full_pipeline - start_time_full_pipeline) * 1000) # in ms
            
            pbar.update(1)
            if i == args.num_warmup - 1:
                pbar.write(f"--- Warmup complete, starting timed runs ({args.num_runs} runs) ---")
    
    avg_full_pipeline_time = np.mean(full_pipeline_times)

    print("\n--- Benchmark Results ---")
    print(f"Full Pipeline Average Time: {avg_full_pipeline_time:.2f} ms")
    print("-------------------------")

    # Save results to CSV (if output_csv flag is provided)
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        try:
            with open(args.output_csv, 'w', newline='') as csvfile:
                fieldnames = ['Metric', 'Time (ms)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerow({'Metric': 'Full Pipeline', 'Time (ms)': f'{avg_full_pipeline_time:.2f}'})
            print(f"✅ Benchmark results saved to: {args.output_csv}")
        except Exception as e:
            print(f"❌ Error saving results to CSV: {e}")

if __name__ == "__main__":
    main()