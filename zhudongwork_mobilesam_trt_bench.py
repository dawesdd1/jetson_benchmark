"""
Usage:

conda activate NanoSAM
export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

### Zhudongwork Benches
# FP16
python /home/copter/jetson_benchmark/zhudongwork_mobilesam_trt_bench.py \
  --image_encoder_path "/home/copter/engine_models/zhudongwork_mobile_sam_encoder_fp16_trt1030.engine" \
  --mask_decoder_path "/home/copter/engine_models/zhudongwork_mobile_sam_decoder_fp16_trt1030.engine" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --device cuda \
  --num_runs 50 \
  --output_csv "/home/copter/jetson_benchmark/output/zhudongwork_mobile_sam_bench_fp16.csv"

#FP32
python /home/copter/jetson_benchmark/zhudongwork_mobilesam_trt_bench.py \
  --image_encoder_path "/home/copter/engine_models/mobile_sam_encoder_fp32.engine" \
  --mask_decoder_path "/home/copter/engine_models/mobile_sam_mask_decoder_fp32.engine" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --device cuda \
  --num_runs 50 \
  --output_csv "/home/copter/jetson_benchmark/output/nanosam_bench_fp32.csv"


"""

import argparse
import os
import glob
import time
import torch
import numpy as np
import csv
from PIL import Image
from nanosam.utils.predictor import Predictor
from tqdm import tqdm 
from datetime import datetime

def str2list(s, type_func):
    return [type_func(item) for item in s.split(',')]

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert bytes to MB
    return 0

def main():
    parser = argparse.ArgumentParser(description="Benchmark NanoSAM inference.")
    parser.add_argument("--image_encoder_path", type=str, required=True,
                        help="Path to the NanoSAM image encoder .engine file (e.g., data/resnet18_image_encoder.engine).")
    parser.add_argument("--mask_decoder_path", type=str, required=True,
                        help="Path to the NanoSAM mask decoder .engine file (e.g., data/mobile_sam_mask_decoder.engine).")
    parser.add_argument("--img_folder", type=str, required=True,
                        help="Path to the folder containing images for benchmarking (e.g., './data/*.jpg').")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the benchmark on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--num_warmup", type=int, default=5,
                        help="Number of warmup runs before measuring performance.")
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of runs to average for performance measurement.")
    parser.add_argument("--output_csv", type=str, default=None,
                    help="Optional: Path to save benchmark results as a CSV file (e.g., 'results/nanosam_bench.csv').")
    
    args = parser.parse_args()

    if args.output_csv:
        # Get current datetime
        now = datetime.now()
        # Format as MMDDTHHMM
        timestamp = now.strftime("%m%dT%H%M") 

        # Split the base path and the file extension
        base_name, ext = os.path.splitext(args.output_csv)
        
        # Insert the timestamp before the extension
        args.output_csv = f"{base_name}_{timestamp}{ext}"

    print(f"🔧 Starting NanoSAM benchmark...")
    print(f"📁 Image Encoder path: {args.image_encoder_path}")
    print(f"📁 Mask Decoder path: {args.mask_decoder_path}")
    print(f"🖼️  Image folder: {args.img_folder}")
    print(f"🎯 Device: {args.device}")

    # Check device availability
    device = torch.device(args.device)
    print(f"✅ Device set to: {device}")
    
    # Initialize memory tracking variables
    memory_tracking_enabled = False
    
    if device.type == "cuda":
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            memory_tracking_enabled = True
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
            
            # Also check nvidia-smi style info if available
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    used_mb, total_mb = map(int, result.stdout.strip().split(', '))
                    print(f"🔥 nvidia-smi: {used_mb}MB used / {total_mb}MB total ({(total_mb-used_mb)}MB free)")
            except:
                pass
        else:
            print("❌ CUDA is not available, falling back to CPU.")
            args.device = "cpu"
            device = torch.device("cpu")

    # Gather images first
    print(f"🔍 Searching for images with pattern: {args.img_folder}")
    imgs = sorted(glob.glob(args.img_folder))
    if not imgs:
        raise FileNotFoundError(f"No images found with pattern {args.img_folder}")
    print(f"📸 Found {len(imgs)} images")
    for i, img in enumerate(imgs[:3]):  # Show first 3
        print(f"   {i+1}. {os.path.basename(img)}")
    if len(imgs) > 3:
        print(f"   ... and {len(imgs) - 3} more")

    # --- Load Model --------------------------------------
    print(f"🧠 Loading NanoSAM model...")
    try:
        # Clear GPU cache before loading model
        if memory_tracking_enabled:
            torch.cuda.empty_cache()
            baseline_memory = get_gpu_memory_mb()
            print(f"📊 Baseline GPU memory: {baseline_memory:.1f} MB")
        
        predictor = Predictor(
            image_encoder_engine=args.image_encoder_path,
            mask_decoder_engine=args.mask_decoder_path
        )
        
        if memory_tracking_enabled:
            model_memory = get_gpu_memory_mb()
            print(f"📊 GPU memory after model load: {model_memory:.1f} MB (+{model_memory - baseline_memory:.1f} MB)")
        
        print("✅ NanoSAM model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading NanoSAM model: {e}")
        return

    # --- Benchmarking --------------------------------------
    image_encoder_times = []
    full_pipeline_times = []
    peak_memory_usage = []
    
    # Use a single sample image for consistent benchmarking, or iterate through a few
    # For a fair benchmark, it's often good to use a representative image size/type.
    # We'll just use the first image found for simplicity.
    sample_img_path = imgs[0]
    sample_image = Image.open(sample_img_path).convert("RGB") # Ensure RGB
    print(f"\n🚀 Starting benchmark runs using image: {os.path.basename(sample_img_path)}")

    # Define a dummy point for prediction for the full pipeline
    # In a real scenario, these would come from a detection model or user input.
    # For benchmarking, a fixed point is sufficient.
    dummy_point = np.array([[sample_image.width // 2, sample_image.height // 2]])
    dummy_label = np.array([1]) # 1 typically means foreground

    total_runs = args.num_warmup + args.num_runs
    with tqdm(total=total_runs, desc="Benchmarking NanoSAM", unit="run") as pbar:
        for i in range(total_runs):
            # Clear GPU cache before each run for consistent memory measurements
            if memory_tracking_enabled:
                torch.cuda.empty_cache()
                pre_run_memory = get_gpu_memory_mb()
            
            # Benchmarking Image Encoder
            start_time_image_encoder = time.perf_counter()
            predictor.set_image(sample_image)
            end_time_image_encoder = time.perf_counter()
            
            # Check memory after image encoding
            if memory_tracking_enabled:
                post_encoder_memory = get_gpu_memory_mb()
            
            # Benchmarking Full Pipeline
            start_time_full_pipeline = time.perf_counter()
            _masks, _scores, _logits = predictor.predict(dummy_point, dummy_label)
            end_time_full_pipeline = time.perf_counter()

            # Check peak memory usage
            if memory_tracking_enabled:
                post_predict_memory = get_gpu_memory_mb()
                run_peak_memory = max(post_encoder_memory, post_predict_memory) - pre_run_memory
            else:
                run_peak_memory = 0

            if i >= args.num_warmup:
                image_encoder_times.append((end_time_image_encoder - start_time_image_encoder) * 1000) # in ms
                full_pipeline_times.append((end_time_full_pipeline - start_time_full_pipeline) * 1000) # in ms
                peak_memory_usage.append(run_peak_memory)
            
            pbar.update(1)
            if i == args.num_warmup - 1:
                pbar.write(f"--- Warmup complete, starting timed runs ({args.num_runs} runs) ---")
    
    avg_image_encoder_time = np.mean(image_encoder_times)
    avg_full_pipeline_time = np.mean(full_pipeline_times)
    avg_peak_memory = np.mean(peak_memory_usage) if memory_tracking_enabled else 0
    max_peak_memory = np.max(peak_memory_usage) if memory_tracking_enabled else 0

    print("\n--- Benchmark Results ---")
    print(f"Image Encoder Average Time: {avg_image_encoder_time:.2f} ms")
    print(f"Full Pipeline Average Time: {avg_full_pipeline_time:.2f} ms")
    if memory_tracking_enabled:
        print(f"Average Peak GPU Memory Usage: {avg_peak_memory:.1f} MB")
        print(f"Maximum Peak GPU Memory Usage: {max_peak_memory:.1f} MB")
    else:
        print("GPU Memory tracking: Not available (CPU mode)")
    print("-------------------------")

    # --- Save results to CSV (if output_csv flag is provided) ---
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        try:
            with open(args.output_csv, 'w', newline='') as csvfile:
                fieldnames = ['Metric', 'Value', 'Unit']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerow({'Metric': 'Image Encoder Average Time', 'Value': f'{avg_image_encoder_time:.2f}', 'Unit': 'ms'})
                writer.writerow({'Metric': 'Full Pipeline Average Time', 'Value': f'{avg_full_pipeline_time:.2f}', 'Unit': 'ms'})
                
                if memory_tracking_enabled:
                    writer.writerow({'Metric': 'Average Peak GPU Memory Usage', 'Value': f'{avg_peak_memory:.1f}', 'Unit': 'MB'})
                    writer.writerow({'Metric': 'Maximum Peak GPU Memory Usage', 'Value': f'{max_peak_memory:.1f}', 'Unit': 'MB'})
                else:
                    writer.writerow({'Metric': 'Average Peak GPU Memory Usage', 'Value': 'N/A', 'Unit': 'CPU mode'})
                    writer.writerow({'Metric': 'Maximum Peak GPU Memory Usage', 'Value': 'N/A', 'Unit': 'CPU mode'})
                    
            print(f"✅ Benchmark results saved to: {args.output_csv}")
        except Exception as e:
            print(f"❌ Error saving results to CSV: {e}")

if __name__ == "__main__":
    main()