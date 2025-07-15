"""
python ./fastsam_bench.py \
  --model_path "/home/copter/FastSAM/weights/FastSAM-s.pt" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --imgsz 1024 \
  --iou 0.1,0.3,0.5 \
  --conf 0.2,0.6,0.8 \
  --device cuda \
  --output_csv fastsam_bench_1024.csv

python ./fastsam_bench.py \
  --model_path "/home/copter/FastSAM/weights/FastSAM-s.pt" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --imgsz 512 \
  --iou 0.1,0.3,0.5 \
  --conf 0.2,0.6,0.8 \
  --device cuda \
  --output_csv fastsam_bench_512.csv

python ./fastsam_bench.py \
  --model_path "/home/copter/FastSAM/weights/FastSAM-s.pt" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --imgsz 256 \
  --iou 0.1,0.3,0.5 \
  --conf 0.2,0.6,0.8 \
  --device cuda \
  --output_csv fastsam_bench_256.csv
"""

#!/usr/bin/env python
import argparse
import glob
import time
import csv
import os
import psutil  # ← pip install psutil
from PIL import Image
import torch
from tqdm import tqdm
from ultralytics.nn.tasks import SegmentationModel

# monkey-patch
try:
    # PyTorch ≥2.6
    torch.serialization.add_safe_globals([SegmentationModel])
except AttributeError:
    # older API
    torch.serialization.safe_globals([SegmentationModel])

# (Optional) If you still hit weights_only errors, monkey-patch torch.load to force weights_only=False:
_orig_load = torch.load
def _patched_load(f, map_location='cpu', **kwargs):
    return _orig_load(f, map_location=map_location, weights_only=False, **kwargs)
torch.load = _patched_load

from fastsam import FastSAM
from utils.tools import convert_box_xywh_to_xyxy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FastSAM speed & memory over IOU/CONF thresholds"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="./weights/FastSAM-s.pt",
        help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--img_folder", type=str, required=True,
        help="Folder or glob of input images, e.g. '/data/*.jpg'"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1024,
        help="Resize shorter edge to this size"
    )
    parser.add_argument(
        "--iou", type=str, default="0.5",
        help="Single IOU or comma-sep list, e.g. '0.1,0.3,0.5'"
    )
    parser.add_argument(
        "--conf", type=str, default="0.4",
        help="Single CONF or comma-sep list, e.g. '0.2,0.5,0.8'"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="'cuda', 'cuda:0', or 'cpu'"
    )
    parser.add_argument(
        "--output_csv", type=str, default="bench_results.csv",
        help="Where to save the results"
    )
    parser.add_argument(
        "--retina", action="store_true",
        help="Use high-resolution masks"
    )
    parser.add_argument(
        "--with_contours", action="store_true",
        help="Draw mask contours (no effect on speed/mem)"
    )
    parser.add_argument(
        "--better_quality", action="store_true",
        help="Apply morphologyEx (no effect on speed/mem)"
    )
    return parser.parse_args()


def str2list(s, cast_fn=float):
    # "0.1,0.5" → [0.1, 0.5], or "0.6" → [0.6]
    return [cast_fn(x) for x in s.split(",")]


def main(args):
    print(f"🚀 Starting FastSAM benchmark...")
    print(f"📁 Model path: {args.model_path}")
    print(f"🖼️  Image folder: {args.img_folder}")
    print(f"📏 Image size: {args.imgsz}")
    print(f"🎯 Device: {args.device}")
    
    # Parse thresholds
    iou_list  = str2list(args.iou,  float)
    conf_list = str2list(args.conf, float)
    print(f"📊 IOU thresholds: {iou_list}")
    print(f"📊 Confidence thresholds: {conf_list}")

    # Check device availability
    device = torch.device(args.device)
    print(f"✅ Device set to: {device}")
    
    if device.type == "cuda":
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
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

    # Model init
    print(f"🤖 Loading FastSAM model...")
    print(f"   Model path: {args.model_path}")
    
    try:
        model = FastSAM(args.model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"📱 Moving model to device: {device}")
    model.to(device)
    print(f"✅ Model ready on {device}")

    # Prepare CSV
    full_path = "output/" + args.output_csv
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    print(f"📝 Output CSV: {full_path}")
    
    total_combinations = len(iou_list) * len(conf_list) * len(imgs)
    print(f"🎯 Total benchmarks to run: {total_combinations}")

    # Additional settings info
    settings = []
    if args.retina:
        settings.append("retina_masks=True")
    if args.with_contours:
        settings.append("with_contours=True")
    if args.better_quality:
        settings.append("better_quality=True")
    
    if settings:
        print(f"⚙️  Additional settings: {', '.join(settings)}")

    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "device", "img", "imgsz",
            "iou", "conf", "time_ms", "gpu_mem_mb", "cpu_mem_mb"
        ])

        # Create combinations for progress bar
        combinations = [
            (iou, conf, img_path)
            for iou in iou_list
            for conf in conf_list
            for img_path in imgs
        ]
        
        # Initialize progress bar
        progress_bar = tqdm(combinations, desc="🚀 Processing", ncols=100)
        
        for iou, conf, img_path in progress_bar:
            # Update progress bar description
            progress_bar.set_postfix({
                "IOU": f"{iou:.1f}",
                "Conf": f"{conf:.1f}",
                "Image": os.path.basename(img_path)[:15]
            })
            
            # Reset GPU counters
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            # Record CPU RSS before
            proc = psutil.Process(os.getpid())
            rss_before = proc.memory_info().rss / 1024**2  # MB

            # Load image
            img = Image.open(img_path).convert("RGB")
            original_size = img.size

            # Run FastSAM inference
            t0 = time.perf_counter()
            
            try:
                everything = model(
                    img,
                    device=device,
                    imgsz=args.imgsz,
                    conf=conf,
                    iou=iou,
                    retina_masks=args.retina
                )
                
                # Force sync & time
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                
                # Try to get mask count if possible
                try:
                    if hasattr(everything, 'masks') and everything.masks is not None:
                        mask_count = len(everything.masks)
                    elif hasattr(everything, 'boxes') and everything.boxes is not None:
                        mask_count = len(everything.boxes)
                    else:
                        mask_count = "N/A"
                except:
                    mask_count = "N/A"
                
            except Exception as e:
                tqdm.write(f"❌ Error processing {os.path.basename(img_path)}: {e}")
                continue

            # Measure performance
            delta_cpu = proc.memory_info().rss / 1024**2 - rss_before
            peak_gpu = (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else 0.0
            )
            elapsed_ms = (t1 - t0) * 1e3

            # Update progress bar with current metrics
            progress_bar.set_postfix({
                "IOU": f"{iou:.1f}",
                "Conf": f"{conf:.1f}",
                "Time": f"{elapsed_ms:.0f}ms",
                "GPU": f"{peak_gpu:.0f}MB",
                "Masks": str(mask_count)
            })

            writer.writerow([
                os.path.basename(args.model_path),
                args.device,
                os.path.basename(img_path),
                args.imgsz,
                iou,
                conf,
                f"{elapsed_ms:.1f}",
                f"{peak_gpu:.1f}",
                f"{delta_cpu:.1f}",
            ])

    print(f"\n🎉 FastSAM benchmark complete!")
    print(f"📊 Results saved to: {full_path}")


if __name__=="__main__":
    args = parse_args()
    main(args)