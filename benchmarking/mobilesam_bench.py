"""
python ./mobilesam_bench.py \
  --model_path "/home/copter/jetson_benchmark/weights/mobilesam/mobile_sam.pt" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --imgsz 1024 \
  --iou 0.1,0.3,0.5 \
  --conf 0.2,0.6,0.8 \
  --device cuda \
  --output_csv mobilesam_bench_1024.csv

python ./mobilesam_bench.py \
  --model_path "/home/copter/jetson_benchmark/weights/mobilesam/mobile_sam.pt" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --imgsz 512 \
  --iou 0.1,0.3,0.5 \
  --conf 0.2,0.6,0.8 \
  --device cuda \
  --output_csv mobilesam_bench_512.csv

python ./mobilesam_bench.py \
  --model_path "/home/copter/jetson_benchmark/weights/mobilesam/mobile_sam.pt" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --imgsz 256 \
  --iou 0.1,0.3,0.5 \
  --conf 0.2,0.6,0.8 \
  --device cuda \
  --output_csv mobilesam_bench_256.csv
"""

#!/usr/bin/env python
import argparse
import glob
import time
import csv
import os
import psutil
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Mobile SAM speed & memory over IOU/CONF thresholds"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--model_type", type=str, default="vit_t",
        help="Which MobileSAM variant (e.g. 'vit_t', 'vit_l', ...)"
    )
    parser.add_argument(
        "--img_folder", type=str, required=True,
        help="Folder or glob of input images, e.g. '/data/*.jpg'"
    )
    parser.add_argument(
        "--imgsz", type=int, default=0,
        help="Resize shorter edge to this size (0 to disable)"
    )
    parser.add_argument(
        "--iou", type=str, default="0.86",
        help="Comma-separated list of pred_iou_thresh to test, e.g. '0.5,0.75,0.9'"
    )
    parser.add_argument(
        "--conf", type=str, default="0.92",
        help="Comma-separated list of stability_score_thresh to test, e.g. '0.8,0.9,0.95'"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="'cuda', 'cuda:0', or 'cpu'"
    )
    parser.add_argument(
        "--output_csv", type=str, default="mobile_sam_bench.csv",
        help="Where to save the results"
    )
    return parser.parse_args()


def str2list(s, cast_fn=float):
    return [cast_fn(x) for x in s.split(",")]


def main(args):
    # --- Device & Path Checks -----------------------------

    print(f"🔧 Starting Mobile SAM benchmark...")
    print(f"📁 Model path: {args.model_path}")
    print(f"🖼️  Image folder: {args.img_folder}")
    print(f"🎯 Device: {args.device}")
    
    iou_list = str2list(args.iou, float)
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

    # --- Load Model --------------------------------------

    print(f"🤖 Loading Mobile SAM model...")
    print(f"   Model type: {args.model_type}")
    print(f"   Checkpoint: {args.model_path}")
    
    try:
        sam = sam_model_registry[args.model_type](checkpoint=args.model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"📱 Moving model to device: {device}")
    sam.to(device=device)
    sam.eval()
    print(f"✅ Model ready on {device}")

    # Prepare CSV
    full_path = "output/" + args.output_csv
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    print(f"📝 Output CSV: {full_path}")
    
    total_combinations = len(iou_list) * len(conf_list) * len(imgs)
    print(f"🎯 Total benchmarks to run: {total_combinations}")

    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "model_type", "device", "img", "imgsz",
            "pred_iou_thresh", "stability_score_thresh",
            "time_ms", "gpu_mem_mb", "cpu_mem_mb"
        ])

        # Create combinations for progress bar
        combinations = [
            (pred_iou, stability_score, img_path)
            for pred_iou in iou_list
            for stability_score in conf_list
            for img_path in imgs
        ]
        
        # Initialize progress bar
        progress_bar = tqdm(combinations, desc="🎯 Processing", ncols=100)
        
        current_iou = None
        current_conf = None
        mask_generator = None
        
        # --- Run Benchmark ---------------------------------------

        for pred_iou, stability_score, img_path in progress_bar:
            # Create new mask generator only when thresholds change
            if current_iou != pred_iou or current_conf != stability_score:
                current_iou = pred_iou
                current_conf = stability_score
                mask_generator = SamAutomaticMaskGenerator(model=sam)
                progress_bar.set_postfix({
                    "IOU": f"{pred_iou:.1f}",
                    "Conf": f"{stability_score:.1f}",
                    "Image": os.path.basename(img_path)[:15]
                })
            
            # Reset GPU counters
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            proc = psutil.Process(os.getpid())
            rss_before = proc.memory_info().rss / 1024**2

            # Load and optionally resize
            img = Image.open(img_path).convert("RGB")
            original_size = img.size
            
            if args.imgsz > 0:
                w, h = img.size
                if h < w:
                    new_h = args.imgsz
                    new_w = int(w * args.imgsz / h)
                else:
                    new_w = args.imgsz
                    new_h = int(h * args.imgsz / w)
                img = img.resize((new_w, new_h))
            
            img_np = np.array(img)

            # Run mask generation
            t0 = time.perf_counter()
            
            try:
                masks = mask_generator.generate(img_np)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                
            except Exception as e:
                tqdm.write(f"❌ Error processing {os.path.basename(img_path)}: {e}")
                continue

            # Measure performance
            elapsed_ms = (t1 - t0) * 1e3
            peak_gpu = (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else 0.0
            )
            delta_cpu = proc.memory_info().rss / 1024**2 - rss_before

            # Update progress bar with current metrics
            progress_bar.set_postfix({
                "IOU": f"{pred_iou:.1f}",
                "Conf": f"{stability_score:.1f}",
                "Time": f"{elapsed_ms:.0f}ms",
                "GPU": f"{peak_gpu:.0f}MB",
                "Masks": len(masks)
            })

            writer.writerow([
                os.path.basename(args.model_path),
                args.model_type,
                args.device,
                os.path.basename(img_path),
                args.imgsz,
                pred_iou,
                stability_score,
                f"{elapsed_ms:.1f}",
                f"{peak_gpu:.1f}",
                f"{delta_cpu:.1f}",
            ])

    print(f"\n🎉 Mobile SAM benchmark complete!")
    print(f"📊 Results saved to: {full_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)