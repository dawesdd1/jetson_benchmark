"""
python ./bench.py \
  --model_path "/home/copter/FastSAM/weights/FastSAM-s.pt" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --imgsz 1024 \
  --iou 0.1,0.3,0.5 \
  --conf 0.2,0.6,0.8 \
  --device cuda \
  --output_csv drone_bench.csv
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
    # “0.1,0.5” → [0.1, 0.5], or “0.6” → [0.6]
    return [cast_fn(x) for x in s.split(",")]


def main(args):
    # Parse thresholds
    iou_list  = str2list(args.iou,  float)
    conf_list = str2list(args.conf, float)

    # Model init
    device = torch.device(args.device)
    model = FastSAM(args.model_path)
    model.to(device)

    # Gather images
    imgs = sorted(glob.glob(args.img_folder))
    if not imgs:
        raise FileNotFoundError(f"No images found with pattern {args.img_folder}")

    # Prepare CSV
    full_path = "output/" + args.output_csv
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "device", "img", "imgsz",
            "iou", "conf", "time_ms", "gpu_mem_mb", "cpu_mem_mb"
        ])

        for iou in iou_list:
            for conf in conf_list:
                for img_path in imgs:
                    # reset GPU counters
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)

                    # record CPU RSS before
                    proc = psutil.Process(os.getpid())
                    rss_before = proc.memory_info().rss / 1024**2  # MB

                    # load & run
                    img = Image.open(img_path).convert("RGB")
                    t0 = time.perf_counter()
                    everything = model(
                        img,
                        device=device,
                        imgsz=args.imgsz,
                        conf=conf,
                        iou=iou,
                        retina_masks=args.retina
                    )
                    # force sync & time
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t1 = time.perf_counter()

                    # measure
                    delta_cpu = proc.memory_info().rss / 1024**2 - rss_before
                    peak_gpu = (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else 0.0
                    )
                    elapsed_ms = (t1 - t0) * 1e3

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

    print(f"✅ Benchmark complete - results saved to {args.output_csv}")


if __name__=="__main__":
    args = parse_args()
    main(args)