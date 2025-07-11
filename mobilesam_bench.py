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
    iou_list = str2list(args.iou, float)
    conf_list = str2list(args.conf, float)

    device = torch.device(args.device)
    # load & prep model
    sam = sam_model_registry[args.model_type](checkpoint=args.model_path)
    sam.to(device=device)
    sam.eval()

    # gather images
    imgs = sorted(glob.glob(args.img_folder))
    if not imgs:
        raise FileNotFoundError(f"No images found with pattern {args.img_folder}")

    # prepare CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "model_type", "device", "img", "imgsz",
            "pred_iou_thresh", "stability_score_thresh",
            "time_ms", "gpu_mem_mb", "cpu_mem_mb"
        ])

        for pred_iou in iou_list:
            for stability_score in conf_list:
                # build a fresh generator for each threshold combo
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    pred_iou_thresh=pred_iou,
                    stability_score_thresh=stability_score
                )
                for img_path in imgs:
                    # reset GPU counters
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)

                    proc = psutil.Process(os.getpid())
                    rss_before = proc.memory_info().rss / 1024**2

                    # load and optionally resize
                    img = Image.open(img_path).convert("RGB")
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

                    # run mask generation
                    t0 = time.perf_counter()
                    masks = mask_generator.generate(img_np)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t1 = time.perf_counter()

                    # measure
                    elapsed_ms = (t1 - t0) * 1e3
                    peak_gpu = (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else 0.0
                    )
                    delta_cpu = proc.memory_info().rss / 1024**2 - rss_before

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

    print(f"✅ Mobile SAM benchmark complete - results saved to {args.output_csv}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
