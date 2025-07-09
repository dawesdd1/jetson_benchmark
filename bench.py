import glob, time, csv, psutil, os
import torch
from FastSAM.Inference  SamPredictor, sam_model_registry# or however you import

# 1. Model init
model = Inference.FastSAMModel(model_path="/home/copter/FastSAM/weights/FastSAM-s.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Parameter grid
IOU_THRESHOLDS  = [0.1, 0.3, 0.5, 0.7]
CONF_THRESHOLDS = [0.1, 0.3, 0.5, 0.7]

# 3. Image list
imgs = glob.glob("/home/copter/jetson_benchmark/bench.py*.*")

# 4. CSV output
with open("bench_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iou", "conf", "img", "time_ms", "gpu_mem_mb", "cpu_mem_mb"])

    for iou in IOU_THRESHOLDS:
        for conf in CONF_THRESHOLDS:
            for img_path in imgs:
                # reset peak stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                # record CPU mem before
                proc = psutil.Process(os.getpid())
                mem_before = proc.memory_info().rss / (1024**2)  # MB

                t0 = time.perf_counter()
                _ = model.predict(
                    image=img_path,
                    iou_threshold=iou,
                    conf_threshold=conf,
                    # ... any other flags
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                # measure GPU and CPU mem
                peak_gpu = torch.cuda.max_memory_allocated(device) / (1024**2)
                mem_after = proc.memory_info().rss / (1024**2)

                writer.writerow([
                    iou,
                    conf,
                    os.path.basename(img_path),
                    (t1 - t0)*1000,
                    peak_gpu,
                    mem_after - mem_before
                ])
