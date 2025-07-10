import glob, time, csv, psutil, os
import torch
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
from fastsam import FastSAM, FastSAMPrompt 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()

def predict(args):
    # load model
    model = FastSAM(model_path="/home/copter/FastSAM/weights/FastSAM-s.pt")
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    input = Image.open(args.img_path)
    input = input.convert("RGB")
    everything_results = model(
        input,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou    
    )
    bboxes = None
    points = None
    point_label = None

    # Define prompting method 
    prompt_process = FastSAMPrompt(input, everything_results, device=args.device)

    # Note: I'm not using manual or text prompts for this test
    # if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
    #         ann = prompt_process.box_prompt(bboxes=args.box_prompt)
    #         bboxes = args.box_prompt
    # elif args.text_prompt != None:
    #     ann = prompt_process.text_prompt(text=args.text_prompt)
    # elif args.point_prompt[0] != [0, 0]:
    #     ann = prompt_process.point_prompt(
    #         points=args.point_prompt, pointlabel=args.point_label
    #     )
    #     points = args.point_prompt
    #     point_label = args.point_label
    # else:
    #     ann = prompt_process.everything_prompt()

    # Return all predictions
    ann = prompt_process.everything_prompt()

    prompt_process.plot(
        annotations=ann,
        output_path=args.output+args.img_path.split("/")[-1],
        bboxes = bboxes,
        points = points,
        point_label = point_label,
        withContours=args.withContours,
        better_quality=args.better_quality,
    )

def main():
    # 1. Model init
    # model = Inference.FastSAMModel(model_path="/home/copter/FastSAM/weights/FastSAM-s.pt")
    model = FastSAM(model_path="/home/copter/FastSAM/weights/FastSAM-s.pt")
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


if __name__=="__main__":
    args = parse_args()
    main(args)