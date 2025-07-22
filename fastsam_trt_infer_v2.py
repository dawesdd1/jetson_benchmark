"""

From the Docs:

python fastsam_trt_infer.py --engine_path <engine_path> --image <image_path> --output <output_path>

##--------- example ---------- 

python fastsam_trt_infer.py \
    --engine_path "/home/copter/engine_models/FastSAM-s_fp32.engine" \
    --image "/home/copter/jetson_benchmark/images/nyc_drone_birdseye_1.png" \
    --output="../fastsam_unnamed.jpg"

"""

import time
import numpy as np
import tensorrt as trt
from cuda import cudart
import time
import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from PIL import Image
from random import randint
import argparse

import common
from fastsam_utils import overlay

class TensorRTInfer:
    """
    Implements inference for a TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)
            
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_input_shape(name, profile_shape[2])
                shape = self.context.get_tensor_shape(name)

            if is_input:
                self.batch_size = shape[0]
            
            size = dtype.itemsize
            for s in shape:
                size *= s
            
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        # Idx 0: ikely boxes/scores, Idx 1: Likely mask proto
        print("Engine output tensor count:", len(self.outputs))
        for idx, output in enumerate(self.outputs):
            print(f"[{idx}] name: {output['name']}, shape: {output['shape']}, dtype: {output['dtype']}")

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0
    
    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        common.memcpy_host_to_device(self.inputs[0]['allocation'], batch)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            common.memcpy_device_to_host(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        return [o['host_allocation'] for o in self.outputs]

def postprocess(preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=True):
    """Postprocess predictions from FastSAM TensorRT engine."""

    # Run NMS on predictions
    print(f"[DEBUG] preds len: {len(preds)}")
    print(f"[DEBUG] 🔧 Sample row [0]: {preds[0][0][:10]}")
    print(f"[DEBUG] 🔧 Sample row [1]: {preds[1][0][:10]}")

    # Perform NMS
    raw_preds = preds[0]  # shape: [8400, 37]

    # Split out components
    xyxy = raw_preds[:, 0:4]
    conf = raw_preds[:, 4:5]  # shape: [8400, 1]
    mask_coeffs = raw_preds[:, 5:]  # shape: [8400, 32]

    # Inject dummy class column (all class 0)
    cls = torch.zeros_like(conf)  # shape: [8400, 1]

    # Reconstruct for NMS: [x1, y1, x2, y2, conf, cls, mask_coeffs...]
    nms_input = torch.cat([xyxy, conf, cls, mask_coeffs], dim=1)

    # Now safe to call NMS
    p = ops.non_max_suppression(nms_input, conf_thres=conf, iou_thres=iou, agnostic=agnostic_nms, max_det=100)
    results = []

    proto = preds[1]  # full mask proto tensor: [32, 160, 160]
    print(f"[DEBUG] proto shape: {proto.shape}")


    print(f"[DEBUG] NMS output type: {type(p)}, len: {len(p)}")
    for i, pi in enumerate(p):
        print(f"[DEBUG] p[{i}] type: {type(pi)}")
        if pi is None or pi.shape[0] == 0:
            print(f"[DEBUG] No detections")
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            results.append(Results(orig_img=orig_img, path="ok", names="segment", boxes=None, masks=None))
            continue

        print(f"[DEBUG] Detected {pi.shape[0]} objects")
        print(f"[DEBUG] mask coeff shape: {pi[:, 6:].shape}")
        
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        img_path = "ok"

        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pi[:, :4] = ops.scale_boxes(img.shape[2:], pi[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pi[:, 6:], pi[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pi[:, 6:], pi[:, :4], img.shape[2:], upsample=True)  # HWC
            if not isinstance(orig_imgs, torch.Tensor):
                pi[:, :4] = ops.scale_boxes(img.shape[2:], pi[:, :4], orig_img.shape)

        results.append(
            Results(orig_img=orig_img, path=img_path, names="segment", boxes=pi[:, :6], masks=masks)
        )

    return results


def pre_processing(img_origin, imgsz=1024):
    h, w = img_origin.shape[:2]
    if h > w:
        scale = min(imgsz / h, imgsz / w)
        inp = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        nw = int(w * scale)
        nh = int(h * scale)
        a = int((nh - nw) / 2)
        inp[: nh, a:a + nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    else:
        scale = min(imgsz / h, imgsz / w)
        inp = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        nw = int(w * scale)
        nh = int(h * scale)
        a = int((nw - nh) / 2)

        inp[a: a + nh, :nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    rgb = np.array([inp], dtype=np.float32) / 255.0
    rgb = np.transpose(rgb, (0, 3, 1, 2))
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    return rgb

def load_and_preprocess_image(image_path, size=640):
    """
    HELPER: Loads an image and preprocesses it to the required square size.

    Args:
        image_path (str): The path to the input image.
        size (int): The target size (width and height) for the output image.

    Returns:
        tuple: A tuple containing the original BGR image and the preprocessed RGB tensor.
    """
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # The pre_processing function from the script handles resizing and padding
    processed_tensor = pre_processing(bgr_image, imgsz=size)
    return bgr_image, processed_tensor


class FastSam(object):
    def __init__(self,
                 model_weights,
                 max_size=640): # Changed default to 640
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TensorRTInfer(model_weights)

    def segment(self, bgr_img, inp, retina_masks, conf, iou, agnostic_nms): # Added 'inp' here
        ## Padded resize
        # Preprocessing is now done outside, 'inp' is the preprocessed tensor.

        ## Inference
        print("[Input]: ", inp.shape) # Now 'inp' is defined
        preds = self.model.infer(inp) # Assuming the infer method exists and works


        # Your TRT model returns: [output0, output1] => [detections, proto]
        output0 = torch.from_numpy(preds[0])  # shape: [1, 37, 8400]
        output1 = torch.from_numpy(preds[1])  # shape: [1, 32, 160, 160]

        # Remove batch dim
        output0 = output0[0]  # [37, 8400]
        output1 = output1[0]  # [32, 160, 160]

        # Transpose output0 from [37, 8400] → [8400, 37] to match YOLO format
        output0 = output0.permute(1, 0)  # [8400, 37]

        preds = [output0, output1]

        # Retina masks
        print("imp shape, BRG img shape, retima_masks:", inp.shape, bgr_img.shape, retina_masks)
        result = postprocess(preds, inp, bgr_img, retina_masks, conf, iou, agnostic_nms)
        masks = result[0].masks.data
        print("len of mask: ", len(masks))

        image_with_masks = np.copy(bgr_img)
        for i, mask_i in enumerate(masks):
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            rand_color = (r, g, b)
            image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
        cv2.imwrite("obj_segment_trt.png", image_with_masks)

        return masks

    def batch_segment(self, img_list, retina_masks, conf, iou, agnostic_nms):
        ## Padded resize
        tenosr = []
        org = []
        for path in img_list:
            bgr_img = cv2.imread(path)
            org.append(bgr_img)
            inp = pre_processing(bgr_img, self.imgsz[0])
            tenosr.append(inp)
        inp = np.concatenate(tenosr, axis=0)
        ## Inference
        print("[Input]: ", inp[0].transpose(0, 1, 2).shape)
        preds = self.model.infer(inp)
        data_0 = torch.from_numpy(preds[5])
        data_1 = [[torch.from_numpy(preds[2]), torch.from_numpy(preds[3]), torch.from_numpy(preds[4])],
                  torch.from_numpy(preds[1]), torch.from_numpy(preds[0])]
        preds = [data_0, data_1]
        print(inp.shape, tenosr[0].shape, retina_masks)
        results = postprocess(preds, inp, org[0], retina_masks, conf, iou, agnostic_nms)
        
        for index, result in enumerate(results):
            masks = result.masks.data
            print("len of mask: ", len(masks))
            image_with_masks = np.copy(org[index])
            for i, mask_i in enumerate(masks):
                r = randint(0, 255)
                g = randint(0, 255)
                b = randint(0, 255)
                rand_color = (r, g, b)
                image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
            image_with_masks = np.hstack([org[index], image_with_masks])
            cv2.imwrite(f"{index}_obj_segment_trt.png", image_with_masks)

        return masks

if __name__ == '__main__':
    # TODO: add these arguments to the argparse
    retina_masks = True
    conf = 0.1
    iou = 0.25
    agnostic_nms = False

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine_path",
        type=str,
        help="The file path of the TensorRT engine."
    )

    parser.add_argument(
        "--image",
        type=str,
        help="The file path of the image provided as input for inference."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The path to output the inference visualization."
    )

    args = parser.parse_args()

    if args.output is None:
        output_path = '.'.join(args.image.split('.')[:-1]) + "_output.jpg"
    else:
        output_path = args.output

    # Initialize the model with the correct weights path
    model = FastSam(model_weights=args.engine_path)

    # Use the new helper function to load and process the image
    original_bgr, processed_rgb_tensor = load_and_preprocess_image(args.image, size=640)

    # The 'segment' method now receives the original image and the preprocessed tensor
    # We'll need to slightly modify the segment method to accept the preprocessed tensor directly.

    # --- Modification to segment method is shown below ---
    masks = model.segment(original_bgr, processed_rgb_tensor, retina_masks, conf, iou, agnostic_nms)

    # The rest of the script for overlaying masks and saving the image remains the same.
    # This part assumes the 'segment' method returns a final image with masks.
    # Based on your script, 'segment' returns masks, so let's create the final image.

    image_with_masks = np.copy(original_bgr)
    for i, mask_i in enumerate(masks):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        # The overlay function combines the image with the mask
        image_with_masks = overlay(image_with_masks, mask_i.cpu().numpy(), color=rand_color, alpha=0.5)

    cv2.imwrite(output_path, image_with_masks)
    print(f"Inference complete. Output saved to {output_path}")
    
    # TODO: to add a flag: single images / batch to the arguments parser
    # #batch inference
    # imgs = ['xx.bmp', 'xx.bmp',
    #          'xx.bmp', 'xx.bmp']
    # masks = model.batch_segment(imgs, retina_masks, conf, iou, agnostic_nms)