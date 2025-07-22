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

def postprocess(preds, img, orig_imgs, retina_masks, conf_thres, iou_thres, agnostic_nms=True):
    """Simplified postprocess that avoids the problematic NMS function."""
    
    print(f"[DEBUG] preds len: {len(preds)}")
    print(f"[DEBUG] preds[0] shape: {preds[0].shape}")  # Should be [8400, 37]
    print(f"[DEBUG] preds[1] shape: {preds[1].shape}")  # Should be [32, 160, 160]

    # Get predictions
    raw_preds = preds[0]  # shape: [8400, 37]
    proto = preds[1]      # shape: [32, 160, 160]

    # Split components from raw predictions
    boxes = raw_preds[:, 0:4]        # [8400, 4] - bbox coordinates  
    scores = raw_preds[:, 4]         # [8400] - confidence scores
    mask_coeffs = raw_preds[:, 5:]   # [8400, 32] - mask coefficients

    print(f"[DEBUG] Score range: {scores.min():.6f} to {scores.max():.6f}")
    print(f"[DEBUG] Number of predictions above {conf_thres}: {(scores > conf_thres).sum()}")

    # Simple confidence filtering (avoiding problematic NMS for now)
    valid_mask = scores > conf_thres
    
    if not valid_mask.any():
        print("[DEBUG] No detections above confidence threshold")
        orig_img = orig_imgs[0] if isinstance(orig_imgs, list) else orig_imgs
        return [Results(orig_img=orig_img, path="ok", names={0: "object"}, boxes=None, masks=None)]
    
    # Filter predictions
    filtered_boxes = boxes[valid_mask]      # [N, 4]
    filtered_scores = scores[valid_mask]    # [N]
    filtered_coeffs = mask_coeffs[valid_mask]  # [N, 32]
    
    print(f"[DEBUG] Filtered to {len(filtered_boxes)} detections")
    
    # Take top detections (simple alternative to NMS)
    num_keep = min(100, len(filtered_scores))  # Keep top 100
    top_indices = torch.topk(filtered_scores, num_keep).indices
    
    final_boxes = filtered_boxes[top_indices]
    final_scores = filtered_scores[top_indices]  
    final_coeffs = filtered_coeffs[top_indices]
    
    print(f"[DEBUG] Keeping top {len(final_boxes)} detections")

    results = []
    orig_img = orig_imgs[0] if isinstance(orig_imgs, list) else orig_imgs
    
    if len(final_boxes) == 0:
        print("[DEBUG] No final detections")
        results.append(Results(orig_img=orig_img, path="ok", names={0: "object"}, boxes=None, masks=None))
        return results

    # Process masks
    try:
        print(f"[DEBUG] Processing masks with coeffs shape: {final_coeffs.shape}")
        print(f"[DEBUG] Proto shape: {proto.shape}")
        print(f"[DEBUG] Boxes shape: {final_boxes.shape}")
        
        if retina_masks:
            # Scale boxes to original image size first
            if not isinstance(orig_imgs, torch.Tensor):
                scaled_boxes = ops.scale_boxes(img.shape[2:], final_boxes, orig_img.shape)
            else:
                scaled_boxes = final_boxes
            masks = ops.process_mask_native(proto, final_coeffs, scaled_boxes, orig_img.shape[:2])
        else:
            # Process masks at model resolution then scale boxes
            masks = ops.process_mask(proto, final_coeffs, final_boxes, img.shape[2:], upsample=True)
            if not isinstance(orig_imgs, torch.Tensor):
                final_boxes = ops.scale_boxes(img.shape[2:], final_boxes, orig_img.shape)
        
        print(f"[DEBUG] Processed masks shape: {masks.shape if masks is not None else None}")
        
    except Exception as e:
        print(f"[DEBUG] Mask processing failed: {e}")
        import traceback
        traceback.print_exc()
        masks = None

    # Create boxes tensor for Results (format: [x1, y1, x2, y2, conf, cls])
    classes = torch.zeros_like(final_scores)  # All class 0
    boxes_tensor = torch.cat([
        final_boxes,
        final_scores.unsqueeze(1),
        classes.unsqueeze(1)
    ], dim=1)

    results.append(Results(
        orig_img=orig_img, 
        path="ok", 
        names={0: "object"}, 
        boxes=boxes_tensor, 
        masks=masks
    ))

    return results


# Also add this simple IoU-based NMS if you want to try it later
def simple_nms(boxes, scores, iou_threshold=0.5):
    """Simple NMS implementation to avoid the problematic ultralytics version."""
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long)
    
    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(sorted_indices) > 0:
        # Take the highest scoring box
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes[current:current+1]  # [1, 4]
        remaining_boxes = boxes[sorted_indices[1:]]  # [N-1, 4]
        
        # Simple IoU calculation
        ious = box_iou(current_box, remaining_boxes).squeeze(0)  # [N-1]
        
        # Keep boxes with IoU below threshold
        sorted_indices = sorted_indices[1:][ious <= iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long)

def box_iou(box1, box2):
    """Calculate IoU between box1 and box2."""
    # box1: [1, 4], box2: [N, 4]
    
    # Get intersection coordinates
    x1 = torch.max(box1[:, 0:1], box2[:, 0:1].T)  # [1, N]
    y1 = torch.max(box1[:, 1:2], box2[:, 1:2].T)  # [1, N]  
    x2 = torch.min(box1[:, 2:3], box2[:, 2:3].T)  # [1, N]
    y2 = torch.min(box1[:, 3:4], box2[:, 3:4].T)  # [1, N]
    
    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate box areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [1]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [N]
    
    # Calculate union
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection  # [1, N]
    
    # Return IoU
    return intersection / (union + 1e-6)


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

    def segment(self, bgr_img, inp, retina_masks, conf, iou, agnostic_nms):
        """
        Enhanced segment method with better mask debugging and visualization
        """
        ## Inference
        print("[Input]: ", inp.shape)
        preds = self.model.infer(inp)

        # Your TRT model returns: [output0, output1] => [detections, proto]
        output0 = torch.from_numpy(preds[0])  # shape: [1, 37, 8400]
        output1 = torch.from_numpy(preds[1])  # shape: [1, 32, 160, 160]

        # Remove batch dim
        output0 = output0[0]  # [37, 8400]
        output1 = output1[0]  # [32, 160, 160]

        # Transpose output0 from [37, 8400] → [8400, 37] to match YOLO format
        output0 = output0.permute(1, 0)  # [8400, 37]

        preds = [output0, output1]

        print("Input shape, BGR img shape, retina_masks:", inp.shape, bgr_img.shape, retina_masks)
        result = postprocess(preds, inp, bgr_img, retina_masks, conf, iou, agnostic_nms)
        
        if not result or len(result) == 0:
            print("ERROR: No results from postprocess")
            return []
        
        if result[0].masks is None:
            print("ERROR: No masks in result")
            return []
            
        masks = result[0].masks.data
        print(f"Number of masks: {len(masks)}")
        print(f"Mask tensor shape: {masks.shape}")
        print(f"Mask dtype: {masks.dtype}")
        print(f"Mask value range: {masks.min():.4f} to {masks.max():.4f}")
        
        # Debug: Check if masks contain any non-zero values
        non_zero_masks = 0
        for i, mask in enumerate(masks):
            non_zero_pixels = (mask > 0.5).sum().item()  # Count pixels above threshold
            if non_zero_pixels > 0:
                non_zero_masks += 1
            if i < 5:  # Debug first 5 masks
                print(f"Mask {i}: {non_zero_pixels} pixels above 0.5 threshold")
        
        print(f"Masks with non-zero pixels: {non_zero_masks}/{len(masks)}")
        
        if non_zero_masks == 0:
            print("WARNING: No masks have pixels above 0.5 threshold!")
            # Try lower threshold
            for i, mask in enumerate(masks[:5]):
                non_zero_pixels = (mask > 0.1).sum().item()
                print(f"Mask {i}: {non_zero_pixels} pixels above 0.1 threshold")

        # Enhanced visualization with multiple approaches
        image_with_masks = np.copy(bgr_img)
        
        # Method 1: Original approach with higher alpha
        print("Applying masks with original method...")
        for i, mask_i in enumerate(masks[:20]):  # Limit to first 20 for performance
            # Convert to numpy if needed
            if torch.is_tensor(mask_i):
                mask_np = mask_i.cpu().numpy()
            else:
                mask_np = mask_i
                
            # Check mask properties
            mask_sum = np.sum(mask_np > 0.5)
            if mask_sum == 0:
                continue  # Skip empty masks
                
            r = randint(50, 255)   # Brighter colors
            g = randint(50, 255)
            b = randint(50, 255)
            rand_color = (r, g, b)
            
            print(f"Mask {i}: {mask_sum} pixels, color {rand_color}")
            
            try:
                image_with_masks = overlay(image_with_masks, mask_np, color=rand_color, alpha=0.7)
            except Exception as e:
                print(f"Overlay failed for mask {i}: {e}")
        
        # Save original method result
        cv2.imwrite("obj_segment_trt_original.png", image_with_masks)
        
        # Method 2: Direct binary mask overlay (debugging approach)
        print("Applying masks with direct binary method...")
        image_binary_masks = np.copy(bgr_img)
        
        for i, mask_i in enumerate(masks[:10]):  # First 10 masks
            if torch.is_tensor(mask_i):
                mask_np = mask_i.cpu().numpy()
            else:
                mask_np = mask_i
            
            # Create binary mask
            binary_mask = (mask_np > 0.5).astype(np.uint8)
            mask_pixels = np.sum(binary_mask)
            
            if mask_pixels == 0:
                continue
                
            print(f"Binary mask {i}: {mask_pixels} pixels")
            
            # Apply bright color directly to mask pixels
            color = [randint(100, 255) for _ in range(3)]
            colored_mask = np.zeros_like(bgr_img)
            colored_mask[binary_mask > 0] = color
            
            # Blend with image
            alpha = 0.6
            image_binary_masks = cv2.addWeighted(image_binary_masks, 1-alpha, colored_mask, alpha, 0)
        
        cv2.imwrite("obj_segment_trt_binary.png", image_binary_masks)
        
        # Method 3: Contour-based visualization
        print("Applying masks with contour method...")
        image_contours = np.copy(bgr_img)
        
        for i, mask_i in enumerate(masks[:10]):
            if torch.is_tensor(mask_i):
                mask_np = mask_i.cpu().numpy()
            else:
                mask_np = mask_i
            
            binary_mask = (mask_np > 0.5).astype(np.uint8)
            
            if np.sum(binary_mask) == 0:
                continue
                
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
                cv2.drawContours(image_contours, contours, -1, color, 2)
                print(f"Contour mask {i}: {len(contours)} contours")
        
        cv2.imwrite("obj_segment_trt_contours.png", image_contours)
        
        print("Saved three visualization versions:")
        print("- obj_segment_trt_original.png (original overlay method)")
        print("- obj_segment_trt_binary.png (direct binary overlay)")  
        print("- obj_segment_trt_contours.png (contour outlines)")
        
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