"""
python /home/copter/jetson_benchmark/livecam_demos/nanosam_webcam_demo.py \
--encoder /home/copter/onnx_models/nvidia_ai_iot_resnet18_image_encoder.onnx \
--decoder /home/copter/onnx_models/nvidia_ai_iot_mobile_sam_mask_decoder.onnx

conda activate nanosam_arm64


another source: https://huggingface.co/dragonSwing/nanosam


# Check camera setting

```
# List video devices
ls /dev/video*

# Get detailed info about video devices
v4l2-ctl --list-devices

# Check device capabilities
v4l2-ctl -d /dev/video0 --list-formats-ext

# Or for USB cameras
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! xvimagesink
```
"""

#!/usr/bin/env python3

import cv2
import numpy as np
import time
import onnxruntime as ort
import torch
from torchvision import transforms
import argparse
from typing import Tuple, Optional, List
from enum import Enum

class PromptMode(Enum):
    POINT = "point"
    HEADING = "heading"
    BBOX = "bbox"
    EVERYTHING = "everything"

class NanoSAMEnhanced:
    def __init__(self, 
                 encoder_path: str,
                 decoder_path: str,
                 device_id: int = 0,
                 input_size: Tuple[int, int] = (1024, 1024)):
        """
        Enhanced NanoSAM with multiple prompt types
        """
        self.device_id = device_id
        self.input_size = input_size
        
        # Initialize ONNX Runtime sessions with proper configuration for Jetson
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Configure session options to avoid thread affinity warnings on ARM64
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1  # Reduce inter-op parallelism
        sess_options.intra_op_num_threads = 4  # Set explicit thread count
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Suppress verbose logging
        sess_options.log_severity_level = 3  # Only show errors
        
        try:
            print("Loading encoder model...")
            self.encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=providers)
            print("Loading decoder model...")
            self.decoder_session = ort.InferenceSession(decoder_path, sess_options, providers=providers)
             
            print(f"Using enc_sess providers: {self.encoder_session.get_providers()}")
            print(f"Using dec_sess providers: {self.decoder_session.get_providers()}")
        except Exception as e:
            print(f"❌ GPU session creation failed: {e}")
            print("🔄 Falling back to CPU-only providers...")
            try:
                # Fallback to CPU only
                cpu_providers = ['CPUExecutionProvider']
                self.encoder_session = ort.InferenceSession(encoder_path, sess_options=sess_options, providers=cpu_providers)
                self.decoder_session = ort.InferenceSession(decoder_path, sess_options=sess_options, providers=cpu_providers)
                print("✅ CPU fallback successful")
            except Exception as cpu_e:
                print(f"❌ CPU fallback also failed: {cpu_e}")
                raise RuntimeError("Failed to load ONNX models with any provider")
        
        # Get input/output names
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name
        
        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        self.decoder_output_name = self.decoder_session.get_outputs()[0].name
        
        print(f"Decoder inputs: {self.decoder_input_names}")
        print(f"Decoder outputs: {[out.name for out in self.decoder_session.get_outputs()]}")
        
        # Check if we have multiple outputs (masks, scores, logits)
        self.decoder_outputs = [out.name for out in self.decoder_session.get_outputs()]
        if len(self.decoder_outputs) > 1:
            print(f"Multiple decoder outputs detected: {self.decoder_outputs}")
            # Look for mask-related output names
            mask_output_candidates = ['masks', 'low_res_masks', 'output_masks', 'segmentation_masks']
            for candidate in mask_output_candidates:
                if candidate in self.decoder_outputs:
                    self.decoder_output_name = candidate
                    print(f"Using mask output: {candidate}")
                    break
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize camera
        self.cap = None
        self.init_camera()
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Prompt state
        # self.current_mode = PromptMode.POINT
        self.current_mode = PromptMode.HEADING
        # self.current_mode = PromptMode.EVERYTHING
        self.positive_points = []
        self.negative_points = []
        self.fixed_points = []
        self.bbox_start = None
        self.bbox_end = None
        self.drawing_bbox = False
        self.everything_mode_active = False
        self.show_bbox = True                   # New toggle for bounding box visualization
        
        # Everything mode grid settings
        self.grid_size = 8   # 32  # Grid points for everything mode
        
        # Debug settings
        self.debug_mode = False
        self.save_masks = False
        self.frame_count = 0
        self.no_prompt_warning_counter = 0  # For throttling "no prompts" messages
        
        print("\n=== Controls ===")
        print("1: Point mode (left click: positive, right click: negative)")
        print("2: Bounding box mode (drag to create box)")
        print("3: Everything mode (segment all objects)")
        print("C: Clear all prompts")
        print("D: Toggle debug mode")
        print("S: Toggle save masks")
        print("Q: Quit")
        print("R: Reset")
    
    def init_camera(self):
        """Initialize camera for Logitech C925e"""
        # Method 1: Try V4L2 (usually best for USB cameras on Jetson)
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            if self.cap.isOpened():
                # Set optimal properties for C925e
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                # Test if we can read a frame
                ret, _ = self.cap.read()
                if ret:
                    print(f"Initialized Logitech C925e with V4L2: /dev/video{self.device_id}")
                    print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
                    return
        except Exception as e:
            print(f"V4L2 initialization failed: {e}")
        
        # Fallback methods...
        gst_pipeline = f'v4l2src device=/dev/video{self.device_id} ! image/jpeg,width=1920,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! appsink drop=1'
        
        try:
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                print(f"Initialized camera with GStreamer: {gst_pipeline}")
                return
        except Exception as e:
            print(f"GStreamer initialization failed: {e}")
        
        # Basic OpenCV fallback
        self.cap = cv2.VideoCapture(self.device_id)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Initialized camera with basic OpenCV: /dev/video{self.device_id}")
        else:
            raise RuntimeError("Failed to initialize camera")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for NanoSAM encoder"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb)
        input_batch = input_tensor.unsqueeze(0).numpy()
        return input_batch
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Run image through encoder"""
        preprocessed = self.preprocess_image(image)
        encoder_outputs = self.encoder_session.run(
            [self.encoder_output_name],
            {self.encoder_input_name: preprocessed}
        )
        return encoder_outputs[0]
    
    def create_everything_prompts(self, frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create grid of points for everything mode"""
        h, w = frame_shape[:2]
        model_h, model_w = self.input_size
        self.grid_size = 5
        
        # Create grid of points
        x_points = np.linspace(start=0, stop=model_w-1, num=self.grid_size, dtype=int)
        y_points = np.linspace(start=0, stop=model_h-1, num=7, dtype=int)
        
        # Calculate y-coordinates for the upper 1/5th
        # upper_nth_height = model_h // 8
        upper_nth_height = len(y_points) // 3
    
        y_points = y_points[1:int(upper_nth_height)]
        # y_points = np.linspace(start=0, stop=upper_nth_height - 1, num=self.grid_size, dtype=int)
        
        print('points for everything mode... x: ', len(x_points), ', y: ', len(y_points))

        # append points in a grid
        points = []
        for y in y_points:
            for x in x_points:
                points.append([x, y])
        
        point_coords = np.array([points], dtype=np.float32)
        point_labels = np.ones((1, len(points)), dtype=np.float32)  # All positive
        
        return point_coords, point_labels

    def set_point_prompt(self, frame_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create point prompts from clicked points"""
        h, w = frame_shape[:2]
        model_h, model_w = self.input_size
        
        px = h // 2
        py = h // 8

        self.fixed_points.append((px, py))

        points = []
        labels = []
        
        # Add hardcoded points
        points.append([px, py])
        labels.append(1)

        # # Add positive points
        # for px, py in self.positive_points:
        #     model_x = int(px * model_w / w)
        #     model_y = int(py * model_h / h)
        #     points.append([model_x, model_y])
        #     labels.append(1)
        
        # # Add negative points
        # for nx, ny in self.negative_points:
        #     model_x = int(nx * model_w / w)
        #     model_y = int(ny * model_h / h)
        #     points.append([model_x, model_y])
        #     labels.append(0)
        
        if not points:
            return None, None
        
        point_coords = np.array([points], dtype=np.float32)
        point_labels = np.array([labels], dtype=np.float32)
        
        return point_coords, point_labels
    
    def create_point_prompts(self, frame_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create point prompts from clicked points"""
        if not self.positive_points and not self.negative_points:
            return None, None
        
        h, w = frame_shape[:2]
        model_h, model_w = self.input_size
        
        points = []
        labels = []
        
        # Add positive points
        for px, py in self.positive_points:
            model_x = int(px * model_w / w)
            model_y = int(py * model_h / h)
            points.append([model_x, model_y])
            labels.append(1)
        
        # Add negative points
        for nx, ny in self.negative_points:
            model_x = int(nx * model_w / w)
            model_y = int(ny * model_h / h)
            points.append([model_x, model_y])
            labels.append(0)
        
        if not points:
            return None, None
        
        point_coords = np.array([points], dtype=np.float32)
        point_labels = np.array([labels], dtype=np.float32)
        
        return point_coords, point_labels
    
    def create_bbox_prompts(self, frame_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create bounding box prompts"""
        if self.bbox_start is None or self.bbox_end is None:
            return None, None
        
        h, w = frame_shape[:2]
        model_h, model_w = self.input_size
        
        # Convert bbox to model coordinates
        x1 = min(self.bbox_start[0], self.bbox_end[0]) * model_w / w
        y1 = min(self.bbox_start[1], self.bbox_end[1]) * model_h / h
        x2 = max(self.bbox_start[0], self.bbox_end[0]) * model_w / w
        y2 = max(self.bbox_start[1], self.bbox_end[1]) * model_h / h
        
        # Convert bbox to corner points
        points = [
            [x1, y1],  # Top-left
            [x2, y2],  # Bottom-right
        ]
        
        point_coords = np.array([points], dtype=np.float32)
        point_labels = np.array([[2, 3]], dtype=np.float32)  # 2,3 for bbox corners
        
        return point_coords, point_labels

    def decode_masks(self, 
                    image_embeddings: np.ndarray,
                    point_coords: Optional[np.ndarray] = None,
                    point_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode masks from embeddings and prompts
        Args:
            image_embeddings (np.ndarray)
            point_coords (np.ndarray)
            point_labels (np.ndarray)

        Return:
            all_detected_masks (list): Detected and filtered masks
        """
        
        if point_coords is None or point_labels is None:
            # Only print warnin g every 30 frames to reduce spam
            self.no_prompt_warning_counter += 1
            if self.no_prompt_warning_counter % 30 == 1:
                print("🔍 No prompts available")
            return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
        
        print(f"🔍 Decoding masks:")
        print(f"   Image embeddings shape: {image_embeddings.shape}")
        print(f"   Point coords shape: {point_coords.shape}")
        print(f"   Point coords: {point_coords}")
        print(f"   Point labels shape: {point_labels.shape}")
        print(f"   Point labels: {point_labels}")
        
        # Prepare decoder inputs
        decoder_inputs = {
            'image_embeddings': image_embeddings,
            'point_coords': point_coords,
            'point_labels': point_labels,
        }
        
        # Add optional inputs with defaults
        if 'mask_input' in self.decoder_input_names:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            decoder_inputs['mask_input'] = mask_input
            print(f"   Added mask_input: {mask_input.shape}")
            
        if 'has_mask_input' in self.decoder_input_names:
            decoder_inputs['has_mask_input'] = np.array([0], dtype=np.float32)
            print(f"   Added has_mask_input: [0]")
        
        print(f"   Decoder input names: {list(decoder_inputs.keys())}")
        
        try:
            # Run the decoder to get all outputs.
            start_time = time.time()
            all_outputs = self.decoder_session.run(None, decoder_inputs)
            decode_time = time.time() - start_time
            print(f"✅ Decoder successful in {decode_time*1000:.2f}ms")

            # Assume the outputs are in the order [scores, masks].
            # This is a common pattern for these models.
            scores_out, masks_out = all_outputs
            
            # Print info about outputs for debugging.
            print(f"   Output 0 (iou_predictions): shape {scores_out.shape}, range [{scores_out.min():.3f}, {scores_out.max():.3f}]")
            print(f"   Output 1 (low_res_masks): shape {masks_out.shape}, range [{masks_out.min():.3f}, {masks_out.max():.3f}]")

            # Set the confidence threshold.
            confidence_threshold = 0.75
            
            # Get the raw masks and scores, ensuring they are correctly shaped for iteration.
            scores = scores_out.flatten() # Flattens to (N,)
            masks_tensor = masks_out[0] # Removes the batch dimension, shape (N, 256, 256)

            all_detected_masks = []
            
            # Iterate through each mask and its corresponding score.
            for i in range(len(scores)):
                score = scores[i]
                mask = masks_tensor[i]

                # Filter masks based on the confidence threshold.
                if score >= confidence_threshold:
                    print(f"   ✅ Accepted mask {i} with score {score:.3f}")
                    all_detected_masks.append(mask)
                else:
                    print(f"   ❌ Rejected mask {i} with score {score:.3f}")
            
            if not all_detected_masks:
                print("   ⚠️ No masks passed the confidence threshold. Returning empty list.")
            
            # Note: The calling function (run) will need to handle this list of masks.
            return all_detected_masks
            
        except Exception as e:
            print(f"❌ Decoder failed: {e}")
            return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)

    def decode_masks_depr(self, 
                    image_embeddings: np.ndarray,
                    point_coords: Optional[np.ndarray] = None,
                    point_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode masks from embeddings and prompts"""
        
        if point_coords is None or point_labels is None:
            # Only print warnin g every 30 frames to reduce spam
            self.no_prompt_warning_counter += 1
            if self.no_prompt_warning_counter % 30 == 1:
                print("🔍 No prompts available")
            return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
        
        print(f"🔍 Decoding masks:")
        print(f"   Image embeddings shape: {image_embeddings.shape}")
        print(f"   Point coords shape: {point_coords.shape}")
        print(f"   Point coords: {point_coords}")
        print(f"   Point labels shape: {point_labels.shape}")
        print(f"   Point labels: {point_labels}")
        
        # Prepare decoder inputs
        decoder_inputs = {
            'image_embeddings': image_embeddings,
            'point_coords': point_coords,
            'point_labels': point_labels,
        }
        
        # Add optional inputs with defaults
        if 'mask_input' in self.decoder_input_names:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            decoder_inputs['mask_input'] = mask_input
            print(f"   Added mask_input: {mask_input.shape}")
            
        if 'has_mask_input' in self.decoder_input_names:
            decoder_inputs['has_mask_input'] = np.array([0], dtype=np.float32)
            print(f"   Added has_mask_input: [0]")
        
        print(f"   Decoder input names: {list(decoder_inputs.keys())}")
        
        try:
            # Run decoder - get ALL outputs to see what's available
            start_time = time.time()
            all_outputs = self.decoder_session.run(None, decoder_inputs)  # Get all outputs
            decode_time = time.time() - start_time
            
            print(f"✅ Decoder successful in {decode_time*1000:.2f}ms")
            print(f"   Number of outputs: {len(all_outputs)}")
            scores_out, masks_out = all_outputs
            
            # Analyze all outputs
            for i, output in enumerate(all_outputs):
                output_name = self.decoder_outputs[i] if i < len(self.decoder_outputs) else f"output_{i}"
                print(f"   Output {i} ({output_name}): shape {output.shape}, range [{output.min():.3f}, {output.max():.3f}]")
            
            # Find the mask output - prioritize low_res_masks over iou_predictions
            mask_output = None
            mask_idx = -1
            
            # Look for mask outputs (should have spatial dimensions)
            for i, output in enumerate(all_outputs):
                output_name = self.decoder_outputs[i] if i < len(self.decoder_outputs) else f"output_{i}"
                
                # Skip IoU predictions (typically shape like (1, N) where N is small)
                if 'iou' in output_name.lower() or (len(output.shape) == 2 and output.shape[1] <= 4):
                    print(f"   Skipping {output_name} (appears to be IoU/confidence scores)")
                    continue
                
                # Look for mask-like outputs (3D or 4D with spatial dimensions)
                if len(output.shape) >= 3:
                    mask_output = output
                    mask_idx = i
                    print(f"   ✅ Using {output_name} as mask output: {output.shape}")
                    break
            
            if mask_output is None:
                print("   ❌ No suitable mask output found!")
                return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
            
            # Handle the mask output shape: (1, 4, 256, 256) -> need to select best mask
            if len(mask_output.shape) == 4 and mask_output.shape[1] > 1:
                print(f"   🔍 Multiple masks detected: {mask_output.shape[1]} masks")
                
                # For NanoSAM, typically the masks are ranked by quality
                # Use the first mask (index 0) which should be the best
                best_mask = mask_output[0, 0]  # Shape: (256, 256)
                print(f"   Selected best mask (index 0): {best_mask.shape}, range [{best_mask.min():.3f}, {best_mask.max():.3f}]")
                
                # Resize to target resolution if needed
                if best_mask.shape != self.input_size:
                    best_mask_resized = cv2.resize(best_mask, self.input_size)
                    print(f"   Resized mask to: {best_mask_resized.shape}")
                else:
                    best_mask_resized = best_mask
                
                # Reshape to expected format: (1, 1, H, W)
                mask_output = best_mask_resized.reshape(1, 1, *self.input_size)
                
            elif len(mask_output.shape) == 3:
                # Shape like (1, 256, 256) or (4, 256, 256)
                if mask_output.shape[0] == 1:
                    # Single mask case
                    single_mask = mask_output[0]
                    if single_mask.shape != self.input_size:
                        single_mask = cv2.resize(single_mask, self.input_size)
                    mask_output = single_mask.reshape(1, 1, *self.input_size)
                else:
                    # Multiple masks, take the first one
                    best_mask = mask_output[0]
                    if best_mask.shape != self.input_size:
                        best_mask = cv2.resize(best_mask, self.input_size)
                    mask_output = best_mask.reshape(1, 1, *self.input_size)
                    
            elif len(mask_output.shape) == 2:
                # Single 2D mask
                if mask_output.shape != self.input_size:
                    mask_output = cv2.resize(mask_output, self.input_size)
                mask_output = mask_output.reshape(1, 1, *self.input_size)
            
            print(f"   Final mask shape: {mask_output.shape}")
            
            # Check for valid masks
            mask_data = mask_output[0, 0] if len(mask_output.shape) == 4 else mask_output
            
            # Count pixels above different thresholds
            thresholds = [0.0, 0.1, 0.5, 0.9]
            for thresh in thresholds:
                count = np.sum(mask_data > thresh)
                total_pixels = mask_data.size
                print(f"   Pixels > {thresh}: {count} ({count/total_pixels*100:.1f}%)")
            
            # If mask seems empty, try negative threshold (some models output negative values for background)
            if np.sum(mask_data > 0) == 0:
                print("   🔍 Checking negative thresholds...")
                for thresh in [-0.9, -0.5, -0.1]:
                    count = np.sum(mask_data < thresh)
                    print(f"   Pixels < {thresh}: {count} ({count/mask_data.size*100:.1f}%)")
            
            # Save debug mask if enabled
            if self.save_masks:
                self.frame_count += 1
                # Normalize for visualization
                debug_mask = mask_data.copy()
                if debug_mask.max() != debug_mask.min():
                    debug_mask = (debug_mask - debug_mask.min()) / (debug_mask.max() - debug_mask.min()) * 255
                else:
                    debug_mask = np.zeros_like(debug_mask)
                cv2.imwrite(f"debug_mask_{self.frame_count:04d}.png", debug_mask.astype(np.uint8))
                print(f"   💾 Saved debug mask: debug_mask_{self.frame_count:04d}.png")
            
            return mask_output
            
        except Exception as e:
            print(f"❌ Decoder failed: {e}")
            return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Enhanced mouse callback for different prompt modes"""
        
        if self.current_mode == PromptMode.POINT:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.positive_points.append((x, y))
                print(f"Added positive point: ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.negative_points.append((x, y))
                print(f"Added negative point: ({x}, {y})")
                
        elif self.current_mode == PromptMode.BBOX:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.bbox_start = (x, y)
                self.bbox_end = None
                self.drawing_bbox = True
                print(f"Started bbox at: ({x}, {y})")
                
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing_bbox:
                self.bbox_end = (x, y)
                
            elif event == cv2.EVENT_LBUTTONUP and self.drawing_bbox:
                self.bbox_end = (x, y)
                self.drawing_bbox = False
                print(f"Completed bbox: {self.bbox_start} to {self.bbox_end}")
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time  
    
    def overlay_masks(self, image: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay multiple segmentation masks on image"""
        result = image.copy()
        
        print(f"🎨 Overlaying masks:")
        print(f"   Input image shape: {image.shape}")
        print(f"   Input masks shape: {masks.shape}")
        
        if len(masks.shape) == 4:
            masks = masks[0]  # Remove batch dimension
            print(f"   Masks after batch removal: {masks.shape}")
        
        h, w = image.shape[:2]
        
        # Colors for different masks
        colors = [
            [0, 255, 0],    # Green
            [255, 0, 0],    # Blue  
            [0, 0, 255],    # Red
            [255, 255, 0],  # Cyan
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Yellow
        ]
        
        mask_applied = False
        
        # Handle single mask case
        if len(masks.shape) == 2:
            masks = [masks]
        
        for i, mask in enumerate(masks):
            print(f"   Processing mask {i}: shape {mask.shape}, range [{mask.min():.3f}, {mask.max():.3f}]")
            
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask, (w, h))
                print(f"   Resized mask to: {mask_resized.shape}")
            else:
                mask_resized = mask
            
            # Try different thresholds to see what works
            # For SAM models, masks can have different value ranges
            thresholds = [0.0, 0.1, 0.3, 0.5]
            
            # Also try adaptive threshold based on mask statistics
            if mask_resized.max() > mask_resized.min():
                adaptive_thresh = mask_resized.mean() + 0.5 * mask_resized.std()
                thresholds.append(adaptive_thresh)
                print(f"   Added adaptive threshold: {adaptive_thresh:.3f}")
            
            for threshold in thresholds:
                mask_binary = (mask_resized > threshold).astype(np.uint8)
                pixel_count = np.sum(mask_binary)
                print(f"   Threshold {threshold:.3f}: {pixel_count} pixels ({pixel_count/(h*w)*100:.1f}%)")
                
                # Accept mask if it has reasonable coverage (0.1% to 50% of image)
                coverage_percent = pixel_count/(h*w)*100
                if 0.1 <= coverage_percent <= 50.0 and not mask_applied:
                    # Create colored overlay
                    color = colors[i % len(colors)]
                    overlay = result.copy()
                    overlay[mask_binary == 1] = color
                    
                    # Blend with result
                    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
                    mask_applied = True
                    print(f"   ✅ Applied mask {i} with threshold {threshold:.3f}, color {color}")
                    break
        
        if not mask_applied:
            print("   ⚠️  No masks were applied (all below threshold)")
        
        return result
    
    def overlay_bboxes(self, image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Overlay bounding boxes on segmented regions."""
        result = image.copy()
        
        print(f"📦 Overlaying bounding boxes:")
        print(f"   Input is a list of {len(masks)} masks.") # Correct way to check the input
        
        h, w = image.shape[:2]

        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        bbox_drawn = False
        
        # This loop correctly iterates over the list of masks
        for i, mask in enumerate(masks):
            # Check for empty masks before processing
            if mask.size == 0:
                print(f"   Skipping mask {i} as it is empty.")
                continue

            # Assuming the masks in the list are 2D arrays (H, W)
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask
            
            mask_binary = (mask_resized > 0).astype(np.uint8) * 255 
            
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"   No contours found for mask {i}.")
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            
            x, y, w_bbox, h_bbox = cv2.boundingRect(largest_contour)
            
            color = colors[i % len(colors)]
            cv2.rectangle(result, (x, y), (x + w_bbox, y + h_bbox), color, 2)
            bbox_drawn = True
            print(f"   ✅ Drawn bbox for mask {i} with color {color}")

        if not bbox_drawn:
            print("   ⚠️  No bounding boxes were drawn.")
        
        return result

    def draw_prompts(self, image: np.ndarray) -> np.ndarray:
        """Draw current prompts on image"""
        result = image.copy()

        if self.current_mode == PromptMode.HEADING:
            px, py = self.fixed_points[0]
            cv2.circle(result, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(result, (px, py), 7, (255, 255, 255), 2)

        # Draw positive points (green circles)
        for px, py in self.positive_points:
            cv2.circle(result, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(result, (px, py), 7, (255, 255, 255), 2)
        
        # Draw negative points (red circles)
        for nx, ny in self.negative_points:
            cv2.circle(result, (nx, ny), 5, (0, 0, 255), -1)
            cv2.circle(result, (nx, ny), 7, (255, 255, 255), 2)
        
        # Draw bounding box
        if self.bbox_start and self.bbox_end:
            cv2.rectangle(result, self.bbox_start, self.bbox_end, (255, 0, 255), 2)
        elif self.bbox_start and self.drawing_bbox and self.bbox_end:
            cv2.rectangle(result, self.bbox_start, self.bbox_end, (128, 0, 128), 2)
        
        return result
    
    def clear_prompts(self):
        """Clear all prompts"""
        self.positive_points = []
        self.negative_points = []
        self.bbox_start = None
        self.bbox_end = None
        self.drawing_bbox = False
        self.everything_mode_active = False
        print("Cleared all prompts")
    
    def run(self):
        """Run enhanced live segmentation"""
        print("Starting enhanced NanoSAM live segmentation...")
        
        cv2.namedWindow('NanoSAM Enhanced')
        cv2.setMouseCallback('NanoSAM Enhanced', self.mouse_callback)
        
        image_embeddings = None
        last_embedding_time = 0
        embedding_interval = 0.5
        inference_fps = 0.0
        inference_latency_ms = 0.0
        inference_start_time = None
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Update image embeddings periodically
                if current_time - last_embedding_time > embedding_interval:
                    image_embeddings = self.encode_image(frame)
                    last_embedding_time = current_time
                
                result_frame = frame.copy()
                
                # Get prompts based on current mode
                point_coords, point_labels = None, None
                    
                if self.current_mode == PromptMode.EVERYTHING and image_embeddings is not None:
                    point_coords, point_labels = self.create_everything_prompts(frame.shape)
                    # Visualize the grid points
                    if point_coords is not None:
                        h, w = frame.shape[:2]
                        model_h, model_w = self.input_size
                        for point in point_coords.squeeze(0).astype(int):
                            frame_x = int(point [0] * w / model_w)
                            frame_y = int(point [1] * h / model_h)
                            cv2.circle(result_frame, (frame_x, frame_y), 2, (255, 255, 0), -1) # Draw a small yellow circle
                elif self.current_mode == PromptMode.HEADING:
                    print("PromptMode.HEADING...")
                    point_coords, point_labels = self.set_point_prompt(frame.shape)
                elif self.current_mode == PromptMode.POINT:
                    point_coords, point_labels = self.create_point_prompts(frame.shape)
                elif self.current_mode == PromptMode.BBOX:
                    point_coords, point_labels = self.create_bbox_prompts(frame.shape)
                
                # Perform segmentation
                if point_coords is not None and image_embeddings is not None:

                    # --- Start Inference Timing ---
                    inference_start_time = time.time()

                    # This now returns a LIST of masks...
                    print(f"\n🚀 Running segmentation - Mode: {self.current_mode.value}")
                    masks_list = self.decode_masks(image_embeddings, point_coords, point_labels)
                    
                    # --- End Inference Timing ---
                    inference_time_s = time.time() - inference_start_time
                    inference_fps = 1.0 / inference_time_s if inference_time_s > 0 else 0
                    inference_latency_ms = inference_time_s * 1000

                    print(f"✅ Got {len(masks_list)} masks to display.")
                    
                    # The overlay functions must now handle a list of masks
                    if masks_list:
                        # The overlay functions need to be updated to accept a list
                        if self.show_bbox:
                            result_frame = self.overlay_bboxes(image=result_frame, masks=masks_list)
                        else:
                            result_frame = self.overlay_masks(image=result_frame, masks=masks_list)
                    else:
                        print(f"❌ No valid masks to display.")
                else:
                    if point_coords is None and self.no_prompt_warning_counter % 30 == 1:
                        print("🔍 No prompts available")
                        inference_fps = 0.0
                        inference_latency_ms = 0.0
                    if image_embeddings is None:
                        print("🔍 No image embeddings available")
                        inference_fps = 0.0
                        inference_latency_ms = 0.0
                
                # Draw prompts
                result_frame = self.draw_prompts(result_frame)
                
                # Update and display info
                self.update_fps()
                
                # Status text
                mode_text = f"Mode: {self.current_mode.value.upper()}"
                cv2.putText(result_frame, mode_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                fps_text = f'FPS: {self.current_fps:.1f}'
                cv2.putText(result_frame, fps_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                inference_text = f'Inference: {inference_latency_ms:.1f} ms ({inference_fps:.1f} FPS)'
                cv2.putText(result_frame, inference_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Point counts
                if self.current_mode == PromptMode.POINT:
                    point_text = f'Points: +{len(self.positive_points)} -{len(self.negative_points)}'
                    cv2.putText(result_frame, point_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Instructions
                instructions = [
                    "1:Point 2:BBox 3:Everything C:Clear D:Debug S:SaveMasks Q:Quit R:Reset"
                ]
                
                y_offset = result_frame.shape[0] - 20
                for instruction in instructions:
                    cv2.putText(result_frame, instruction, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset -= 20
                
                cv2.imshow('NanoSAM Enhanced', result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.current_mode = PromptMode.POINT
                    print("Switched to Point mode")
                elif key == ord('2'):
                    self.current_mode = PromptMode.BBOX
                    print("Switched to Bounding Box mode")
                elif key == ord('3'):
                    self.current_mode = PromptMode.EVERYTHING
                    print("Switched to Everything mode")
                elif key == ord('c'):
                    self.clear_prompts()
                elif key == ord('r'):
                    self.clear_prompts()
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('s'):
                    self.save_masks = not self.save_masks
                    print(f"Save masks: {'ON' if self.save_masks else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Enhanced NanoSAM Live Segmenation')
    parser.add_argument('--encoder', required=True, help='Path to encoder ONNX model')
    parser.add_argument('--decoder', required=True, help='Path to decoder ONNX model')
    parser.add_argument('--device', type=int, default=0, help='Camera device ID')
    parser.add_argument('--size', type=int, nargs=2, default=[1024, 1024], 
                       help='Input size for model (width height)')
    parser.add_argument('--grid-size', type=int, default=32, 
                       help='Grid size for everything mode (default: 32)')
    
    args = parser.parse_args()
    
    segmenter = NanoSAMEnhanced(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        device_id=args.device,
        input_size=tuple(args.size)
    )
    
    segmenter.grid_size = args.grid_size
    segmenter.run()

if __name__ == '__main__':
    main()