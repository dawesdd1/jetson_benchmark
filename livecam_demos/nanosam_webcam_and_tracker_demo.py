"""
python /home/copter/jetson_benchmark/livecam_demos/nanosam_webcam_and_tracker_demo.py \
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
from PIL import Image
import PIL 

# tracking imports
from nanosam.utils.predictor import Predictor
from nanosam.utils.tracker import Tracker
from nanosam.utils.predictor import ONNXPredictor
from nanosam.utils.tracker import ONNXTracker

import logging

# ---- LOGGING CONF ----------------------- #

logger = logging.getLogger(__name__)

# For development/debugging
logging.basicConfig(level=logging.DEBUG)

# ----------------------------------------- #

class PromptMode(Enum):
    POINT = "point"
    HEADING = "heading"
    BBOX = "bbox"
    EVERYTHING = "everything"
    TRACKING = "tracking"

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
            logger.info("Loading encoder model...")
            self.encoder_session = ort.InferenceSession(
                encoder_path, 
                sess_options, 
                providers=providers
            )
            logger.info("Loading decoder model...")
            self.decoder_session = ort.InferenceSession(
                decoder_path, 
                sess_options, 
                providers=providers
            )
             
            logger.info(f"Using enc_sess providers: {self.encoder_session.get_providers()}")
            logger.info(f"Using dec_sess providers: {self.decoder_session.get_providers()}")
        except Exception as e:
            logger.info(f"❌ GPU session creation failed: {e}")
            logger.info("🔄 Falling back to CPU-only providers...")
            try:
                # Fallback to CPU only
                cpu_providers = ['CPUExecutionProvider']
                self.encoder_session = ort.InferenceSession(encoder_path, sess_options=sess_options, providers=cpu_providers)
                self.decoder_session = ort.InferenceSession(decoder_path, sess_options=sess_options, providers=cpu_providers)
                logger.info("✅ CPU fallback successful")
            except Exception as cpu_e:
                logger.info(f"❌ CPU fallback also failed: {cpu_e}")
                raise RuntimeError("Failed to load ONNX models with any provider")
        
        # Get input/output names
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name
        
        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        self.decoder_output_name = self.decoder_session.get_outputs()[0].name
        
        logger.info(f"Decoder inputs: {self.decoder_input_names}")
        logger.info(f"Decoder outputs: {[out.name for out in self.decoder_session.get_outputs()]}")
        
        # Check if we have multiple outputs (masks, scores, logits)
        self.decoder_outputs = [out.name for out in self.decoder_session.get_outputs()]
        if len(self.decoder_outputs) > 1:
            logger.info(f"Multiple decoder outputs detected: {self.decoder_outputs}")
            # Look for mask-related output names
            mask_output_candidates = ['masks', 'low_res_masks', 'output_masks', 'segmentation_masks']
            for candidate in mask_output_candidates:
                if candidate in self.decoder_outputs:
                    self.decoder_output_name = candidate
                    logger.info(f"Using mask output: {candidate}")
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
        self.current_mode = PromptMode.TRACKING
        # self.current_mode = PromptMode.HEADING
        # self.current_mode = PromptMode.EVERYTHING
        self.show_bbox = True                   # New toggle for bounding box visualization


        self.positive_points = []
        self.negative_points = []
        self.fixed_points = []
        self.bbox_start = None
        self.bbox_end = None
        self.drawing_bbox = False
        self.everything_mode_active = False

        
        # Everything mode grid settings
        self.grid_size = 8   # 32  # Grid points for everything mode
        
        # Tracker initialization
        try:
            # Create predictor with ONNX models (not TensorRT engines)
            self.predictor = ONNXPredictor(
                image_encoder_size=input_size[0],
                orig_image_encoder_size=input_size[0],
                encoder_session=self.encoder_session,    # Pass existing session
                decoder_session=self.decoder_session     # Pass existing session
            )
            self.tracker = ONNXTracker(self.predictor)
            logger.info("✅ Tracker initialized with ONNX models")
        except Exception as e:
            logger.info(f"❌ Failed to initialize tracker: {e}")
            self.tracker = None
    
        # Tracking state
        self.tracking_active = False
        self.tracked_mask = None
        self.tracked_point = None
        self.track_token = None
        self.token = None
        self._targets = []
        self._features = []

        # Debug settings
        self.debug_mode = False
        self.save_masks = False
        self.frame_count = 0
        self.no_prompt_warning_counter = 0  # For throttling "no prompts" messages
        
        logger.info("\n=== Controls ===")
        logger.info("1: Point mode (left click: positive, right click: negative)")
        logger.info("2: Bounding box mode (drag to create box)")
        logger.info("3: Everything mode (segment all objects)")
        logger.info("4: Tracker mode (double-click to track)")
        logger.info("C: Clear all prompts")
        logger.info("D: Toggle debug mode")
        logger.info("S: Toggle save masks")
        logger.info("Q: Quit")
        logger.info("R: Reset")
    
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
                    logger.info(f"Initialized Logitech C925e with V4L2: /dev/video{self.device_id}")
                    logger.info(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    logger.info(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
                    return
        except Exception as e:
            logger.info(f"V4L2 initialization failed: {e}")
        
        # Fallback methods...
        gst_pipeline = f'v4l2src device=/dev/video{self.device_id} ! image/jpeg,width=1920,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! appsink drop=1'
        
        try:
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                logger.info(f"Initialized camera with GStreamer: {gst_pipeline}")
                return
        except Exception as e:
            logger.info(f"GStreamer initialization failed: {e}")
        
        # Basic OpenCV fallback
        self.cap = cv2.VideoCapture(self.device_id)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            logger.info(f"Initialized camera with basic OpenCV: /dev/video{self.device_id}")
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
        
        logger.info('points for everything mode... x: ', len(x_points), ', y: ', len(y_points))

        # append points in a grid
        points = []
        for y in y_points:
            for x in x_points:
                points.append([x, y])
        
        point_coords = np.array([points], dtype=np.float32)
        point_labels = np.ones((1, len(points)), dtype=np.float32)  # All positive
        
        return point_coords, point_labels

    def set_point_prompt(self, frame_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create point prompts from a set headig point
        Corresponds to setting self.current_mode = PromptMode.HEADING
        """
        h, w = frame_shape[:2]
        model_h, model_w = self.input_size
        
        px = w // 2
        py = h // 8

        self.fixed_points.append((px, py))

        points = []
        labels = []
        
        # Add scaled hardcoded points
        model_x = int(px * model_w / w)
        model_y = int(py * model_h / h)
        points.append([model_x, model_y])
        labels.append(1)
        
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
                logger.info("🔍 No prompts available")
            return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
        
        logger.info(f"🔍 Decoding masks:")
        logger.info(f"   Image embeddings shape: {image_embeddings.shape}")
        logger.info(f"   Point coords shape: {point_coords.shape}")
        logger.info(f"   Point coords: {point_coords}")
        logger.info(f"   Point labels shape: {point_labels.shape}")
        logger.info(f"   Point labels: {point_labels}")
        
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
            logger.info(f"   Added mask_input: {mask_input.shape}")
            
        if 'has_mask_input' in self.decoder_input_names:
            decoder_inputs['has_mask_input'] = np.array([0], dtype=np.float32)
            logger.info(f"   Added has_mask_input: [0]")
        
        logger.info(f"   Decoder input names: {list(decoder_inputs.keys())}")
        
        try:
            # Run the decoder to get all outputs.
            start_time = time.time()
            all_outputs = self.decoder_session.run(None, decoder_inputs)
            decode_time = time.time() - start_time
            logger.info(f"✅ Decoder successful in {decode_time*1000:.2f}ms")

            # Assume the outputs are in the order [scores, masks].
            # This is a common pattern for these models.
            scores_out, masks_out = all_outputs
            
            # Print info about outputs for debugging.
            logger.info(f"   Output 0 (iou_predictions): shape {scores_out.shape}, range [{scores_out.min():.3f}, {scores_out.max():.3f}]")
            logger.info(f"   Output 1 (low_res_masks): shape {masks_out.shape}, range [{masks_out.min():.3f}, {masks_out.max():.3f}]")

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
                    logger.info(f"   ✅ Accepted mask {i} with score {score:.3f}")
                    all_detected_masks.append(mask)
                else:
                    logger.info(f"   ❌ Rejected mask {i} with score {score:.3f}")
            
            if not all_detected_masks:
                logger.info("   ⚠️ No masks passed the confidence threshold. Returning empty list.")
            
            # Note: The calling function (run) will need to handle this list of masks.
            return all_detected_masks
            
        except Exception as e:
            logger.info(f"❌ Decoder failed: {e}")
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
                logger.info("🔍 No prompts available")
            return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
        
        logger.info(f"🔍 Decoding masks:")
        logger.info(f"   Image embeddings shape: {image_embeddings.shape}")
        logger.info(f"   Point coords shape: {point_coords.shape}")
        logger.info(f"   Point coords: {point_coords}")
        logger.info(f"   Point labels shape: {point_labels.shape}")
        logger.info(f"   Point labels: {point_labels}")
        
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
            logger.info(f"   Added mask_input: {mask_input.shape}")
            
        if 'has_mask_input' in self.decoder_input_names:
            decoder_inputs['has_mask_input'] = np.array([0], dtype=np.float32)
            logger.info(f"   Added has_mask_input: [0]")
        
        logger.info(f"   Decoder input names: {list(decoder_inputs.keys())}")
        
        try:
            # Run decoder - get ALL outputs to see what's available
            start_time = time.time()
            all_outputs = self.decoder_session.run(None, decoder_inputs)  # Get all outputs
            decode_time = time.time() - start_time
            
            logger.info(f"✅ Decoder successful in {decode_time*1000:.2f}ms")
            logger.info(f"   Number of outputs: {len(all_outputs)}")
            scores_out, masks_out = all_outputs
            
            # Analyze all outputs
            for i, output in enumerate(all_outputs):
                output_name = self.decoder_outputs[i] if i < len(self.decoder_outputs) else f"output_{i}"
                logger.info(f"   Output {i} ({output_name}): shape {output.shape}, range [{output.min():.3f}, {output.max():.3f}]")
            
            # Find the mask output - prioritize low_res_masks over iou_predictions
            mask_output = None
            mask_idx = -1
            
            # Look for mask outputs (should have spatial dimensions)
            for i, output in enumerate(all_outputs):
                output_name = self.decoder_outputs[i] if i < len(self.decoder_outputs) else f"output_{i}"
                
                # Skip IoU predictions (typically shape like (1, N) where N is small)
                if 'iou' in output_name.lower() or (len(output.shape) == 2 and output.shape[1] <= 4):
                    logger.info(f"   Skipping {output_name} (appears to be IoU/confidence scores)")
                    continue
                
                # Look for mask-like outputs (3D or 4D with spatial dimensions)
                if len(output.shape) >= 3:
                    mask_output = output
                    mask_idx = i
                    logger.info(f"   ✅ Using {output_name} as mask output: {output.shape}")
                    break
            
            if mask_output is None:
                logger.info("   ❌ No suitable mask output found!")
                return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
            
            # Handle the mask output shape: (1, 4, 256, 256) -> need to select best mask
            if len(mask_output.shape) == 4 and mask_output.shape[1] > 1:
                logger.info(f"   🔍 Multiple masks detected: {mask_output.shape[1]} masks")
                
                # For NanoSAM, typically the masks are ranked by quality
                # Use the first mask (index 0) which should be the best
                best_mask = mask_output[0, 0]  # Shape: (256, 256)
                logger.info(f"   Selected best mask (index 0): {best_mask.shape}, range [{best_mask.min():.3f}, {best_mask.max():.3f}]")
                
                # Resize to target resolution if needed
                if best_mask.shape != self.input_size:
                    best_mask_resized = cv2.resize(best_mask, self.input_size)
                    logger.info(f"   Resized mask to: {best_mask_resized.shape}")
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
            
            logger.info(f"   Final mask shape: {mask_output.shape}")
            
            # Check for valid masks
            mask_data = mask_output[0, 0] if len(mask_output.shape) == 4 else mask_output
            
            # Count pixels above different thresholds
            thresholds = [0.0, 0.1, 0.5, 0.9]
            for thresh in thresholds:
                count = np.sum(mask_data > thresh)
                total_pixels = mask_data.size
                logger.info(f"   Pixels > {thresh}: {count} ({count/total_pixels*100:.1f}%)")
            
            # If mask seems empty, try negative threshold (some models output negative values for background)
            if np.sum(mask_data > 0) == 0:
                logger.info("   🔍 Checking negative thresholds...")
                for thresh in [-0.9, -0.5, -0.1]:
                    count = np.sum(mask_data < thresh)
                    logger.info(f"   Pixels < {thresh}: {count} ({count/mask_data.size*100:.1f}%)")
            
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
                logger.info(f"   💾 Saved debug mask: debug_mask_{self.frame_count:04d}.png")
            
            return mask_output
            
        except Exception as e:
            logger.info(f"❌ Decoder failed: {e}")
            return np.zeros((1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for different prompt modes
        - Single-click
        - Double-click
        """
        # Add debugging for ALL mouse events
        event_names = {
            cv2.EVENT_MOUSEMOVE: "MOUSEMOVE",
            cv2.EVENT_LBUTTONDOWN: "LBUTTONDOWN", 
            cv2.EVENT_LBUTTONUP: "LBUTTONUP",
            cv2.EVENT_RBUTTONDOWN: "RBUTTONDOWN",
            cv2.EVENT_RBUTTONUP: "RBUTTONUP",
            cv2.EVENT_LBUTTONDBLCLK: "LBUTTONDBLCLK",
            cv2.EVENT_RBUTTONDBLCLK: "RBUTTONDBLCLK"
        }

        # Print ALL events except mouse move (too verbose)
        if event != cv2.EVENT_MOUSEMOVE:
            event_name = event_names.get(event, f"UNKNOWN({event})")
            logger.info(f"MOUSE EVENT: {event_name} at ({x}, {y}), current mode: {self.current_mode}")

        # Handle Double-click for tracking
        if event == cv2.EVENT_LBUTTONDBLCLK:
            logger.info(f"Double-click detected at ({x}, {y}) - Initializing tracking")
            
            # Use the stored current frame instead of param
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                image_pil = PIL.Image.fromarray(image_rgb)
                
                # Initialize tracking
                try:
                    self.tracked_mask = self.tracker.init(image_pil, point=(x, y))
                    self.tracked_point = (x, y)
                    self.tracking_active = True
                    logger.info("Tracking initialized successfully")
                except Exception as e:
                    logger.info(f"Failed to initialize tracking: {e}")
            
            return  # EXIT HERE - Don't process other mouse events when double-clicking
        
        # Only process single clicks if NOT tracking
        if self.tracking_active:
            return  # Ignore all other mouse events when tracking is active
        
        # Handle Single-clip prompts
        if self.current_mode == PromptMode.POINT:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.positive_points.append((x, y))
                logger.info(f"Added positive point: ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.negative_points.append((x, y))
                logger.info(f"Added negative point: ({x}, {y})")
                
        elif self.current_mode == PromptMode.BBOX:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.bbox_start = (x, y)
                self.bbox_end = None
                self.drawing_bbox = True
                logger.info(f"Started bbox at: ({x}, {y})")
                
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing_bbox:
                self.bbox_end = (x, y)
                
            elif event == cv2.EVENT_LBUTTONUP and self.drawing_bbox:
                self.bbox_end = (x, y)
                self.drawing_bbox = False
                logger.info(f"Completed bbox: {self.bbox_start} to {self.bbox_end}")
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time  
    
    def overlay_masks(self, image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5) -> np.ndarray:
        """Overlay multiple segmentation masks on image"""
        result = image.copy()
        
        logger.info(f"🎨 Overlaying masks:")
        logger.info(f"   Input image shape: {image.shape}")
        logger.info(f"   Input is a list of {len(masks)} masks.") # Correct way to check the input
        
        if len(masks.shape) == 4:
            masks = masks[0]  # Remove batch dimension
            logger.info(f"   Masks after batch removal: {masks.shape}")
        
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
            logger.info(f"   Processing mask {i}: shape {mask.shape}, range [{mask.min():.3f}, {mask.max():.3f}]")
            
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask, (w, h))
                logger.info(f"   Resized mask to: {mask_resized.shape}")
            else:
                mask_resized = mask
            
            # Try different thresholds to see what works
            # For SAM models, masks can have different value ranges
            thresholds = [0.0, 0.1, 0.3, 0.5]
            
            # Also try adaptive threshold based on mask statistics
            if mask_resized.max() > mask_resized.min():
                adaptive_thresh = mask_resized.mean() + 0.5 * mask_resized.std()
                thresholds.append(adaptive_thresh)
                logger.info(f"   Added adaptive threshold: {adaptive_thresh:.3f}")
            
            for threshold in thresholds:
                mask_binary = (mask_resized > threshold).astype(np.uint8)
                pixel_count = np.sum(mask_binary)
                logger.info(f"   Threshold {threshold:.3f}: {pixel_count} pixels ({pixel_count/(h*w)*100:.1f}%)")
                
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
                    logger.info(f"   ✅ Applied mask {i} with threshold {threshold:.3f}, color {color}")
                    break
        
        if not mask_applied:
            logger.info("   ⚠️  No masks were applied (all below threshold)")
        
        return result
    
    def overlay_bboxes(self, image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Overlay bounding boxes on segmented regions."""
        result = image.copy()
        
        logger.info(f"📦 Overlaying bounding boxes:")
        logger.info(f"   Input is a list of {len(masks)} masks.") # Correct way to check the input
        
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
                logger.info(f"   Skipping mask {i} as it is empty.")
                continue

            # Assuming the masks in the list are 2D arrays (H, W)
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask
            
            mask_binary = (mask_resized > 0).astype(np.uint8) * 255 
            
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.info(f"   No contours found for mask {i}.")
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            
            x, y, w_bbox, h_bbox = cv2.boundingRect(largest_contour)
            
            color = colors[i % len(colors)]
            cv2.rectangle(result, (x, y), (x + w_bbox, y + h_bbox), color, 2)
            bbox_drawn = True
            logger.info(f"   ✅ Drawn bbox for mask {i} with color {color}")

        if not bbox_drawn:
            logger.info("   ⚠️  No bounding boxes were drawn.")
        
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
    
    def draw_tracking(self, image: np.ndarray) -> np.ndarray:
        """Draw tracking results compatible with OnnxTracker"""
        result = image.copy()
        
        # Draw tracked mask
        if self.tracked_mask is not None and self.tracking_active:
            try:
                logger.info(f"[draw_tracking] Drawing tracked mask: type={type(self.tracked_mask)}")
                
                # Handle different mask formats
                if isinstance(self.tracked_mask, torch.Tensor):
                    mask_np = self.tracked_mask.detach().cpu().numpy()
                else:
                    mask_np = self.tracked_mask
                
                logger.info(f"[draw_tracking] Mask shape: {mask_np.shape}")
                
                # Remove batch/channel dimensions if present
                if len(mask_np.shape) == 4:
                    mask_np = mask_np[0, 0]
                elif len(mask_np.shape) == 3:
                    mask_np = mask_np[0]
                elif len(mask_np.shape) == 2:
                    pass  # Already 2D
                else:
                    logger.info(f"[draw_tracking] Unexpected mask shape: {mask_np.shape}")
                    return result
                
                h, w = image.shape[:2]
                
                # Resize mask to match frame
                if mask_np.shape != (h, w):
                    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    mask_resized = mask_np
                
                logger.info(f"[draw_tracking] Mask range: [{mask_resized.min():.3f}, {mask_resized.max():.3f}]")
                
                # Create binary mask with proper threshold
                # Use 0.0 as threshold since SAM outputs are typically > 0 for foreground
                binary_mask = (mask_resized > 0.0).astype(np.uint8)
                
                # Count pixels for debugging
                pixel_count = np.sum(binary_mask)
                coverage = pixel_count / (h * w) * 100
                logger.info(f"[draw_tracking] Binary mask: {pixel_count} pixels ({coverage:.1f}% coverage)")
                
                if pixel_count > 0:  # Only draw if mask has content
                    # Create colored overlay (green for tracking)
                    overlay = result.copy()
                    overlay[binary_mask == 1] = [0, 255, 0]  # Green in BGR
                    
                    # Blend with result
                    alpha = 0.3  # Make it semi-transparent
                    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
                    logger.info(f"[draw_tracking] Applied green mask overlay")
                else:
                    logger.info(f"[draw_tracking] No pixels to draw in mask")
                    
            except Exception as e:
                logger.info(f"[draw_tracking] Failed to draw tracking mask: {e}")
                import traceback
                traceback.print_exc()
        
        # Draw tracked point (similar to draw_prompts style)
        if self.tracked_point is not None and self.tracking_active:
            x, y = self.tracked_point
            x, y = int(x), int(y)  # Guard against floats
            # Draw filled circle with white outline (consistent with prompts)
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)  # Green filled
            cv2.circle(result, (x, y), 7, (255, 255, 255), 2)  # White outline
            logger.info(f"[draw_tracking] Drew tracking point at ({x}, {y})")
        
        return result

    def clear_prompts(self):
        """Clear all prompts"""
        self.positive_points = []
        self.negative_points = []
        self.tracked_point = None
        self.bbox_start = None
        self.bbox_end = None
        self.drawing_bbox = False
        self.everything_mode_active = False
        logger.info("Cleared all prompts")
    
    def run(self):
        """Run webcam segmentation"""
        logger.info("Starting NanoSAM live webcam segmentation...")
        
        window_name = 'NanoSAM Webcam'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
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
                
                # Set the frame as parameter for mouse callback (for double-click tracking)
                self.current_frame = frame

                # Handle tracking updates
                if self.tracking_active and self.tracker.token is not None:
                    try:
                        # Convert frame to PIL for tracker
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_pil = PIL.Image.fromarray(image_rgb)
                        
                        # Update tracking
                        mask_result, point_result = self.tracker.update(image_pil)
                        if mask_result is not None and point_result is not None:
                            self.tracked_mask = mask_result
                            self.tracked_point = point_result
                            # logger.info(f"Tracking updated - point: {self.tracked_point}")
                        else:
                            logger.info("Tracking update failed: no results returned")
                            self.tracking_active = False
                    except Exception as e:
                        logger.info(f"Tracking update failed: {e}")
                        self.tracking_active = False

                current_time = time.time()
                
                # Update image embeddings periodically
                if current_time - last_embedding_time > embedding_interval:
                    image_embeddings = self.encode_image(frame)
                    last_embedding_time = current_time
                
                result_frame = frame.copy()     # copy initial frame to draw on later
                
                # Get prompts based on current mode
                point_coords, point_labels = None, None
                    
                if self.current_mode == PromptMode.EVERYTHING and image_embeddings is not None:
                    point_coords, point_labels = self.create_everything_prompts(frame.shape)
                    # Visualize the grid points
                    if point_coords is not None:
                        h, w = frame.shape[:2]
                        model_h, model_w = self.input_size
                        for point in point_coords.squeeze(0).astype(int):
                            frame_x = int(point[0] * w / model_w)
                            frame_y = int(point[1] * h / model_h)
                            cv2.circle(result_frame, (frame_x, frame_y), 2, (255, 255, 0), -1)
                elif self.current_mode == PromptMode.HEADING:
                    logger.info("PromptMode.HEADING...")
                    point_coords, point_labels = self.set_point_prompt(frame.shape)
                elif self.current_mode == PromptMode.POINT:
                    point_coords, point_labels = self.create_point_prompts(frame.shape)
                elif self.current_mode == PromptMode.BBOX:
                    point_coords, point_labels = self.create_bbox_prompts(frame.shape)
                elif self.current_mode == PromptMode.TRACKING:
                    # In TRACKING mode, don't generate prompts for static segmentation
                    # The tracking system handles its own segmentation
                    point_coords, point_labels = None, None
                    # logger.info("TRACKING mode - no static prompts generated")

                # Perform static segmentation
                if point_coords is not None and image_embeddings is not None:
                    # Start Inference Timing
                    inference_start_time = time.time()

                    logger.info(f"\n🚀 Running segmentation - Mode: {self.current_mode.value}")
                    masks_list = self.decode_masks(image_embeddings, point_coords, point_labels)
                    
                    # End Inference Timing
                    inference_time_s = time.time() - inference_start_time
                    inference_fps = 1.0 / inference_time_s if inference_time_s > 0 else 0
                    inference_latency_ms = inference_time_s * 1000

                    logger.info(f"✅ Got {len(masks_list)} masks to display.")
                    
                    if masks_list:
                        if self.show_bbox:
                            result_frame = self.overlay_bboxes(image=result_frame, masks=masks_list)
                        else:
                            result_frame = self.overlay_masks(image=result_frame, masks=masks_list)
                    else:
                        logger.info(f"❌ No valid masks to display.")
                else:
                    if point_coords is None and self.no_prompt_warning_counter % 30 == 1:
                        logger.info("🔍 No prompts available")
                        inference_fps = 0.0
                        inference_latency_ms = 0.0
                    if image_embeddings is None:
                        logger.info("🔍 No image embeddings available")
                        inference_fps = 0.0
                        inference_latency_ms = 0.0
                
                # Draw prompts and tracking
                result_frame = self.draw_prompts(result_frame)
                result_frame = self.draw_tracking(result_frame)
                
                # Update and display info
                self.update_fps()
                
                # ---- Webcam Text Status (top right corner) ------------- #
                
                mode_text = f"Mode: {self.current_mode.value.upper()}"
                cv2.putText(result_frame, mode_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                fps_text = f'FPS: {self.current_fps:.1f}'
                cv2.putText(result_frame, fps_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                inference_text = f'Inference: {inference_latency_ms:.1f} ms ({inference_fps:.1f} FPS)'
                cv2.putText(result_frame, inference_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Tracking status
                if self.tracking_active:
                    tracking_text = "Tracking: ACTIVE"
                    cv2.putText(result_frame, tracking_text, (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Point counts
                if self.current_mode == PromptMode.POINT:
                    point_text = f'Points: +{len(self.positive_points)} -{len(self.negative_points)}'
                    y_pos = 150 if self.tracking_active else 120
                    cv2.putText(result_frame, point_text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Instructions
                instructions = [
                    "1:Point 2:BBox 3:Everything C:Clear T:StopTrack D:Debug S:SaveMasks Q:Quit R:Reset",
                    "Double-click anywhere to start tracking"
                ]
                
                y_offset = result_frame.shape[0] - 20
                for instruction in instructions:
                    cv2.putText(result_frame, instruction, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset -= 20
                
                cv2.imshow(window_name, result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.current_mode = PromptMode.POINT
                    logger.info("Switched to Point mode")
                elif key == ord('2'):
                    self.current_mode = PromptMode.BBOX
                    logger.info("Switched to Bounding Box mode")
                elif key == ord('3'):
                    self.current_mode = PromptMode.EVERYTHING
                    logger.info("Switched to Everything mode")
                elif key == ord('4'):
                    self.current_mode = PromptMode.TRACKING
                    self.clear_prompts()  # Clear any existing prompts
                    logger.info("Switched to Tracking mode - double-click to start tracking")
                elif key == ord('t'):
                    if self.tracking_active:
                        logger.info("Stopping tracking")
                        self.tracker.reset()
                        self.tracking_active = False
                        self.tracked_mask = None
                        self.tracked_point = None
                    else:
                        logger.info("Double-click to start tracking")
                elif key == ord('c'):
                    self.clear_prompts()
                elif key == ord('r'):
                    self.clear_prompts()
                    # Reset tracking as well
                    if self.tracking_active:
                        self.tracker.reset()
                        self.tracking_active = False
                        self.tracked_mask = None
                        self.tracked_point = None
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    logger.info(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('s'):
                    self.save_masks = not self.save_masks
                    logger.info(f"Save masks: {'ON' if self.save_masks else 'OFF'}")
        
        except KeyboardInterrupt:
            logger.info("\nStopping key interrupt 'q'...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# ---- MAIN ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Enhanced NanoSAM Live Segmentation')
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