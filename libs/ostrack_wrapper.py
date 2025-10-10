"""
Production wrapper for OSTrack

requires:
- onnx: 1.16.1 (or higher)
- onnxconverter-common: 1.16.0 (or higher)
- onnxruntime-gpu: 1.20.0 stays untouched (Jetson JP6.1 specific)
- tensorrt 10.3.0
"""

import os
import numpy as np
from contextlib import nullcontext
import torch
try:
    # If some other code set this to bf16, put it back.
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
except Exception:
    pass
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tempfile
import shutil
from collections import deque
from enum import Enum
import time
import onnxruntime as ort
from easydict import EasyDict as edict
import logging
import sys
from collections import OrderedDict, deque
from typing import Tuple, Optional, List
import argparse
from pathlib import Path
from tqdm import tqdm

# Add these imports to the top of your script if they are not already there
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Add OSTrack to Python path
ostrack_path_str = "/home/copter/EdgeTAM/OSTrack"
sys.path.append(ostrack_path_str)

# --- OSTrack Imports ---
# from lib.models import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file
from lib.test.evaluation.tracker import Tracker 
from lib.models.ostrack.ostrack import OSTrack as OSTrackOfficial, build_ostrack

class OSTrackWrapper:
    """
    Wrapper for the OSTrack model that provides a stable API 
    for single-object tracking, supporting multiple backends (PyTorch, ONNX, TensorRT).
    
    API Features
    - Public API: __init__, tracker_init, __call__, reset, is_lost
    - BBox Format: Consistently uses pixel xywh (x, y, width, height).
    - Lost-Track Logic: Internalized and exposed via the `is_lost` property.
    - Backend Agnostic: Automatically selects backend based on model file extension.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        template_size: int = 128,
        search_size: int = 256,
        score_thresh: float = 0.7,
        min_box_ratio: float = 0.05, # min_box_size = min_box_ratio * search_size
        verbose: bool = False,
    ):
        """
        Initializes the tracker and loads the model into the specified backend.
        
        Args:
            model_path (str): Path to the model file (.pt, .onnx, or .engine), absolute path preferred
            device (str): Device to run on (e.g., "cuda:0").
            template_size (int): The size of the template image patch.
            search_size (int): The size of the search image patch.
            score_thresh (float): Confidence score below which the track is considered lost.
            min_box_ratio (float): A ratio of the search size. If predicted box width or height 
                                   is smaller than this, the track is considered lost.
            verbose (str): Flag for verbose logging of inference time of the {model_type} model
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.template_size = template_size
        self.search_size = search_size
        self.score_thresh = score_thresh
        self.min_box_threshold = min_box_ratio * search_size
        
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        # OSTrack search/template factors
        self.template_factor = 2.0
        self.search_factor = 4.0
        
        # Logging states
        self.verbose = verbose
        self._inference_times: List[float] = [] 
        self.backend_name = "Unknown"

        # Internal state
        self._backend_session = None
        self._inference_fn = None
        self.reset()

        # Backend factory: Load model based on file extension
        if model_path.endswith(".engine"):
            self.backend_name = "TensorRT"
            self._load_tensorrt_backend(model_path)
            logging.info("✅ OSTrack TensorRT backend loaded successfully.")
        elif model_path.endswith(".onnx"):
            self.backend_name = "ONNX"
            self._load_onnx_backend(model_path)
            logging.info("✅ OSTrack ONNX backend loaded successfully.")
        elif model_path.endswith(".pt") or model_path.endswith(".pth"):
            # Placeholder for PyTorch backend
            self.backend_name = "Pytorch"
            self._load_pytorch_backend(model_path)
            logging.info("✅ OSTrack PyTorch backend loaded successfully.")
        else:
            raise ValueError(f"Unsupported model file type: {model_path}")

    def reset(self):
        """Resets the tracker's state, clearing the template and tracking data."""
        self.initialized = False
        self.is_lost_flag = True  # A non-initialized tracker is considered "lost"
        self.template_tensor = None
        self.target_pos = None  # Center position (cx, cy)
        self.target_sz = None   # Size (w, h)
        logging.info("Tracker state has been reset.")

    def tracker_init(self, image: np.ndarray, bbox: Tuple[int, int, int, int]=None, mask: Optional[np.ndarray] = None):
        """
        Initializes the tracker with the first frame and a bounding box.
        
        Args:
            image (np.ndarray): The initial frame (BGR, HxWxC, uint8).
            bbox (Tuple[int, int, int, int]): The initial bounding box in xywh format.
        """
        self.reset()

        if mask is not None:
            bbox = self._bbox_from_mask(mask)
        
        if bbox is None:
            logger.error("No bbox or mask provided for initialization")
            return False
        
        # default is to get a bbox_xywh
        x, y, w, h = bbox
        
        # Set initial target position and size
        self.target_pos = np.array([x + w / 2, y + h / 2])
        self.target_sz = np.array([w, h])
        
        # Preprocess the template from the initial frame
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.template_tensor = self._preprocess(frame_rgb, self.target_pos, 
                                                self.target_sz * self.template_factor, 
                                                self.template_size)
        
        self.initialized = True
        self.is_lost_flag = False
        logging.info(f"Tracker initialized with bbox (xywh): {bbox}")
        return True

    def __call__(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
            """
            Updates the tracker with a new frame and returns the new bounding box.
            
            Args:
                image (np.ndarray): The current frame (BGR, HxWxC, uint8).
                
            Returns:
                Optional[Tuple[int, int, int, int]]: The new bounding box in xywh format,
                                                    or None if the track is lost.
            """
            if not self.initialized:
                logging.warning("Tracker not initialized. Call tracker_init() first.")
                self.is_lost_flag = True
                return None

            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess the search region
            search_tensor = self._preprocess(frame_rgb, self.target_pos,
                                            self.target_sz * self.search_factor,
                                            self.search_size)
            
            # --- START: Performance Logging Logic ---
            if self.verbose:
                start_time = time.perf_counter()
                outputs = self._inference_fn(self.template_tensor, search_tensor)
                inference_ms = (time.perf_counter() - start_time) * 1000
                self._inference_times.append(inference_ms)
                
                if len(self._inference_times) >= 100:
                    avg_time = np.mean(self._inference_times)
                    logging.info(f"⏱️ {self.backend_name} inference: {avg_time:.2f}ms avg (last 100 frames)")
                    self._inference_times.clear() # Reset for the next batch
            else:
                # Perform inference without timing
                outputs = self._inference_fn(self.template_tensor, search_tensor)
            # --- END: Performance Logging Logic ---
            
            # Postprocess the outputs to get the new bbox and score
            pred_box_normalized = outputs['pred_boxes'][0, 0]
            score = float(np.mean(outputs.get('conf_scores', [self.score_thresh])[0, 0]))

            # Lost-Track Logic
            bbox_xywh = self._postprocess(pred_box_normalized, frame_rgb.shape)
            px, py, pw, ph = bbox_xywh
            
            if score < self.score_thresh or pw < self.min_box_threshold or ph < self.min_box_threshold:
                self.is_lost_flag = True
                logging.warning(f"Track lost. Score: {score:.2f} | Size: ({pw}, {ph})")
                return None
            
            # If track is successful, update state
            self.is_lost_flag = False
            self.target_pos = np.array([px + pw / 2, py + ph / 2])
            self.target_sz = np.array([pw, ph])
            
            return bbox_xywh

    @property
    def is_lost(self) -> bool:
        """Returns True if the track is considered lost, False otherwise."""
        return self.is_lost_flag

    # --- Backend Loading Methods ---

    def _load_tensorrt_backend(self, engine_path: str):
        """Loads a TensorRT engine and sets up the inference context."""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()

        # Allocate buffers
        inputs, outputs, bindings, stream = {}, {}, [], cuda.Stream()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = engine.get_tensor_shape(name)
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            buffer_info = {'host': host_mem, 'device': device_mem, 'shape': shape}
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs[name] = buffer_info
            else:
                outputs[name] = buffer_info
        
        self._backend_session = {'context': context, 'inputs': inputs, 'outputs': outputs, 'bindings': bindings, 'stream': stream}

        def inference_fn(template, search):
            # Assumes input names are 'template' and 'search'
            np.copyto(inputs['template']['host'], template.ravel())
            np.copyto(inputs['search']['host'], search.ravel())
            
            # Transfer input data to the GPU
            for inp in inputs.values():
                cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
            # Transfer predictions back from the GPU
            for out in outputs.values():
                cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
            
            stream.synchronize()
            
            # Reshape and return
            return {name: out['host'].reshape(out['shape']) for name, out in outputs.items()}
        
        self._inference_fn = inference_fn

    def _load_onnx_backend(self, onnx_path: str):
        """Loads an ONNX model into an ONNX Runtime session."""
        # --- START: Added SessionOptions Logic ---
        # Configure session options for performance and thread safety
        session_options = ort.SessionOptions() #
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL #
        
        # Explicitly set thread count to avoid affinity warnings and ensure deterministic behavior
        session_options.intra_op_num_threads = 1 #
        session_options.inter_op_num_threads = 1 #
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL #
        # --- END: Added SessionOptions Logic ---

        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Pass the configured session_options to the InferenceSession constructor
        self._backend_session = ort.InferenceSession(
            onnx_path, 
            sess_options=session_options, # Pass the options here
            providers=providers
        )
        
        # Check which provider is actually being used
        actual_provider = self._backend_session.get_providers()[0]
        logging.info(f"ONNX Runtime is using provider: {actual_provider}")

        def inference_fn(template, search):
            outputs = self._backend_session.run(None, {'template': template, 'search': search})
            # Standardize output format to a dictionary
            return {'pred_boxes': outputs[0], 'conf_scores': outputs[1]}
        
        self._inference_fn = inference_fn
        
    def _load_pytorch_backend(self, model_path: str):
        """Placeholder for loading a PyTorch model."""
        # This is where you would load your PyTorch model
        # from lib.models import build_ostrack
        # from lib.config.ostrack.config import cfg, update_config_from_file
        # self._backend_session = build_ostrack(...)
        # self._backend_session.load_state_dict(torch.load(model_path, map_location=self.device))
        # self._backend_session.eval().to(self.device)
        
        # def inference_fn(template, search):
        #     with torch.no_grad():
        #         template_torch = torch.from_numpy(template).to(self.device)
        #         search_torch = torch.from_numpy(search).to(self.device)
        #         raw_outputs = self._backend_session(template_torch, search_torch)
        #     return {k: v.cpu().numpy() for k, v in raw_outputs.items()}
            
        # self._inference_fn = inference_fn
        raise NotImplementedError("PyTorch backend is not yet implemented in this wrapper.")

    # --- Pre/Post-processing Helpers ---

    def _preprocess(self, image_rgb: np.ndarray, center: np.ndarray, size: np.ndarray, output_size: int) -> np.ndarray:
        """Crops, resizes, and normalizes an image patch."""
        # 1. Crop image patch
        crop = self._crop_image(image_rgb, center, size, output_size)
        
        # 2. Convert to tensor format (1, C, H, W) and normalize
        tensor = crop.transpose(2, 0, 1)
        tensor = tensor[np.newaxis, ...].astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor.astype(np.float32)

    def _postprocess(self, pred_box_norm: np.ndarray, frame_shape: Tuple) -> Tuple[int, int, int, int]:
        """Converts normalized predicted box to absolute pixel xywh format."""
        search_area_sz = self.target_sz * self.search_factor
        
        # Denormalize with respect to the search area
        cx = pred_box_norm[0] * search_area_sz[0]
        cy = pred_box_norm[1] * search_area_sz[1]
        w = pred_box_norm[2] * search_area_sz[0]
        h = pred_box_norm[3] * search_area_sz[1]
        
        # Offset by the search area's center
        search_center = self.target_pos
        cx += search_center[0] - search_area_sz[0] / 2
        cy += search_center[1] - search_area_sz[1] / 2
        
        # Convert center-based to top-left-based (xywh)
        x = cx - w / 2
        y = cy - h / 2
        
        # Clip to frame boundaries
        img_h, img_w = frame_shape[:2]
        x = int(np.clip(x, 0, img_w))
        y = int(np.clip(y, 0, img_h))
        w = int(np.clip(w, 1, img_w - x))
        h = int(np.clip(h, 1, img_h - y))
        
        return (x, y, w, h)

    def _crop_image(self, img: np.ndarray, center: np.ndarray, size: np.ndarray, output_size: int) -> np.ndarray:
        """Crops a patch from the image with padding if necessary."""
        cx, cy = center
        w, h = size
        
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        
        img_h, img_w = img.shape[:2]
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        crop = img[y1:y2, x1:x2]
        
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
        
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    def _bbox_from_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract bounding box from binary mask."""
        if mask is None or mask.sum() == 0:
            return None
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox_xywh = np.array([x, y, w, h])
        return bbox_xywh
