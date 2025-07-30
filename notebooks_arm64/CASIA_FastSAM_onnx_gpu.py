#!/usr/bin/env python3
"""
FastSAM ONNX Inference Script
=============================
A refactored script for running FastSAM model inference using ONNX Runtime with GPU acceleration.
Supports TensorRT and CUDA execution providers with CPU fallback.

Usage:
python /home/copter/jetson_benchmark/notebooks_arm64/CASIA_FastSAM_onnx_gpu.py --model_path "/home/copter/onnx_models/CASIA-IVA-Lab_FastSAM-s.onnx" --image_path "/home/copter/jetson_benchmark/images/dogs.jpg"
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import torch
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from PIL import Image
import os
import sys
import time
from typing import List, Tuple, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastSAMONNX:
    """FastSAM ONNX model wrapper for inference with GPU acceleration."""
    
    def __init__(self, model_path: str, imgsz: int = 1024):
        """
        Initialize the FastSAM ONNX model.
        
        Args:
            model_path (str): Path to the ONNX model file
            imgsz (int): Input image size for the model (default: 1024)
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.session = None
        self._setup_environment()
        self._initialize_session()
    
    def _setup_environment(self):
        """Set up environment variables and paths."""
        # Add FastSAM utils path if needed
        fastsam_path = '/home/copter/FastSAM'
        if os.path.exists(fastsam_path) and fastsam_path not in sys.path:
            sys.path.append(fastsam_path)
        
        # Update library path for cuDNN
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        cudnn_path = '/usr/lib/aarch64-linux-gnu'
        if cudnn_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f'{cudnn_path}:{current_ld_path}'
    
    def _get_execution_providers(self) -> List[Tuple[str, dict]]:
        """
        Configure and return available execution providers in order of preference.
        
        Returns:
            List of tuples containing provider names and their options
        """
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")
        
        providers = []
        
        # TensorRT (best for Jetson)
        if 'TensorrtExecutionProvider' in available_providers:
            logger.info("✅ TensorRT provider available")
            trt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 1 * 1024 * 1024 * 1024,  # 1GB
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_timing_cache_enable': True,
            }
            providers.append(('TensorrtExecutionProvider', trt_options))
        
        # CUDA with cuDNN
        if 'CUDAExecutionProvider' in available_providers:
            logger.info("✅ CUDA provider available")
            cuda_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 1 * 1024 * 1024 * 1024,  # 1GB
                'cudnn_conv_algo_search': 'HEURISTIC',
            }
            providers.append(('CUDAExecutionProvider', cuda_options))
        
        # CPU fallback
        cpu_options = {
            'intra_op_num_threads': 6,
            'inter_op_num_threads': 1,
            'enable_cpu_mem_arena': True,
        }
        providers.append(('CPUExecutionProvider', cpu_options))
        
        provider_names = [p[0] if isinstance(p, tuple) else p for p in providers]
        logger.info(f"Provider chain: {provider_names}")
        
        return providers
    
    def _initialize_session(self):
        """Initialize the ONNX Runtime session."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        providers = self._get_execution_providers()
        
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            active_provider = self.session.get_providers()[0]
            logger.info(f"✅ Session initialized with provider: {active_provider}")
            
            # Log input/output info
            input_info = self.session.get_inputs()[0]
            logger.info(f"Model input: {input_info.name}, shape: {input_info.shape}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            raise
    
    def preprocess(self, img_origin: np.ndarray) -> np.ndarray:
        """
        Pre-process an input image for the FastSAM ONNX model.
        
        Args:
            img_origin (np.ndarray): Original input image (BGR format)
            
        Returns:
            np.ndarray: Pre-processed image ready for inference
                       Shape: (1, 3, imgsz, imgsz) in RGB format, normalized to [0, 1]
        """
        h, w = img_origin.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        
        # Create blank canvas
        inp = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        
        # Calculate new dimensions
        nw = int(w * scale)
        nh = int(h * scale)
        
        # Resize and convert BGR to RGB
        resized = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
        
        # Center the image on the canvas
        if h > w:
            # Portrait orientation
            offset = int((self.imgsz - nw) / 2)
            inp[:nh, offset:offset + nw, :] = resized
        else:
            # Landscape orientation
            offset = int((self.imgsz - nh) / 2)
            inp[offset:offset + nh, :nw, :] = resized
        
        # Normalize and transpose to NCHW format
        rgb = np.array([inp], dtype=np.float32) / 255.0
        return np.transpose(rgb, (0, 3, 1, 2))
    
    def postprocess(self, preds: List[np.ndarray], img: np.ndarray, orig_imgs: np.ndarray, 
                   retina_masks: bool = True, conf: float = 0.25, iou: float = 0.45, 
                   agnostic_nms: bool = False) -> List[Results]:
        """
        Post-process raw ONNX model predictions to generate segmentation masks and bounding boxes.
        
        Args:
            preds (List[np.ndarray]): Raw predictions from the ONNX model
            img (np.ndarray): Pre-processed input image
            orig_imgs (np.ndarray): Original un-preprocessed image(s)
            retina_masks (bool): Whether to use retina masks (more detailed)
            conf (float): Confidence threshold for object detection
            iou (float): IoU threshold for Non-Maximum Suppression
            agnostic_nms (bool): Whether to perform class-agnostic NMS
            
        Returns:
            List[Results]: Processed bounding boxes and masks
        """
        # Apply Non-Maximum Suppression
        p = ops.non_max_suppression(
            preds[0], conf, iou, agnostic_nms, max_det=100, nc=1
        )
        
        results = []
        # Handle different ONNX output structures
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
        
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            img_path = "inference_output"  # Placeholder
            
            if not len(pred):  # No detections
                results.append(Results(
                    orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]
                ))
                continue
            
            # Process masks
            if retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(
                    proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2]
                )
            else:
                masks = ops.process_mask(
                    proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True
                )
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            results.append(Results(
                orig_img=orig_img, path=img_path, names="segment", 
                boxes=pred[:, :6], masks=masks
            ))
        
        return results
    
    def predict(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45, 
                retina_masks: bool = True, verbose: bool = True) -> List[Results]:
        """
        Run inference on an image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            retina_masks (bool): Use retina masks
            verbose (bool): Print timing information
            
        Returns:
            List[Results]: Inference results with masks and boxes
        """
        if self.session is None:
            raise RuntimeError("Model session not initialized")
        
        start_time = time.time()
        
        # Preprocess
        preprocessed = self.preprocess(image)
        if verbose:
            logger.info(f"Preprocessing completed in {time.time() - start_time:.3f}s")
        
        # Run inference
        inference_start = time.time()
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: preprocessed})
        
        inference_time = time.time() - inference_start
        if verbose:
            logger.info(f"Inference completed in {inference_time:.3f}s")
        
        # Postprocess
        postprocess_start = time.time()
        results = self.postprocess(
            outputs, preprocessed, image, retina_masks, conf, iou
        )
        
        if verbose:
            postprocess_time = time.time() - postprocess_start
            total_time = time.time() - start_time
            logger.info(f"Postprocessing completed in {postprocess_time:.3f}s")
            logger.info(f"Total processing time: {total_time:.3f}s")
        
        return results


def load_image(image_path: str) -> np.ndarray:
    """Load and validate an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    logger.info(f"Loaded image: {image_path}, shape: {img.shape}")
    return img


def visualize_results(image: np.ndarray, results: List[Results], 
                     save_path: Optional[str] = None, show: bool = True):
    """
    Visualize inference results.
    
    Args:
        image (np.ndarray): Original image in BGR format
        results (List[Results]): Inference results
        save_path (Optional[str]): Path to save the visualization
        show (bool): Whether to display the image
    """
    plt.figure(figsize=(12, 8))
    
    # Convert BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    
    # Add results visualization here if needed
    # This is a placeholder - you can extend this based on your needs
    
    plt.axis('on')
    plt.title(f"FastSAM Inference Results ({len(results)} detections)")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="FastSAM ONNX Inference")
    parser.add_argument("--model_path", required=True, help="Path to ONNX model file")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--retina_masks", action="store_true", help="Use retina masks")
    parser.add_argument("--no_show", action="store_true", help="Don't display results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize model
        logger.info("Initializing FastSAM ONNX model...")
        model = FastSAMONNX(args.model_path, imgsz=args.imgsz)
        
        # Load image
        logger.info(f"Loading image: {args.image_path}")
        image = load_image(args.image_path)
        
        # Run inference
        logger.info("Running inference...")
        results = model.predict(
            image, 
            conf=args.conf, 
            iou=args.iou, 
            retina_masks=args.retina_masks
        )
        
        logger.info(f"Inference completed. Found {len(results)} results.")
        
        # Visualize results
        output_path = os.path.join(args.output_dir, "inference_result.png")
        visualize_results(image, results, save_path=output_path, show=not args.no_show)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())