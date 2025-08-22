#!/usr/bin/env python3
"""
NanoSAM ONNX Inference Script
=============================
A refactored script for running FastSAM model inference using ONNX Runtime with GPU acceleration.
Supports TensorRT and CUDA execution providers with CPU fallback.

Usage:
python /home/copter/jetson_benchmark/notebooks_arm64/nanosam_video_demo_onnx.py
"""

import argparse
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import onnx
import tensorrt     # import even if not used
import torch
from PIL import Image
import os
import sys
import time
from typing import List, Tuple, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_versions():
    print(f"ONNX version: {onnx.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"TensorRT version: {tensorrt.__version__}")
    print("Available ONNX Runtime providers: ", ort.get_available_providers())

def list_onnx_sessions():
    """Find all ONNX InferenceSession objects in current namespace"""
    sessions = {}
    
    # Check global variables
    for name, obj in globals().items():
        if isinstance(obj, ort.InferenceSession):
            try:
                providers = obj.get_providers()
                inputs = [inp.name for inp in obj.get_inputs()]
                outputs = [out.name for out in obj.get_outputs()]
                sessions[name] = {
                    'object': obj,
                    'providers': providers,
                    'inputs': inputs,
                    'outputs': outputs
                }
            except:
                sessions[name] = {'object': obj, 'status': 'invalid/corrupted'}
    
    # Check garbage collector for unreferenced sessions
    unreferenced_sessions = []
    for obj in gc.get_objects():
        if isinstance(obj, ort.InferenceSession):
            # Check if this session is not in our named sessions
            if obj not in [s.get('object') for s in sessions.values()]:
                try:
                    providers = obj.get_providers()
                    unreferenced_sessions.append({
                        'id': id(obj),
                        'providers': providers,
                        'status': 'unreferenced'
                    })
                except:
                    unreferenced_sessions.append({
                        'id': id(obj),
                        'status': 'unreferenced/corrupted'
                    })
    
    return sessions, unreferenced_sessions

def cleanup_all_onnx_sessions():
    """Clean up ALL ONNX sessions (named and unreferenced)"""
    sessions, unreferenced = list_onnx_sessions()
    
    # Clean up named sessions
    for name in list(sessions.keys()):
        try:
            if name in globals():
                del globals()[name]
                print(f"Deleted session: {name}")
        except Exception as e:
            print(f"Error deleting {name}: {e}")
    
    # Force garbage collection to clean unreferenced sessions
    collected = gc.collect()
    print(f"Garbage collected {collected} objects")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")

def load_image(image_path: str) -> np.ndarray:
    """Load and validate an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    logger.info(f"Loaded image: {image_path}, shape: {img.shape}")
    return img

def load_example_image():
    image_path = "/home/copter/jetson_benchmark/images/dogs.jpg"
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1024, 1024), resample=Image.BILINEAR)
    w, h = img.size 
    logger.info(f"Loaded image: {image_path}, resized to: {w}x{h}")
    return img

def set_onnx_conf():
    print("--- Checking Available Execution Providers ---")
    # available_providers = ort.get_available_providers() 

    providers = [
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]

    # Configure session options
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    # session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return providers, session_options

def load_new_onnx_sessions(
        encoder_model_path, 
        decoder_model_path, 
        providers,
        session_options=None,
    ):

    enc_sess = None
    dec_sess = None

    try:
        if session_options != None:
            enc_sess = ort.InferenceSession(encoder_model_path, sess_options=session_options, providers=providers)
            dec_sess = ort.InferenceSession(decoder_model_path, sess_options=session_options, providers=providers)
        else:
            enc_sess = ort.InferenceSession(encoder_model_path, sess_options=session_options)
            dec_sess = ort.InferenceSession(decoder_model_path, sess_options=session_options)


        # Verify which provider was actually chosen by ONNX Runtime
        if 'TensorrtExecutionProvider' in enc_sess.get_providers() and 'TensorrtExecutionProvider' in dec_sess.get_providers():
            print("🚀 ONNX models successfully loaded and are using TensorrtExecutionProvider.")
        elif 'CUDAExecutionProvider' in enc_sess.get_providers() and 'CUDAExecutionProvider' in dec_sess.get_providers():
            print("✅ ONNX models successfully loaded and are using CUDAExecutionProvider fallback (TensorRT not available or failed).")
        else:
            print("⚠️ ONNX models loaded successfully with CPUExecutionProvider (neither TensorRT nor CUDA available or failed).")
            print(f"Encoder session fallback: see providers... {enc_sess.get_providers()}")
            print(f"Decoder session fallback: see providers... {dec_sess.get_providers()}")

    except Exception as e:
        print(f"❌ GPU session creation failed: {e}")
        print("🔄 Falling back to CPU-only providers...")
        
        try:
            # Fallback to CPU only
            cpu_providers = ['CPUExecutionProvider']
            enc_sess = ort.InferenceSession(encoder_model_path, sess_options=session_options, providers=cpu_providers)
            dec_sess = ort.InferenceSession(decoder_model_path, sess_options=session_options, providers=cpu_providers)
            print("✅ CPU fallback successful")
        except Exception as cpu_e:
            print(f"❌ CPU fallback also failed: {cpu_e}")
            raise RuntimeError("Failed to load ONNX models with any provider")
    return enc_sess, dec_sess

def encode_prompt(enc_sess, image):
    try:
        logger.info("Starting image encoding...")
        
        # Get input details
        enc_input = enc_sess.get_inputs()[0]
        enc_input_name = enc_input.name
        expected_shape = enc_input.shape
        logger.info(f"Expected input shape: {expected_shape}")
        logger.info(f"Input name: {enc_input_name}")
        
        # Preprocess image
        img_arr = np.array(image).astype(np.float32) / 255.0
        input_image = img_arr.transpose(2, 0, 1)[None, :, :, :]
        
        logger.info(f"Actual input shape: {input_image.shape}")
        logger.info(f"Input dtype: {input_image.dtype}")
        logger.info(f"Input memory size: {input_image.nbytes / 1024 / 1024:.1f} MB")
        
        # Memory check
        if input_image.nbytes > 100 * 1024 * 1024:  # 100MB
            logger.warning(f"Large input tensor: {input_image.nbytes / 1024 / 1024:.1f} MB")
        
        # Validate input shape
        if list(input_image.shape) != list(expected_shape) and expected_shape[0] != -1:
            logger.warning(f"Shape mismatch! Expected: {expected_shape}, Got: {input_image.shape}")
        

        logger.info("Running encoder inference...")
        image_embeddings = enc_sess.run(None, {enc_input_name: input_image})[0]
        logger.info("Encoding completed successfully")
        return image_embeddings
        
    except Exception as e:
        logger.error(f"Error during encoding: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# def visualize_results(image: np.ndarray, results: List[Results], 
#                      save_path: Optional[str] = None, show: bool = True):
#     """
#     Visualize inference results.
    
#     Args:
#         image (np.ndarray): Original image in BGR format
#         results (List[Results]): Inference results
#         save_path (Optional[str]): Path to save the visualization
#         show (bool): Whether to display the image
#     """
#     plt.figure(figsize=(12, 8))
    
#     # Convert BGR to RGB for matplotlib
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(rgb_image)
    
#     # Add results visualization here if needed
#     # This is a placeholder - you can extend this based on your needs
    
#     plt.axis('on')
#     plt.title(f"FastSAM Inference Results ({len(results)} detections)")
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=150)
#         logger.info(f"Visualization saved to: {save_path}")
    
#     if show:
#         plt.show()
    
#     plt.close()


def main():
    # """Main execution function."""
    # parser = argparse.ArgumentParser(description="NanoSAM ONNX Inference")
    # parser.add_argument("--model_path", required=True, help="Path to ONNX model file")
    # parser.add_argument("--image_path", required=True, help="Path to input image")
    # parser.add_argument("--output_dir", default="./output", help="Output directory")
    # parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    # parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    # parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    # parser.add_argument("--retina_masks", action="store_true", help="Use retina masks")
    # parser.add_argument("--no_show", action="store_true", help="Don't display results")
    
    # args = parser.parse_args()
    
    # # Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)
    
    enc_sess = None
    dec_sess = None
    
    try:
        logger.info(" 🧹 Mem cleared and gc...")
        torch.cuda.empty_cache()
        gc.collect()

        # Set model paths
        encoder_model_path="/home/copter/onnx_models/nvidia_ai_iot_resnet18_image_encoder.onnx"
        decoder_model_path="/home/copter/onnx_mo dels/nvidia_ai_iot_mobile_sam_mask_decoder.onnx"

        # Check package versions
        check_versions()
        (providers, session_options) = set_onnx_conf()

        # Load ONNX sessions
        logger.info("Initializing NanoSAM ONNX model...")
        (enc_sess, dec_sess) = load_new_onnx_sessions(
            encoder_model_path, 
            decoder_model_path, 
            providers,
            session_options
        )

        img = load_example_image()
        encode_prompt(enc_sess, img)

        logger.info("Processing completed successfully!")
        os._exit(0)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            if enc_sess is not None:
                del enc_sess
            if dec_sess is not None:
                del dec_sess
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass  # Ignore cleanup errors
            
        return 1
    
    finally:
        # Final safety cleanup
        try:
            if 'enc_sess' in locals() and enc_sess is not None:
                del enc_sess
            if 'dec_sess' in locals() and dec_sess is not None:
                del dec_sess
        except:
            pass

if __name__ == "__main__":
    exit(main())