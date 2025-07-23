#!/usr/bin/env python3
"""
Script to inspect TensorRT (10.3.0) engine files and show their tensor names and shapes.
This helps debug tensor name mismatches in NanoSAM engines.

python /home/copter/jetson_benchmark/utils/inspect_nanosam_engine.py \
--image_encoder "/home/copter/engine_models/nvidia_nanosam_resnet18_image_encoder_fp16.engine" \
--mask_decoder "/home/copter/engine_models/nvidia_nanosam_mask_decoder_fp16.engine"
"""

import tensorrt as trt
import argparse
import sys

def inspect_engine(engine_path):
    """Inspect a TensorRT engine file and print its I/O tensor information."""
    print(f"\n🔍 Inspecting TensorRT engine: {engine_path}")
    print("=" * 60)
    
    try:
        # Create TensorRT logger
        logger = trt.Logger(trt.Logger.WARNING)
        
        # Load the engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            print(f"❌ Failed to load engine from {engine_path}")
            return False
        
        print(f"✅ Engine loaded successfully")
        print(f"📊 Engine info:")
        
        # Handle different TensorRT API versions
        try:
            if hasattr(engine, 'max_batch_size'):
                print(f"   - Max batch size: {engine.max_batch_size}")
            else:
                print(f"   - Max batch size: Not available (dynamic batching)")
        except:
            print(f"   - Max batch size: Not available")
            
        try:
            if hasattr(engine, 'num_layers'):
                print(f"   - Number of layers: {engine.num_layers}")
        except:
            print(f"   - Number of layers: Not available")
        
        # Try modern API first (TensorRT 8.5+)
        try:
            if hasattr(engine, 'num_io_tensors'):
                print(f"   - Number of I/O tensors: {engine.num_io_tensors}")
                
                print(f"\n📋 I/O Tensor Details (Modern API):")
                print("-" * 40)
                
                for i in range(engine.num_io_tensors):
                    tensor_name = engine.get_tensor_name(i)
                    tensor_shape = engine.get_tensor_shape(tensor_name)
                    tensor_dtype = engine.get_tensor_dtype(tensor_name)
                    tensor_mode = engine.get_tensor_mode(tensor_name)
                    
                    mode_str = "INPUT" if tensor_mode == trt.TensorIOMode.INPUT else "OUTPUT"
                    
                    print(f"   [{i}] {tensor_name}")
                    print(f"       - Mode: {mode_str}")
                    print(f"       - Shape: {tensor_shape}")
                    print(f"       - Data type: {tensor_dtype}")
                    print()
            else:
                raise AttributeError("Modern API not available")
                
        except Exception as modern_e:
            print(f"   Modern API failed: {modern_e}")
            
            # Fall back to legacy binding API
            print(f"\n📋 Binding Information (Legacy API):")
            print("-" * 40)
            try:
                if hasattr(engine, 'num_bindings'):
                    print(f"   - Number of bindings: {engine.num_bindings}")
                    
                    for i in range(engine.num_bindings):
                        binding_name = engine.get_binding_name(i)
                        binding_shape = engine.get_binding_shape(i)
                        binding_dtype = engine.get_binding_dtype(i)
                        is_input = engine.binding_is_input(i)
                        
                        mode_str = "INPUT" if is_input else "OUTPUT"
                        print(f"   [{i}] {binding_name}")
                        print(f"       - Mode: {mode_str}")
                        print(f"       - Shape: {binding_shape}")
                        print(f"       - Data type: {binding_dtype}")
                        print()
                else:
                    print(f"   ❌ No binding information available")
            except Exception as legacy_e:
                print(f"   Legacy binding API also failed: {legacy_e}")
                
                # Try even older API methods
                print(f"\n📋 Alternative Inspection Methods:")
                print("-" * 40)
                try:
                    # Try to get any available engine attributes
                    attrs = [attr for attr in dir(engine) if not attr.startswith('_')]
                    print(f"   Available engine methods/attributes:")
                    for attr in attrs[:20]:  # Show first 20 to avoid spam
                        print(f"     - {attr}")
                    if len(attrs) > 20:
                        print(f"     ... and {len(attrs) - 20} more")
                except Exception as attr_e:
                    print(f"   Could not inspect engine attributes: {attr_e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting engine: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Inspect TensorRT engine files")
    parser.add_argument("--image_encoder", type=str, 
                       help="Path to image encoder engine file")
    parser.add_argument("--mask_decoder", type=str,
                       help="Path to mask decoder engine file")
    parser.add_argument("--engine", type=str,
                       help="Path to a single engine file to inspect")
    
    args = parser.parse_args()
    
    if not any([args.image_encoder, args.mask_decoder, args.engine]):
        print("❌ Please provide at least one engine file to inspect")
        parser.print_help()
        return
    
    print(f"🔧 TensorRT Engine Inspector")
    print(f"📍 TensorRT version: {trt.__version__}")
    
    success = True
    
    if args.engine:
        success &= inspect_engine(args.engine)
    
    if args.image_encoder:
        success &= inspect_engine(args.image_encoder)
    
    if args.mask_decoder:
        success &= inspect_engine(args.mask_decoder)
    
    if success:
        print("\n✅ All engines inspected successfully")
        print("\n💡 Common tensor names should be:")
        print("   - Image Encoder Input: 'input', 'x', or 'images'")
        print("   - Image Encoder Output: 'output', 'features', or 'embeddings'")
        print("   - Mask Decoder Inputs: 'image_embeddings', 'point_coords', 'point_labels'")
        print("   - Mask Decoder Outputs: 'masks', 'iou_predictions', 'low_res_masks'")
    else:
        print("\n❌ Some engines failed inspection")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
