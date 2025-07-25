"""
📊  TensorRT Engine Metadata Inspection

python /home/copter/jetson_benchmark/utils/tensorrt_engine_metadata.py \
  --engine_path "/home/copter/engine_models/FastSAM-s_fp16.engine"


python /home/copter/jetson_benchmark/utils/tensorrt_engine_metadata.py \
  --engine_path "/home/copter/engine_models/zhudongwork_mobile_sam_encoder_fp16.engine"


python /home/copter/jetson_benchmark/utils/tensorrt_engine_metadata.py \
  --engine_path "/home/copter/engine_models/zhudongwork_mobile_sam_encoder_fp16_trt1030.engine"


python /home/copter/jetson_benchmark/utils/tensorrt_engine_metadata.py \
  --engine_path "/home/copter/engine_models/zhudongwork_mobile_sam_decoder_fp16_trt1030.engine"
"""

import argparse
import os
import sys
import tensorrt as trt
import struct
import datetime

def inspect_engine_metadata(engine_path):
    """Extract metadata from a TensorRT engine file"""
    print(f"🔍 Inspecting TensorRT Engine: {os.path.basename(engine_path)}")
    print("=" * 60)
    
    # Basic file info
    if os.path.exists(engine_path):
        file_size = os.path.getsize(engine_path)
        file_mtime = os.path.getmtime(engine_path)
        print(f"📁 File size: {file_size / (1024*1024):.1f} MB")
        print(f"📅 Last modified: {datetime.datetime.fromtimestamp(file_mtime)}")
    else:
        print(f"❌ Engine file not found: {engine_path}")
        return False
    
    try:
        # Create TensorRT logger and runtime
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        print(f"🔧 Current TensorRT version: {trt.__version__}")
        
        # Try to deserialize the engine
        try:
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                print(f"❌ Failed to deserialize engine - likely version mismatch")
                return analyze_engine_binary(engine_data)
            
            print(f"✅ Engine deserialized successfully with current TensorRT")
            
            # Try to get the actual build version from engine metadata
            print(f"\n🔧 Engine Build Information:")
            print("-" * 40)
            
            # Method 1: Try to get version from engine serialization info
            try:
                # Some engines store build info in serialization metadata
                serialized_size = len(engine_data)
                print(f"   Serialized engine size: {serialized_size / (1024*1024):.1f} MB")
                
                # Try to extract version info from the beginning of the binary
                header_data = engine_data[:1024].decode('latin-1', errors='ignore')
                
                import re
                # Look for embedded version strings
                version_matches = re.findall(r'(\d+\.\d+\.\d+(?:\.\d+)?)', header_data)
                if version_matches:
                    print(f"   Possible build version(s) found in header:")
                    for version in set(version_matches):
                        # Filter out obviously wrong versions (like dates)
                        parts = version.split('.')
                        if len(parts) >= 3 and int(parts[0]) <= 15 and int(parts[1]) <= 20:
                            print(f"     - {version}")
                
                # Look for TensorRT specific markers
                if 'TensorRT' in header_data or 'tensorrt' in header_data:
                    trt_context = re.search(r'(?i)tensorrt[^\d]*(\d+\.\d+\.\d+)', header_data)
                    if trt_context:
                        print(f"   TensorRT version in header: {trt_context.group(1)}")
                
            except Exception as e:
                print(f"   Could not extract build version from header: {e}")
            
            # Method 2: Check if we can get build info from engine properties
            try:
                # Try to access any available version information
                if hasattr(engine, 'get_build_config'):
                    print(f"   Engine has build config information")
                
                # Some engines might have embedded build timestamps
                engine_str = str(engine)
                if 'version' in engine_str.lower():
                    print(f"   Engine string contains version info")
                    
            except:
                pass
            
            # Method 3: Based on successful deserialization, infer version range
            current_version = trt.__version__
            major, minor = current_version.split('.')[:2]
            print(f"   Compatible version range: {major}.{minor}.x")
            print(f"   Since deserialization succeeded, engine was likely built with TensorRT {major}.{minor}.x")
            
            # Extract engine information
            print(f"\n🔧 Engine Information:")
            print("-" * 40)
            
            # Get number of layers
            try:
                if hasattr(engine, 'num_layers'):
                    print(f"   Number of layers: {engine.num_layers}")
            except:
                pass
            
            # Get device memory size  
            try:
                device_mem_size = engine.device_memory_size
                print(f"   Device memory required: {device_mem_size / (1024*1024):.1f} MB")
            except:
                pass
            
            # Get workspace size
            try:
                workspace_size = engine.device_memory_size
                print(f"   Workspace size: {workspace_size / (1024*1024):.1f} MB")
            except:
                pass
            
            # Get I/O tensor information
            try:
                if hasattr(engine, 'num_io_tensors'):
                    print(f"   Number of I/O tensors: {engine.num_io_tensors}")
                    
                    print(f"\n📋 Tensor Information:")
                    print("-" * 40)
                    for i in range(engine.num_io_tensors):
                        name = engine.get_tensor_name(i)
                        shape = engine.get_tensor_shape(name)
                        dtype = engine.get_tensor_dtype(name)
                        mode = engine.get_tensor_mode(name)
                        
                        mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
                        print(f"   [{i}] {name}: {shape} ({dtype}) - {mode_str}")
                        
                elif hasattr(engine, 'num_bindings'):
                    print(f"   Number of bindings: {engine.num_bindings}")
                    
                    print(f"\n📋 Binding Information:")
                    print("-" * 40)
                    for i in range(engine.num_bindings):
                        name = engine.get_binding_name(i)
                        shape = engine.get_binding_shape(i)
                        dtype = engine.get_binding_dtype(i)
                        is_input = engine.binding_is_input(i)
                        
                        mode_str = "INPUT" if is_input else "OUTPUT"
                        print(f"   [{i}] {name}: {shape} ({dtype}) - {mode_str}")
            except Exception as e:
                print(f"   Could not get tensor info: {e}")
            
            return True
            
        except Exception as deserialize_error:
            print(f"❌ Could not deserialize with current TensorRT {trt.__version__}")
            print(f"   Error: {deserialize_error}")
            print(f"   This suggests the engine was built with a different TensorRT version")
            
            # Try to extract info from binary
            return analyze_engine_binary(engine_data)
            
    except Exception as e:
        print(f"❌ Error inspecting engine: {e}")
        return False

def analyze_engine_binary(engine_data):
    """Try to extract metadata from engine binary data"""
    print(f"\n🔍 Binary Analysis (Engine built with different TensorRT version):")
    print("-" * 60)
    
    try:
        # Look for version strings in binary data
        data_str = engine_data.decode('latin-1', errors='ignore')
        
        # Search for TensorRT version patterns
        import re
        
        # Pattern for TensorRT version numbers
        version_patterns = [
            r'TensorRT[^\d]*(\d+\.\d+\.\d+[\.\d]*)',
            r'tensorrt[^\d]*(\d+\.\d+\.\d+[\.\d]*)',
            r'(\d+\.\d+\.\d+[\.\d]*)[^\d]*TensorRT',
            r'version[^\d]*(\d+\.\d+\.\d+[\.\d]*)',
        ]
        
        found_versions = set()
        for pattern in version_patterns:
            matches = re.findall(pattern, data_str, re.IGNORECASE)
            found_versions.update(matches)
        
        if found_versions:
            print(f"   🔢 Possible TensorRT versions found:")
            for version in sorted(found_versions):
                if len(version.split('.')) >= 3:  # Valid version format
                    print(f"     - {version}")
        
        # Look for CUDA compute capability
        cuda_patterns = [
            r'sm_(\d+)',
            r'compute_(\d+)',
            r'CUDA[^\d]*(\d+\.\d+)',
        ]
        
        found_cuda = set()
        for pattern in cuda_patterns:
            matches = re.findall(pattern, data_str, re.IGNORECASE)
            found_cuda.update(matches)
        
        if found_cuda:
            print(f"   🎮 Possible CUDA compute capabilities:")
            for cuda in sorted(found_cuda):
                print(f"     - {cuda}")
        
        # Look for GPU architecture hints
        arch_patterns = [
            r'(ampere|hopper|ada|turing|volta|pascal)',
            r'(rtx\d+|gtx\d+|tesla|quadro)',
            r'(a100|h100|v100|p100|jetson)',
        ]
        
        found_archs = set()
        for pattern in arch_patterns:
            matches = re.findall(pattern, data_str, re.IGNORECASE)
            found_archs.update([m.lower() for m in matches])
        
        if found_archs:
            print(f"   🏗️  Possible GPU architectures/models:")
            for arch in sorted(found_archs):
                print(f"     - {arch}")
        
        # Check file magic number / header
        if len(engine_data) >= 16:
            header_bytes = engine_data[:16]
            print(f"   📄 File header (hex): {header_bytes.hex()}")
            
            # Try to identify TensorRT version from header patterns
            if b'TRT' in header_bytes or b'tensorrt' in header_bytes:
                print(f"   ✅ Confirmed TensorRT engine file")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Binary analysis failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect TensorRT engine metadata")
    parser.add_argument("--engine_path", help="Path to TensorRT engine file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.engine_path):
        print(f"❌ Engine file not found: {args.engine_path}")
        sys.exit(1)
    
    success = inspect_engine_metadata(args.engine_path)
    
    if success:
        print(f"\n✅ Engine inspection completed")
        print(f"\n💡 Tips:")
        print(f"   - If deserialization failed, the engine was built with a different TensorRT version")
        print(f"   - Engines are only compatible within the same major.minor TensorRT version")
        print(f"   - For best performance, rebuild engines with your current TensorRT version")
    else:
        print(f"\n❌ Engine inspection failed")
        sys.exit(1)

if __name__ == "__main__":
    main()