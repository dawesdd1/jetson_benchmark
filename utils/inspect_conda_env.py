"""
Requiremnts:
TensorRT (10.3.0) engine
pip install pycuda

python /home/copter/jetson_benchmark/utils/inspect_conda_env.py

Comprehensive Conda Environment Inspector for Jetson/TensorRT Development
Inspects system info, JetPack version, C++ libraries, Python environment, and key packages
"""

import os
import sys
import subprocess
import platform
import importlib
import glob
from pathlib import Path

def run_command(cmd, capture_output=True, text=True, shell=True):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=text, shell=shell)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def get_ubuntu_version():
    """Get Ubuntu version information"""
    print("🐧 Ubuntu System Information")
    print("-" * 40)
    
    # Method 1: /etc/os-release
    os_release = run_command("cat /etc/os-release")
    if os_release:
        for line in os_release.split('\n'):
            if line.startswith('PRETTY_NAME='):
                ubuntu_name = line.split('=')[1].strip('"')
                print(f"   OS: {ubuntu_name}")
            elif line.startswith('VERSION_ID='):
                ubuntu_version = line.split('=')[1].strip('"')
                print(f"   Version ID: {ubuntu_version}")
    
    # Method 2: lsb_release
    lsb_release = run_command("lsb_release -d")
    if lsb_release:
        print(f"   LSB: {lsb_release.split(':', 1)[1].strip()}")
    
    # Kernel version
    kernel = run_command("uname -r")
    if kernel:
        print(f"   Kernel: {kernel}")
    
    # Architecture
    arch = run_command("uname -m")
    if arch:
        print(f"   Architecture: {arch}")

def get_jetpack_version():
    """Get JetPack version information"""
    print("\n🚀 JetPack Information")
    print("-" * 40)
    
    # Method 1: Check /etc/nv_tegra_release
    tegra_release = run_command("cat /etc/nv_tegra_release 2>/dev/null")
    if tegra_release:
        print(f"   Tegra Release: {tegra_release}")
    
    # Method 2: Check jetson_release if available
    jetson_release = run_command("jetson_release -v 2>/dev/null")
    if jetson_release:
        print(f"   Jetson Release: {jetson_release}")
    
    # Method 3: Check dpkg for nvidia packages
    nvidia_l4t = run_command("dpkg -l | grep nvidia-l4t-core | head -1")
    if nvidia_l4t:
        parts = nvidia_l4t.split()
        if len(parts) >= 3:
            print(f"   L4T Core Version: {parts[2]}")
    
    # Method 4: Check CUDA version
    cuda_version = run_command("nvcc --version 2>/dev/null | grep 'release' | awk '{print $6}' | cut -c2-")
    if cuda_version:
        print(f"   CUDA Version: {cuda_version}")
    else:
        # Alternative CUDA check
        cuda_alt = run_command("cat /usr/local/cuda/version.txt 2>/dev/null")
        if cuda_alt:
            print(f"   CUDA Version: {cuda_alt}")
    
    # Method 5: Check nvidia-smi if available
    nvidia_smi = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null")
    if nvidia_smi:
        print(f"   NVIDIA Driver: {nvidia_smi}")
    
    # Method 6: Check for Jetson model
    jetson_model = run_command("cat /proc/device-tree/model 2>/dev/null | tr -d '\\0'")
    if jetson_model:
        print(f"   Device Model: {jetson_model}")

def get_cpp_libraries():
    """Get critical C++ library versions"""
    print("\n🔧 Critical C++ Libraries")
    print("-" * 40)
    
    # GLIBC version (system-wide)
    glibc_version = run_command("ldd --version | head -1")
    if glibc_version:
        print(f"   System GLIBC: {glibc_version}")
    
    # Get conda environment path
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    
    # Check conda environment GLIBCXX first
    if conda_prefix:
        print(f"\n   🐍 Conda Environment C++ Libraries ({os.path.basename(conda_prefix)}):")
        conda_libstdcpp_paths = [
            f"{conda_prefix}/lib/libstdc++.so.6",
            f"{conda_prefix}/lib/x86_64-linux-gnu/libstdc++.so.6",
            f"{conda_prefix}/lib/aarch64-linux-gnu/libstdc++.so.6"
        ]
        
        conda_lib_found = False
        for path in conda_libstdcpp_paths:
            if os.path.exists(path):
                conda_lib_found = True
                # Get symbolic link target
                real_path = run_command(f"readlink -f {path}")
                if real_path:
                    print(f"     libstdc++.so.6 -> {real_path}")
                
                # Check GLIBCXX versions in conda environment
                glibcxx_conda = run_command(f"strings {path} | grep GLIBCXX | tail -10 2>/dev/null")
                if glibcxx_conda:
                    print(f"     GLIBCXX versions available:")
                    for line in glibcxx_conda.split('\n'):
                        if line.strip() and 'GLIBCXX_3.4' in line:
                            print(f"       {line.strip()}")
                break
        
        if not conda_lib_found:
            print(f"     ❌ No libstdc++.so.6 found in conda environment")
            print(f"     📁 Checked paths:")
            for path in conda_libstdcpp_paths:
                print(f"       - {path}")
    
    # System GLIBCXX version
    print(f"\n   🖥️  System C++ Libraries:")
    print("     GLIBCXX versions available:")
    glibcxx_check = run_command("strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -10 2>/dev/null")
    if not glibcxx_check:  # Try alternative paths for ARM64
        glibcxx_check = run_command("strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -10 2>/dev/null")
    if glibcxx_check:
        for line in glibcxx_check.split('\n'):
            if line.strip() and 'GLIBCXX_3.4' in line:
                print(f"       {line.strip()}")
    
    # GCC version (system)
    gcc_version = run_command("gcc --version | head -1")
    if gcc_version:
        print(f"   System GCC: {gcc_version}")
    
    # G++ version (system)
    gpp_version = run_command("g++ --version | head -1")
    if gpp_version:
        print(f"   System G++: {gpp_version}")
    
    # Check for system libstdc++ location and version
    print(f"\n   🖥️  System libstdc++ location:")
    libstdcpp_paths = [
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib/aarch64-linux-gnu/libstdc++.so.6",
        "/usr/lib64/libstdc++.so.6"
    ]
    
    for path in libstdcpp_paths:
        if os.path.exists(path):
            # Get symbolic link target
            real_path = run_command(f"readlink -f {path}")
            if real_path:
                print(f"     libstdc++.so.6 -> {real_path}")
            break
    
    # Check which libstdc++ Python will actually use
    print(f"\n   🐍 Python's libstdc++ (what packages will use):")
    python_libstdcpp = run_command(f'python -c "import ctypes.util; lib = ctypes.util.find_library(\'stdc++\'); print(f\'Python finds libstdc++ at: {{lib}}\')"')
    if python_libstdcpp:
        print(f"     {python_libstdcpp}")
    
    # Check LD_LIBRARY_PATH
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    print(f"   📚 LD_LIBRARY_PATH: {ld_library_path}")
    
    # Summary
    print(f"\n   ⚠️  Important: Conda packages will typically use:")
    if conda_prefix and conda_lib_found:
        print(f"     - Conda environment libraries from {conda_prefix}/lib/")
    else:
        print(f"     - System libraries (conda env has no C++ libs)")
    print(f"   💡 C++ ABI mismatches between these can cause crashes!")

def get_python_info():
    """Get Python environment information"""
    print("\n🐍 Python Environment")
    print("-" * 40)
    
    print(f"   Python Version: {sys.version}")
    print(f"   Python Executable: {sys.executable}")
    print(f"   Python Path: {sys.path[0]}")
    
    # Conda environment info
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
    print(f"   Conda Environment: {conda_env}")
    
    conda_prefix = os.environ.get('CONDA_PREFIX', 'N/A')
    print(f"   Conda Prefix: {conda_prefix}")
    
    # Virtual environment info
    virtual_env = os.environ.get('VIRTUAL_ENV', 'Not in virtual environment')
    print(f"   Virtual Environment: {virtual_env}")

def check_package_version(package_name):
    """Check if a package is installed and return its version"""
    try:
        # Use subprocess to isolate potentially problematic imports
        cmd = f'python -c "import {package_name}; print(getattr({package_name}, \'__version__\', getattr({package_name}, \'version\', getattr({package_name}, \'VERSION\', \'Version not found\'))))"'
        result = run_command(cmd)
        return result if result else "Import successful, version not found"
    except Exception as e:
        # Fallback to direct import for simple cases
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                return module.__version__
            elif hasattr(module, 'version'):
                return module.version
            elif hasattr(module, 'VERSION'):
                return module.VERSION
            else:
                return "Version not found"
        except ImportError:
            return "Not installed"
        except Exception as e2:
            return f"Error: {str(e2)}"

def get_key_packages():
    """Check versions of key packages for ML/AI development"""
    print("\n📦 Key Package Versions")
    print("-" * 40)
    
    # Define key packages to check
    key_packages = {
        # Core ML/AI packages
        'numpy': 'numpy',
        'pandas': 'pandas',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'matplotlib': 'matplotlib',
        
        # Deep Learning frameworks
        'torch': 'torch',
        'torchvision': 'torchvision',
        'tensorflow': 'tensorflow',
        'keras': 'keras',
        
        # TensorRT and ONNX
        'tensorrt': 'tensorrt',
        'onnx': 'onnx',
        'onnxruntime': 'onnxruntime',
        'onnxruntime-gpu': 'onnxruntime',  # Same module name
        
        # CUDA packages
        'pycuda': 'pycuda',
        'cupy': 'cupy',
        'numba': 'numba',
        
        # Other common packages
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',
        'tqdm': 'tqdm',
        'requests': 'requests',
        'jupyter': 'jupyter',
        'ipython': 'IPython',
        
        # Specialized packages
        'transformers': 'transformers',
        'datasets': 'datasets',
        'accelerate': 'accelerate',
        'diffusers': 'diffusers',
    }
    
    # Check each package
    max_name_length = max(len(name) for name in key_packages.keys())
    
    for package_name, import_name in key_packages.items():
        version = check_package_version(import_name)
        print(f"   {package_name:<{max_name_length}} : {version}")
    
    # Special checks for packages with unique version methods
    print(f"\n📦 Special Package Checks")
    print("-" * 40)
    
    # TensorRT special check (safer)
    print("   Checking TensorRT...")
    try:
        trt_version = run_command('python -c "import tensorrt as trt; print(f\'TensorRT Version: {trt.__version__}\')"')
        if trt_version:
            print(f"   {trt_version}")
        else:
            print("   TensorRT: Import failed")
        
        trt_build = run_command('python -c "import tensorrt as trt; print(f\'TensorRT Build: {trt.Builder.get_version_string()}\')"')
        if trt_build:
            print(f"   {trt_build}")
    except Exception as e:
        print(f"   TensorRT: Error checking - {str(e)}")
    
    # PyTorch CUDA availability (safer - this is where it crashed)
    print("   Checking PyTorch CUDA...")
    try:
        # Use subprocess to avoid crashes in main process
        torch_cuda = run_command('python -c "import torch; print(f\'PyTorch CUDA Available: {torch.cuda.is_available()}\')"')
        if torch_cuda:
            print(f"   {torch_cuda}")
        else:
            print("   PyTorch CUDA: Check failed")
        
        torch_cuda_version = run_command('python -c "import torch; print(f\'PyTorch CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \'N/A\'}\')"')
        if torch_cuda_version:
            print(f"   {torch_cuda_version}")
        
        torch_cudnn = run_command('python -c "import torch; print(f\'PyTorch cuDNN Version: {torch.backends.cudnn.version() if torch.cuda.is_available() else \'N/A\'}\')"')
        if torch_cudnn:
            print(f"   {torch_cudnn}")
        
        gpu_count = run_command('python -c "import torch; print(f\'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\')"')
        if gpu_count:
            print(f"   {gpu_count}")
        
        gpu_name = run_command('python -c "import torch; print(f\'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else \'N/A\'}\')"')
        if gpu_name:
            print(f"   {gpu_name}")
            
    except Exception as e:
        print(f"   PyTorch CUDA: Error checking - {str(e)}")
    
    # ONNX Runtime providers (safer)
    print("   Checking ONNX Runtime...")
    try:
        ort_providers = run_command('python -c "import onnxruntime as ort; print(f\'ONNX Runtime Providers: {\\\', \\\'.join(ort.get_available_providers())}\')"')
        if ort_providers:
            print(f"   {ort_providers}")
        else:
            print("   ONNX Runtime: Not available")
    except Exception as e:
        print(f"   ONNX Runtime: Error checking - {str(e)}")

def get_conda_list():
    """Get conda list output for installed packages"""
    print("\n📋 Conda Package List (Key Packages Only)")
    print("-" * 40)
    
    # Get conda list
    conda_list = run_command("conda list")
    if conda_list:
        # Filter for key packages
        key_terms = [
            'tensor', 'torch', 'cuda', 'onnx', 'opencv', 'numpy', 
            'scipy', 'pandas', 'matplotlib', 'pillow', 'sklearn',
            'jupyter', 'notebook', 'transformers', 'diffusers'
        ]
        
        lines = conda_list.split('\n')
        header_printed = False
        
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            # Print header once
            if not header_printed and not line.startswith('#'):
                print("   Package Name                Version       Build Channel")
                print("   " + "-" * 60)
                header_printed = True
            
            # Check if line contains any key terms
            line_lower = line.lower()
            if any(term in line_lower for term in key_terms):
                # Format the line nicely
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0][:24]
                    version = parts[1][:12]
                    build = parts[2][:12] if len(parts) > 2 else ''
                    channel = parts[3] if len(parts) > 3 else ''
                    print(f"   {name:<24} {version:<12} {build:<12} {channel}")

def main():
    """Main inspection function"""
    # Get environment name
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
    virtual_env = os.environ.get('VIRTUAL_ENV', '')
    
    if conda_env != 'Unknown' and conda_env != 'base':
        env_display = f"Conda Environment: {conda_env}"
    elif virtual_env:
        env_name = os.path.basename(virtual_env)
        env_display = f"Virtual Environment: {env_name}"
    else:
        env_display = "System Python Environment"
    
    print("🔍 Comprehensive Conda Environment Inspector")
    print("=" * 60)
    print(f"Environment: {env_display}")
    print(f"Inspection Date: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")
    print("=" * 60)
    
    # System information
    get_ubuntu_version()
    
    # JetPack information
    get_jetpack_version()
    
    # C++ libraries
    get_cpp_libraries()
    
    # Python environment
    get_python_info()
    
    # Key packages
    get_key_packages()
    
    # Conda package list
    get_conda_list()
    
    print("\n" + "=" * 60)
    print("🎉 Environment inspection completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()