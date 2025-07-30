#!/bin/bash

echo "=== JETPACK 6.1 SYSTEM AUDIT FOR ONNX RUNTIME BUILD ==="
echo ""

# JetPack Version
echo "📦 JETPACK VERSION:"
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
else
    echo "❌ JetPack version file not found"
fi
echo ""

# CUDA Version and Paths
echo "🔥 CUDA INFORMATION:"
echo "CUDA Version:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "❌ nvcc not found in PATH"
fi

echo "CUDA Installation Paths:"
echo "  CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "  /usr/local/cuda exists: $([ -d /usr/local/cuda ] && echo "✅ Yes" || echo "❌ No")"
echo "  /usr/local/cuda/bin/nvcc exists: $([ -f /usr/local/cuda/bin/nvcc ] && echo "✅ Yes" || echo "❌ No")"
echo "  /usr/local/cuda/lib64 exists: $([ -d /usr/local/cuda/lib64 ] && echo "✅ Yes" || echo "❌ No")"

# CUDA Runtime Version
if [ -f /usr/local/cuda/version.json ]; then
    echo "CUDA Runtime Version:"
    cat /usr/local/cuda/version.json | grep -E '"cuda"|"version"' | head -4
fi
echo ""

# cuDNN Version and Paths
echo "🧠 CUDNN INFORMATION:"
echo "cuDNN Libraries Found:"
find /usr -name "*cudnn*" -type f 2>/dev/null | grep -E "\\.so\\.[0-9]" | head -10

echo ""
echo "cuDNN Version Detection:"
if [ -f /usr/include/aarch64-linux-gnu/cudnn_version_v9.h ]; then
    echo "cuDNN 9 Headers Found:"
    grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/include/aarch64-linux-gnu/cudnn_version_v9.h
elif [ -f /usr/include/cudnn_version.h ]; then
    echo "cuDNN Headers Found:"
    grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/include/cudnn_version.h
else
    echo "❌ cuDNN headers not found"
fi

echo ""
echo "cuDNN Library Versions:"
ls -la /usr/lib/aarch64-linux-gnu/libcudnn.so* 2>/dev/null || echo "❌ cuDNN libraries not found"
echo ""

# TensorRT Version and Paths
echo "⚡ TENSORRT INFORMATION:"
echo "TensorRT Libraries Found:"
find /usr -name "*tensorrt*" -type f 2>/dev/null | head -10

echo ""
echo "TensorRT Version Detection:"
if command -v dpkg &> /dev/null; then
    echo "TensorRT Packages Installed:"
    dpkg -l | grep -i tensorrt | awk '{print $2, $3}'
fi

echo ""
echo "TensorRT Library Versions:"
ls -la /usr/lib/aarch64-linux-gnu/libnvinfer* 2>/dev/null || echo "❌ TensorRT libraries not found"

echo ""
# Check for TensorRT headers
if [ -f /usr/include/aarch64-linux-gnu/NvInfer.h ]; then
    echo "TensorRT Headers Found:"
    grep -E "NV_TENSORRT_MAJOR|NV_TENSORRT_MINOR|NV_TENSORRT_PATCH" /usr/include/aarch64-linux-gnu/NvInfer.h 2>/dev/null | head -3
else
    echo "❌ TensorRT headers not found"
fi
echo ""

# GCC and Build Tools
echo "🔨 BUILD TOOLS:"
echo "GCC Version:"
gcc --version 2>/dev/null | head -1 || echo "❌ GCC not found"

echo "G++ Version:"
g++ --version 2>/dev/null | head -1 || echo "❌ G++ not found"

echo "CMake Version:"
cmake --version 2>/dev/null | head -1 || echo "❌ CMake not found"

echo "Python Version:"
python3 --version 2>/dev/null || echo "❌ Python3 not found"
echo ""

# Library Path Information
echo "📚 LIBRARY PATHS:"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-Not set}"
echo "PKG_CONFIG_PATH: ${PKG_CONFIG_PATH:-Not set}"
echo ""

# System Architecture
echo "🏗️ SYSTEM ARCHITECTURE:"
echo "Architecture: $(uname -m)"
echo "Kernel: $(uname -r)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo ""

# Memory Information
echo "💾 SYSTEM RESOURCES:"
echo "Total Memory: $(free -h | grep 'Mem:' | awk '{print $2}')"
echo "Available Memory: $(free -h | grep 'Mem:' | awk '{print $7}')"
echo "CPU Cores: $(nproc)"
echo ""

echo "=== END SYSTEM AUDIT ==="