# JETSON ORIN TRT SETUP


Setp trt


apt install packages

```bash
sudo apt update
sudo apt install tensorrt tensorrt-libs libnvinfer-dev python3-libnvinfer
```

Verify install

```bash
# Verify install
ls /usr/src/tensorrt/bin

# trtexec not found
export PATH="/usr/src/tensorrt/bin:$PATH"
source ~/.bashrc
trtexec --help

# Verify python bundings
sudo find / -name "_tensorrt.so" 2>/dev/null
sudo find / -name "tensorrt.so" 2>/dev/null
/usr/lib/python3.10/dist-packages/tensorrt/tensorrt.so
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.10/dist-packages/
python3 -c "import tensorrt; print(tensorrt.__version__)"
# >> version  10.3.0

# need to downgrade np
pip uninstall numpy -y
pip install numpy==1.26.1

```


```bash
conda create -n tensorrt python=3.10 -y
conda activate tensorrt 
pip install ultralytics timm
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
python3 tensor_rt/fastsam_trt_convert.py
```




### NON VENV RAW SYSTEM BUILD

```bash
# Use system pip with Python 3.10 instead of conda's pip
/usr/bin/python3.10 -m pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl --user

# Or force install with current pip3 (which should work):
pip3 install --no-cache --force-reinstall --no-deps https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl --user

# Ultralytics to avoid dependency conflicts:
python3.10 -m pip install ultralytics --user --no-deps
python3.10 -m pip install pyyaml tqdm psutil pillow --user

# Next, clean up any existing torchvision packages
python3.10 -m pip uninstall torchvision -y

# If the vision directory already exists, remove it
rm -rf ~/vision

# Clone the compatible version
cd ~
git clone --branch v0.20.0 https://github.com/pytorch/vision.git
cd ~/vision/

# Build and install specifically for Python 3.10
python3.10 setup.py build
python3.10 setup.py install --user

# For an arm64 enabled onnxruntime go to this site
https://elinux.org/Jetson_Zoo#ONNX_Runtime

# trtexec not found
export PATH="/usr/src/tensorrt/bin:$PATH"
source ~/.bashrc
trtexec --help


# Verify install
python3.10 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
else:
    print('CUDA not available - this is the problem!')
"

python3.10 -c "
import torch
import torchvision
import numpy as np
import onnx
import onnxruntime
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}') 
print(f'NumPy: {np.__version__}')
print(f'ONNX: {onnx.__version__}')
print(f'ONNX Runtime: {onnxruntime.__version__}')
print(f'CUDA still available: {torch.cuda.is_available()}')
"

python3.10 jetson_benchmark/tensor_rt/fastsam_trt_convert.py
```


## NanoSAM (Nvidian Repo)


```bash
cd ~
mkdir onnx_models

# ...download models there

# trtexec not found
export PATH="/usr/src/tensorrt/bin:$PATH"
source ~/.bashrc
trtexec --help

# FP16
trtexec \
    --onnx=onnx_models/mobile_sam_mask_decoder.onnx \
    --saveEngine=onnx_models/mobile_sam_mask_decoder_fp16.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10 \
    --fp16

# FP32
trtexec \
    --onnx=onnx_models/mobile_sam_mask_decoder.onnx \
    --saveEngine=onnx_models/mobile_sam_mask_decoder_fp32.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10

trtexec \
    --onnx=onnx_models/mobile_sam_encoder.onnx \
    --saveEngine=onnx_models/mobile_sam_encoder_fp16.engine \
    --fp16

# Monitor progress via...
tegrastats
```