# jetson_benchmark

## SYSTEM SPECS
```
cat /sys/firmware/devicetree/base/model
cat /etc/nv_tegra_release
```

## Model Weights (all benchmarks)

```bash
cd mkdir weights
cd weights
cd mkdir fastsam mobilesam
```


## MobileSAM instructions

```bash 
conda create -n mobilesam_py310 python=3.10 -y
conda activate mobilesam_py310
pip install git+https://github.com/ChaoningZhang/MobileSAM.git  # installl one-line command
```

Install rogue dependencies

``` bash
pip install psutil Pillow numpy==1.26.1 timm
```

Install jetback 6.1 compatible torch and torchvision

``` bash
# First, clean up the CPU version
python -m pip uninstall torch torchvision --user -y

# install jp61 compatible torch with cuda enabled
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# download torch 0.2.0 (compatible with jp61 torch)
cd ~
git clone --branch v0.20.0 https://github.com/pytorch/vision.git
cd ~/vision/
python3 setup.py install
```


run the script
``` bash
  python ./mobilesam_bench.py \
  --model_path "/home/copter/jetson_benchmark/weights/mobilesam/mobile_sam.pt" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --imgsz 256 \
  --iou 0.1,0.3,0.5 \
  --conf 0.2,0.6,0.8 \
  --device cuda \
  --output_csv mobilesam_bench_256.csv
```

observability

```watch nvidia_smi```





## FastSAM instructions

```bash 
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
```


```bash 
conda create -n FastSAM python=3.9 -y # may actually need 3.10 bc of Nvidia & PyTorch wheel

conda activate FastSAM

pip install git+https://github.com/openai/CLIP.git

# Add FastSAM as an importable package (setup,py is in the repo by default), now you can `import fastsam`
cd FastSAM
pip install -e .
```

Note on a jetson nvidia provides wheel files here `https://developer.download.nvidia.com/compute/redist/jp/` for various versions of jetpack

```bash
# On Jetson Orin (Ubuntu 22.04 -- JP6.1 -- ARM64) 
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Downgrade numpy to satisfy nvidia
pip uninstall numpy
pip install numpy==1.26.1

# download torch 0.2.0 (compatible with jp61 torch)
git clone --branch v0.20.0 https://github.com/pytorch/vision.git
cd ~/vision/
python3 setup.py install


# (optional trouble shoot if lib cuSPARSELt in missing
wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

# 5. Install the cusparselt libraries using apt
sudo apt-get -y install libcusparselt0 libcusparselt-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev git build-essential cmake

# verify cuda enabled torch
python3 -c "import torch; print(torch.cuda.is_available())"

```

# NanoSAM (MobileSAM Variant)

```bash
# Create Jetson jp61 conda env
conda create -n NanoSAM python=3.10 -y
conda activate NanoSAM

# PyTorch on Jetson Orin Starter Pack (Ubuntu 22.04 -- JP6.1 -- ARM64) 
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
# Downgrade numpy to satisfy nvidia
pip uninstall numpy -y
pip install numpy==1.26.1
# download torch 0.2.0 (compatible with jp61 torch)
cd ~
git clone --branch v0.20.0 https://github.com/pytorch/vision.git
cd ~/vision/
python3 setup.py install

pip install psutil Pillow timm packaging--user --no-deps # avoids user conflicts

# Install torch2trt (required for NanoSAM) && trtexec not found
export PATH="/usr/src/tensorrt/bin:$PATH"
source ~/.bashrc
trtexec --help
conda activate NanoSAM


# Reestablish trtexec python bindings
sudo apt update
sudo apt install python3-libnvinfer
conda activate NanoSAM
export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

# Install torch2trt (required for NanoSAM) && trtexec not found
cd ~
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
conda activate NanoSAM
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
python setup.py install

# Install NanoSAM repo
git clone https://github.com/NVIDIA-AI-IOT/nanosam
cd nanosam
python3 setup.py develop --user

### libstdc++ OUT OF DATE ERROR
# Prepend system library paths to LD_LIBRARY_PATH (Temporary)
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Also ensure PYTHONPATH is still set for TensorRT Python bindings
export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH

# cd and run the bench
cd ~/jetson_benchmark
python ./nanosam_bench.py \
  --image_encoder_path "/home/copter/onnx_models/mobile_sam_encoder_fp16.engine" \
  --mask_decoder_path "/home/copter/onnx_models/mobile_sam_mask_decoder_fp16.engine" \
  --img_folder "/home/copter/jetson_benchmark/images/*.png" \
  --device cuda \
  --num_runs 50 \
  --output_csv "/home/copter/jetson_benchmark/output/nanosam_bench_fp16_0718T1038.csv"
```

# OPENCV Build Jetson Specific

```
# Remove pip-installed OpenCV
pip3 uninstall opencv-python opencv-contrib-python opencv-python-headless -y

# Remove apt-installed OpenCV packages
sudo apt remove --purge python3-opencv libopencv-dev opencv-data opencv-licenses opencv-samples-data
sudo apt remove --purge libopencv-*
sudo apt autoremove

# -----------------------

cd ~/opencv/build
sudo rm -rf * # Cleans out all build files
cd ~/opencv_contrib
cd ~
sudo rm -rf opencv opencv_contrib opencv.zip opencv_contrib.zip

# ----------------------

wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.8.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.0.zip
unzip opencv.zip
unzip opencv_contrib.zip

mv opencv-4.8.0 opencv
mv opencv_contrib-4.8.0 opencv_contrib

# ----------------------

cd ~/opencv
mkdir build
cd build

# IMPORTANT: Adjust CUDA_ARCH_BIN for your specific Jetson model!
# For Orin series (Orin Nano, Orin NX, AGX Orin) on JetPack 6.x:
CUDA_ARCH_BIN="8.7"

cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
    -D CUDA_ARCH_PTX="" \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_QT=OFF \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_ENABLE_NONFREE=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    ..

# Explanation of key flags:
# -D CMAKE_BUILD_TYPE=RELEASE: Optimized build.
# -D CMAKE_INSTALL_PREFIX=/usr/local: Installs to a standard location.
# -D WITH_CUDA=ON: Enables CUDA support.
# -D WITH_CUDNN=ON: Enables cuDNN support (deep learning primitives).
# -D CUDA_ARCH_BIN="8.7": CRITICAL! Matches your Jetson's compute capability.
# -D CUDA_ARCH_PTX="": Prevents PTX generation (saves space and time).
# -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=ON: Performance optimizations.
# -D WITH_LIBV4L=ON: For V4L2 camera support.
# -D WITH_GSTREAMER=ON: Highly recommended for video processing on Jetson.
# -D WITH_QT=OFF: Disables Qt GUI support. Set to ON if you need it, but it adds complexity and potentially build time.
# -D BUILD_opencv_python3=ON: Builds Python 3 bindings.
# -D OPENCV_ENABLE_NONFREE=OFF: Set to ON if you need non-free algorithms like SIFT/SURF.
# -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules: Includes the contrib modules.

# Start the compilation (this will take a LONG time, 2-4+ hours depending on your Jetson model)
# The -j$(nproc) flag uses all available CPU cores.
make -j$(nproc)

# Install the built libraries
sudo make install
sudo ldconfig
```



## Clean Eviroinments

``` bash
conda deactivate
conda remove -n fastsam_clean --all -yy
conda create -n fastsam_clean python=3.10 -y
conda activate fastsam_clean

# Now install the CUDA PyTorch
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# download torch 0.2.0 (compatible with jp61 torch)
cd ~/vision/
python3 setup.py install

# Reinstall other packages
pip install ultralytics onnx onnx-graphsurgeon Pillow timm  pycudanumpy==1.26.1 
pip3 install --no-deps https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl


# COPY over trt into your chosen conda env
ls /usr/lib/python3.10/dist-packages/tensorrt/tensorrt.so
cp -r /usr/lib/python3.10/dist-packages/tensorrt* $CONDA_PREFIX/lib/python3.10/site-packages/
python -c "import tensorrt; print(tensorrt.__version__)"

# C++ Core Abort Dumps
conda install -c conda-forge libstdcxx-ng=12
# Verify with 
strings /home/copter/miniconda3/envs/tensorrt/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
```



```
# priors:
sudo apt-get update
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

# conda stuff
conda deactivate
conda create -n trt_clean python=3.10 -y
conda activate trt_clean

# Now install the CUDA PyTorch
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# download torch 0.2.0 (compatible with jp61 torch)
cd ~/vision/
python3 setup.py install

# Reinstall other packages
pip install ultralytics  'opencv-python<4.9' onnx onnx-graphsurgeon Pillow timm  pycuda numpy==1.26.1 
pip3 install --no-deps https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl


# COPY over trt into your chosen conda env
ls /usr/lib/python3.10/dist-packages/tensorrt/tensorrt.so
cp -r /usr/lib/python3.10/dist-packages/tensorrt* $CONDA_PREFIX/lib/python3.10/site-packages/
python -c "import tensorrt; print(tensorrt.__version__)"

```


