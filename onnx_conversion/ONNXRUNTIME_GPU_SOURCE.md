# ONNX RUNTIME GPU

https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/onnxruntime-gpu/overview/1.23.0.dev20250725001

### (Shortcut) If Built Already

```
cd ~/onnxruntime
pip uninstall onnxruntime -y
pip install build/Linux/Release/dist/onnxruntime_gpu-*.whl --force-reinstall
pip install sympy==1.13.1 

```


### 1 Sudo apt packages

```
# Update package lists and upgrade existing packages
sudo apt update
sudo apt upgrade -y

# Install essential build tools and dependencies
sudo apt install -y build-essential git cmake curl \
libcurl4-openssl-dev libopenblas-dev software-properties-common \
python3-setuptools python3-wheel

```


### 2 (optionsl)

```
# Create an 8GB swap file (adjust size if needed)
sudo fallocate -l 8G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile

# Make the swap file permanent by adding it to /etc/fstab
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# Verify that the swap is active
free -h

```

### 3 Clone the repo (it pulls the latest version 1.23.0)
```
# Clone ONNX Runtime
cd ~
git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime

```


### 4 Build the repo
```
# from claude
./build.sh --config Release --update --build --parallel --build_wheel \
    --use_cuda --use_tensorrt \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/aarch64-linux-gnu \
    --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --parallel 4  # Limit parallel jobs for Jetson memory

# from gemini
./build.sh --config Release \
           --update \
           --build \
           --parallel \
           --build_wheel \
           --use_tensorrt \
           --tensorrt_home /usr/src/tensorrt/ \
           --use_cuda \
           --cuda_home /usr/local/cuda/ \
           --cudnn_home /usr/lib/aarch64-linux-gnu/ \
           --enable_pybind \
           --skip_tests


# Install the built wheel
pip install build/Linux/Release/dist/onnxruntime_gpu-*.whl
```



### CLEAN RETRY
```
# 1. Clean conda environment libraries that might conflict
conda activate FastSAM_onnx
conda remove --force libstdcxx-ng libgcc-ng -y
conda install -c conda-forge libstdcxx-ng=12 libgcc-ng=12 -y

# 2. Set clean library path (system libraries only)
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"

# 3. Clean rebuild
cd ~/onnxruntime
rm -rf build/
./build.sh --config Release --update --build --parallel --build_wheel \
    --use_cuda --use_tensorrt \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/aarch64-linux-gnu \
    --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --parallel 4

# 4. Install and test
pip install build/Linux/Release/dist/onnxruntime_gpu-*.whl --force-reinstall

# 5. Test in clean environment
cd ~  # Important: not in source directory
python -c "
import onnxruntime as ort
print('✅ Version:', ort.__version__)
print('✅ Providers:', ort.get_available_providers())
print('✅ No crashes!')
"
```
