


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

### 3 Clone the repo
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