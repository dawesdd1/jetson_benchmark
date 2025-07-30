# Startup

## Nvidia containter toolkit 
https://github.com/NVIDIA/nvidia-container-toolkit

&&

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


# Nvidia torch JP (all versions)
https://developer.download.nvidia.com/compute/redist/jp/

nvidia-apikey: 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

## Official NVIDIA Docker Images
get your free api key

```
sudo docker login nvcr.io
# Username: $oauthtoken
# Password: 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
```

## Build things

``` bash
# Build the Docker image
# sudo docker build -f Dockerfile.jetson_bench -t tensorrt-8.6.1-benchmark .
sudo docker build -f Dockerfile.dusty_jetson_bench -t tensorrt-8.6.1-benchmark .


#            
#                                               ☝️ This names your image "tensorrt-8.6.1-benchmark"

# Run the container with GPU access and host networking
sudo docker run -it --rm \
    --runtime nvidia \
    --gpus all \
    --network host \
    --privileged \
    -v /home/copter/engine_models:/workspace/engines \
    -v /home/copter/jetson_benchmark/benchmarking:/workspace/scripts \
    tensorrt-8.6.1-benchmark
```


```
# Check that your image exists
sudo docker images | grep tensorrt-8.6.1-benchmark

# Start the container
sudo docker run -it --rm \
    --runtime nvidia \
    --gpus all \
    --network host \
    --privileged \
    -v /home/copter/engine_models:/workspace/engines \
    -v /home/copter/jetson_benchmark/benchmarking:/workspace/scripts \
    tensorrt-8.6.1-benchmark

# Its running you see this
# (base) copter@ubuntu:~$ --> root@ubuntu:/workspace#
```