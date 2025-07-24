# SAM_TensorRT
Optimized for an Jetson Arm64 system

NOTE
** MAY BE untrained and will need to be trained on a relevant dataset (ie. COCO 2017)



## apt prerequisite packages
```bash
sudo apt-get update
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
```


## Conda Env
```bash
conda create -n nanosam_arm64 python=3.10 -y
conda activate nanosam_arm64

# COPY over trt into your chosen conda env
ls /usr/lib/python3.10/dist-packages/tensorrt/tensorrt.so
cp -r /usr/lib/python3.10/dist-packages/tensorrt* $CONDA_PREFIX/lib/python3.10/site-packages/
python -c "import tensorrt; print(tensorrt.__version__)"

# torch2trt (on x86 VM)
# git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd ~/torch2trt
python setup.py install

pip install torch>=1.7 torchvision>=0.8
pip install onnx==1.13.1 onnxruntime==1.14.1 numpy==1.26.4 --default-timeout=1000 

# typical np version mismatch
pip uninstall -y numpy
pip install "numpy>=1.19.2,<2.0"
```

## Install NanoSAM (Nvidia AI IOT) into conda env
```bash
# git clone https://github.com/zhudongwork/SAM_TensorRT.git
cd ~/SAM_TensorRT
python setup.py develop --user
```

## Training
```bash
```

## Onnx -->> TensorRT Conversion

```bash
# TensorRT (use system engine)
conda deactivate
sudo find / -name trtexec 2>/dev/null
export PATH=$PATH:/usr/src/tensorrt/bin
trtexec --help

# ENCODER
trtexec \
    --onnx=/home/copter/onnx_models/zhudongwork_mobile_sam_encoder.onnx \
    --saveEngine=/home/copter/engine_models/zhudongwork_mobile_sam_encoder_fp32_trt1030.engine \
    --memPoolSize=workspace:10240

trtexec \
    --onnx=/home/copter/onnx_models/zhudongwork_mobile_sam_encoder.onnx \
    --saveEngine=/home/copter/engine_models/zhudongwork_mobile_sam_encoder_fp16_trt1030.engine \
    --memPoolSize=workspace:10240 \
    --fp16


# DECODER
trtexec \
    --onnx=/home/copter/onnx_models/zhudongwork_mobile_sam_decoder.onnx \
    --saveEngine=/home/copter/engine_models/zhudongwork_mobile_sam_decoder_fp32_trt1030.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10


trtexec \
    --onnx=/home/copter/onnx_models/zhudongwork_mobile_sam_decoder.onnx \
    --saveEngine=/home/copter/engine_models/zhudongwork_mobile_sam_decoder_fp16_trt1030.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10 \
    --fp16


## ORIGINAL METHOD USED TensorRT 8.5.3.1
/usr/local/TensorRT-8.5.3.1/bin/trtexec --onnx=./models/mobile_sam_encoder.onnx --saveEngine=./models/mobile_sam_encoder.engine --memPoolSize=workspace:10240

/usr/local/TensorRT-8.5.3.1/bin/trtexec --onnx=./models/mobile_sam_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10 --saveEngine=./models/mobile_sam_decoder.engine

```

## Basic Engine Usage 
```bash
python3 examples/basic_usage.py \
    --image_encoder=data/resnet18_image_encoder.engine \
    --mask_decoder=data/mobile_sam_mask_decoder.engine
```