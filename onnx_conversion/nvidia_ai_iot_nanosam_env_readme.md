# MobileSAM (Nvidia AI IOT)
Optimized for an x86 system (not for arm64 jetson)

NOTE
`chaoningzhang_mobile_sam.pt` is untrained and will need to be trained on a relevant dataset (ie. COCO 2017)

## Conda Env
```bash
conda create -n nanosam_clean python=3.10 -y
conda activate nanosam_clean

# torch2trt (on x86 VM)
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install

pip install torch>=1.7 torchvision>=0.8
pip install onnx==1.12.0 and onnxruntime==1.13.1 numpy==1.26.4 --default-timeout=1000   # (closest is onnx==1.13.1  && onnxruntime==1.14.1)

# typical np version mismatch
pip uninstall -y numpy
pip install "numpy>=1.19.2,<2.0"
```

## Install NanoSAM (Nvidia AI IOT) into conda env
```bash
git clone https://github.com/NVIDIA-AI-IOT/nanosam
cd nanosam
python3 setup.py develop --user
```

## Onnx Conversion
```bash
# Repo only mentions the decoder
# ALTERNATIVE: there were .onnx files only ready for download
python3 -m nanosam.tools.export_sam_mask_decoder_onnx \
    --model-type=vit_t \
    --checkpoint=assets/mobile_sam.pt \
    --output=data/mobile_sam_mask_decoder.onnx
```

## Training
```bash 
```

## TensorRT Conversion
```bash
# ENCODER
python3 examples/basic_usage.py \
    --image_encoder=data/resnet18_image_encoder.engine \
    --mask_decoder=data/mobile_sam_mask_decoder.engine

# DECODER
trtexec \
    --onnx=data/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10
```