# MobileSAM (& MobileSAM v2)
Optimized for an x86 system (not for arm64 jetson)

NOTE
`chaoningzhang_mobile_sam.pt` is just the prompt encoder + mask decoder (i.e. the bit that takes image embeddings + your prompt and spits out masks). It does not include the image encoder that turns raw RGB pixels into those embeddings

## Conda Env
```bash
conda create -n mobilesam_onnx python=3.10 -y
conda activate mobilesam_onnx
pip install torch>=1.7 torchvision>=0.8
pip install onnx==1.12.0 onnxruntime==1.13.1 numpy==1.26.4 --default-timeout=1000   # (closest is onnx==1.13.1  && onnxruntime==1.14.1)

# typical np version mismatch
pip uninstall -y numpy
pip install "numpy>=1.19.2,<2.0"
```

## Install MobileSAM (ChaoningZhang) into conda env
```bash
cd ~/repos/MobileSAM
pip install -e .
cd ~/repos
```
## Onnx Conversion
```bash
python /home/dawesdd1/repos/jetson_benchmark/onnx_conversion/chaoning_export_onnx_model.py \
--checkpoint /home/dawesdd1/repos/onnx_and_pt_weights/chaoningzhang_mobile_sam.pt --model-type vit_t --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/chaoningzhang_mobile_sam.onnx
```