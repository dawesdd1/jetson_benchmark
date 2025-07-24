# FastSam_Awsome_TensorRT & TensorRT-FastSAM

Covers the onnx creation part of FastSam_Awsome_TensorRT
Pipeline: .pt --> .onnx --> .engine

## Get the correct repo
```bash
git clone https://github.com/Linaom1214/TensorRT-FastSAM.git
cd TensorRT-FastSAM/
```


## Conda Env
```bash
conda create -n fastsam_awesome_onnx python=3.10 -y
conda activate fastsam_awesome_onnx
pip install torch>=1.7 torchvision>=0.8
pip install onnx==1.13.1 onnxruntime==1.14.1  --default-timeout=1000
pip install Pillow>=7.1.2 matplotlib>=3.2.2 numpy==1.26.4 ultralytics==8.0.120 --default-timeout=1000
pip install git+https://github.com/openai/CLIP.git

# typical np version mismatch
pip uninstall -y numpy
pip install "numpy>=1.19.2,<2.0"
```


## FastSAM PyTorch Check -->> Onnx Conversion
```bash
# python export.py --weights <path_to_FastSAM-x.pt> --output <onnx_model_path>

##--------- example: size-x ---------- 
python export.py \
--weights /home/dawesdd1/repos/onnx_and_pt_weights/CASIA-IVA-Lab_FastSAM-x.pt --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/CASIA-IVA-Lab_FastSAM-x.onnx

python export.py \
--weights /home/dawesdd1/repos/onnx_and_pt_weights/CASIA-IVA-Lab_FastSAM-x.pt --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/CASIA-IVA-Lab_FastSAM-x_imgz1024.onnx --max_size 1024

python export.py \
--weights /home/dawesdd1/repos/onnx_and_pt_weights/CASIA-IVA-Lab_FastSAM-x.pt --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/CASIA-IVA-Lab_FastSAM-x_imgz640.onnx --max_size 640

python export.py \
--weights /home/dawesdd1/repos/onnx_and_pt_weights/CASIA-IVA-Lab_FastSAM-x.pt --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/CASIA-IVA-Lab_FastSAM-x_imgz512.onnx --max_size 512

##--------- example: size-s ---------- 
python export.py \
--weights /home/dawesdd1/repos/onnx_and_pt_weights/CASIA-IVA-Lab_FastSAM-s.pt --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/CASIA-IVA-Lab_FastSAM-s.onnx

python export.py \
--weights /home/dawesdd1/repos/onnx_and_pt_weights/CASIA-IVA-Lab_FastSAM-s.pt --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/CASIA-IVA-Lab_FastSAM-s_imgz1024.onnx --max_size 1024

python export.py \
--weights /home/dawesdd1/repos/onnx_and_pt_weights/CASIA-IVA-Lab_FastSAM-s.pt --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/CASIA-IVA-Lab_FastSAM-s_imgz640.onnx --max_size 640

python export.py \
--weights /home/dawesdd1/repos/onnx_and_pt_weights/CASIA-IVA-Lab_FastSAM-s.pt --output /home/dawesdd1/repos/onnx_and_pt_weights/conversion_output/CASIA-IVA-Lab_FastSAM-s_imgz512.onnx --max_size 512
```
