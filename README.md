# jetson_benchmark



## MobileSAM instructions
```bash 
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

```bash 
conda create -n mobilesam python=3.8 -y

conda activate mobilesam

cd MobileSAM; pip install -e .
```



## FastSAM instructions

```bash 
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
```


```bash 
conda create -n FastSAM python=3.9 -y # may actually need 3.10 bc of Nvidia & PyTorch wheel

conda activate FastSAM

pip install git+https://github.com/openai/CLIP.git

# On Jetson Orin (Ubuntu 22.04 -- JP6.1 -- ARM64) 
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# (optional trouble shoot if lib cuSPARSELt in missing
wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

# 5. Install the cusparselt libraries using apt
sudo apt-get -y install libcusparselt0 libcusparselt-dev

# verify cuda enabled torch
python3 -c "import torch; print(torch.cuda.is_available())"

```