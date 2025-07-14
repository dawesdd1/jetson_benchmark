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
conda create -n FastSAM python=3.9 -y

conda activate FastSAM

pip install git+https://github.com/openai/CLIP.git
```