@echo off
REM 一键创建 TF 和 Flower 两个环境，解决 NumPy 版本冲突
rem net use F: "\\vmware-host\Shared Folders" /persistent:yes
echo ===== 创建 TensorFlow 环境 (tf_env) =====
conda create -n tf_env python=3.10 -y
call conda activate tf_env

echo 安装 TensorFlow 2.12 + NumPy 1.23.5 + Pillow
pip install numpy==1.23.5
pip install tensorflow==2.12.0
pip install pillow

echo 测试 TF 环境版本
python -c "import tensorflow as tf; import numpy as np; print('TF env ->', tf.__version__, np.__version__)"

echo ===== 创建 Flower 环境 (flower_env) =====
conda create -n flower_env python=3.10 -y
call conda activate flower_env

echo 安装 Flower 1.20 + NumPy 1.26.6
pip install numpy==1.26.6
pip install flwr==1.20.0
pip install tensorflow==2.15.0
keras==2.12.0
h5py==3.8.0
pillow==9.5.0
tensorflow-hub==0.13.0
pip install argparse
pip install json
pip install base64
echo 测试 Flower 环境版本
python -c "import flwr as fl; import numpy as np; print('Flower env ->', fl.__version__, np.__version__)"

echo ===== 初始化完成 =====
echo 使用 'conda activate tf_env' 或 'conda activate flower_env' 进入对应环境运行任务
pause
