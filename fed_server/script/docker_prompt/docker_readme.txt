
Ampere
NVIDIA GeForce RTX 4070 Ti
# 构建镜像
docker build -t tf-2.12-gpu-fixed .

# 运行测试
docker run --gpus all tf-2.12-gpu-fixed

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

nvidia-smi
nvcc --version
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\demo_suite"
.\bandwidthTest.exe  # 应输出 "Result = PASS"

检查网络连接
docker pull registry.cn-hangzhou.aliyuncs.com/tensorflow/tensorflow:2.12.0-gpu
ping registry-1.docker.io
curl -v https://registry-1.docker.io/v2/
docker images
#重要
#docker run --gpus all -it tensorflow/tensorflow:2.12.0-gpu python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
docker run --gpus all -it tensorflow/tensorflow:2.12.0-gpu
docker run -it --gpus all -v C:\tim:/workspace tensorflow/tensorflow:2.12.0-gpu bash
docker run -it --gpus all -v D:\RL:/workspace tensorflow/tensorflow:2.12.0-gpu bash
docker exec -it hardcore_wilson bash
docker ps
docker stop a1b2c3d4e5f6  # 使用容器ID或名称
docker rm a1b2c3d4e5f6  # 使用容器ID或名称
docker rmi a1b2c3d4e5f6  # 使用镜像ID或名称
docker logs a1b2c3d4e5f6  # 使用容器ID或名称
docker rmi tensorflow/tensorflow:2.12.0-gpu
docker system prune -a
pip install -r lab_requirements.txt
pillow==9.5.0
h5py==3.8.0
tensorflow-hub==0.13.0


挂载临时文件（动态修改）
docker run -v /host/path/file.txt:/app/file.txt my-image
或者安装 VS Code 的 CLI 版（code-server）
curl -fsSL https://code-server.dev/install.sh | sh

docker run -d -p 8080:8080 -v "$(pwd):/vit"  --name my-vscode  codercom/code-server

sudo systemctl enable --now code-server@$USER
echo "实际密码是: $(grep 'password:' ~/.config/code-server/config.yaml | awk '{print $2}')"
9261dae8d0b69fabe7faf131
a5a18825bfed7487839d1cbc
code-server --bind-addr 0.0.0.0:8080 --auth password

docker exec -it --user root my-vscode /bin/bash

docker export my-vscode -o my-vscode-container.tar
docker import my-vscode-container.tar my-vscode-restored:latest


docker save -o my-vscode-backup.tar my-vscode-backup:latest
docker load -i my-vscode-backup.tar

跑GPU
docker run --gpus all nvidia/cuda:12.2.0-base nvidia-smi
wsl --install
wsl.exe --install Ubuntu-24.04
# 进入 PyTorch GPU 容器并访问 Windows 项目
docker run -it --gpus all -v C:\Users\你的用户名\projects:/workspace pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime bash

# 进入 TensorFlow GPU 容器
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
docker run -dit --gpus all -v C:\tim:/workspace tensorflow/tensorflow:2.14.0-gpu bash
conda run -n myenv python -m ipykernel install --name my_env --display-name "my_env" --sys-prefix

docker run -dit  --user root jupyter_env -p 9999:8888 -v C:/tim:/home/jovyan/work jupyter/base-notebook -c "start-notebook.sh --NotebookApp.token='' --NotebookApp.allow_root=True"
docker exec -dit --user root jupyter_env /bin/bash 
docker run -dit --user root -p 8888:8888 -v C:/tim:/home/jovyan/work jupyter/base-notebook start-notebook.sh --NotebookApp.token='' --NotebookApp.allow_root=True
docker exec -dit --user root jupyter_env /bin/bash start-notebook.sh --NotebookApp.token='' --NotebookApp.allow_root=True 
docker run --hostname=0d5ce201c9d5 --user=1000 --mac-address=62:25:b1:36:fc:08 --env=PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --env=DEBIAN_FRONTEND=noninteractive --env=CONDA_DIR=/opt/conda --env=SHELL=/bin/bash --env=NB_USER=jovyan --env=NB_UID=1000 --env=NB_GID=100 --env=LC_ALL=en_US.UTF-8 --env=LANG=en_US.UTF-8 --env=LANGUAGE=en_US.UTF-8 --env=HOME=/home/jovyan --env=JUPYTER_PORT=8888 --volume=C:/tim:/home/jovyan/work --network=bridge --workdir=/home/jovyan -p 8888:8888 --restart=no --label='maintainer=Jupyter Project <jupyter@googlegroups.com>' --label='org.opencontainers.image.ref.name=ubuntu' --label='org.opencontainers.image.version=22.04' --runtime=runc -t -d jupyter/base-notebook