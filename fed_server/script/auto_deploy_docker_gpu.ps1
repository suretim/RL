# ==========================
# Windows + WSL2 + Docker GPU 一键安装脚本
# 运行前：右键 PowerShell → 以管理员身份运行
# ==========================

Write-Host "=== 1. 启用 WSL2 ==="
wsl --install
wsl --set-default-version 2

Write-Host "=== 2. 检查并提示安装 NVIDIA 驱动（WSL 支持版） ==="
Write-Host "请到 https://www.nvidia.com/Download/index.aspx 下载并安装最新驱动（>=465）"
pause

Write-Host "=== 3. 安装 Ubuntu 作为默认 WSL2 发行版 ==="
wsl --install -d Ubuntu
wsl --set-version Ubuntu 2
wsl --set-default Ubuntu

Write-Host "=== 4. 安装 CUDA 工具包（WSL2 内执行） ==="
wsl bash -c "
  set -e
  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
  sudo apt-key del 7fa2af80 || true
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/\${distribution}/x86_64/cuda-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/\${distribution}/x86_64/ /' | sudo tee /etc/apt/sources.list.d/cuda.list
  sudo apt update
  sudo apt install -y cuda
"

Write-Host "=== 5. 安装 Docker Desktop（请手动点击安装包） ==="
Write-Host "下载地址：https://www.docker.com/products/docker-desktop/"
pause

Write-Host "=== 6. 配置 Docker Desktop 使用 WSL2 ==="
Write-Host "请打开 Docker Desktop → Settings → General → 勾选 'Use the WSL 2 based engine'"
Write-Host "Settings → Resources → WSL Integration → 勾选 Ubuntu"
pause

Write-Host "=== 7. 安装 NVIDIA Docker 支持（WSL2 内执行） ==="
wsl bash -c "
  set -e
  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt update
  sudo apt install -y nvidia-docker2
  sudo service docker restart || true
"

Write-Host "=== 8. 测试 GPU 容器 ==="
wsl bash -c "docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi"

Write-Host "=== 安装完成！Docker + GPU 环境已就绪 ==="
