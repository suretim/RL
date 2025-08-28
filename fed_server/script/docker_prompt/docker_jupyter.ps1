# ===============================
# PowerShell 脚本：注册所有 Conda 环境 kernel
# ===============================

$Container = "jupyter_env"  # Docker 容器名

# 1️⃣ 获取容器内所有 Conda 环境
$envs_json = docker exec -i --user root $Container bash -c "conda env list --json"
$envs = ($envs_json | ConvertFrom-Json).envs

Write-Host "🔍 检测到 Conda 环境："
$envs | ForEach-Object { Write-Host " - $_" }

# 2️⃣ 遍历每个环境，生成脚本内容
foreach ($envPath in $envs) {
    $envName = Split-Path $envPath -Leaf
    $ScriptContent = @"
#!/bin/bash
set -e
echo "📦 注册 Conda 环境 kernel: $envName"
conda install -n $envName ipykernel -y
conda run -n $envName python -m ipykernel install --name $envName --display-name "Python ($envName)" --sys-prefix
echo "✅ $envName 注册完成"
"@

    # 转成 UTF8 bytes（无 BOM，LF 换行）
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($ScriptContent)

    # 脚本在容器内路径
    $ScriptPath = "/root/register_kernel_$envName.sh"

    # 写入容器
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "docker"
    $psi.Arguments = "exec -i --user root $Container bash -c 'cat > $ScriptPath'"
    $psi.RedirectStandardInput = $true
    $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
    $proc.StandardInput.BaseStream.Flush()
    $proc.StandardInput.Close()
    $proc.WaitForExit()

    # 执行脚本并删除
    docker exec -it --user root $Container bash -c "chmod +x $ScriptPath && bash $ScriptPath && rm -f $ScriptPath"
}

# 3️⃣ 显示所有 kernel
docker exec -i --user root $Container bash -c "jupyter kernelspec list"
Write-Host "🎉 所有 Conda 环境 kernel 注册完成，可在 Notebook/Lab 面板选择"

#docker exec -it --user root $Container bash -c "chmod +x $ScriptPath && bash $ScriptPath"
conda run -n myenv python -m ipykernel install --name myenv --display-name "my_env" --sys-prefix
#conda run -n \$ENV_NAME python -m ipykernel install --name \$ENV_NAME --display-name "Python (\$ENV_NAME)" --sys-prefix
docker exec -it --user root jupyter_env bash -c "conda run -n myenv python -m ipykernel install"
docker exec -it --user root jupyter_env /bin/bash -c "start-notebook.sh --NotebookApp.token='' --NotebookApp.allow_root=True"
# 启动 Jupyter Lab
jupyter lab --ip=0.0.0.0 --no-browser --allow-root












