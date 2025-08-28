param(
    [Parameter(Mandatory = $true)]
    [string]$EnvInput
)
#Usage
# 删除路径形式的环境
#.\remove-conda-env.ps1 "D:\conda_envs\tf_hub_env"

# 删除环境名形式
#.\remove-conda-env.ps1 "tf_hub_env"


# 检查是否包含路径分隔符（判定是路径还是环境名）
if ($EnvInput -match '[\\/]+') {
    Write-Host "删除 Conda 环境（按路径）: $EnvInput" -ForegroundColor Yellow
    conda remove --prefix $EnvInput --all -y
} else {
    Write-Host "删除 Conda 环境（按名称）: $EnvInput" -ForegroundColor Yellow
    conda remove --name $EnvInput --all -y
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 删除完成" -ForegroundColor Green
} else {
    Write-Host "❌ 删除失败" -ForegroundColor Red
}
