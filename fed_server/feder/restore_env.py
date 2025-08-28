import os
import shutil
from dotenv import load_dotenv
#python -m venv .venv2
#Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
#.venv\Scripts\activate
#pip freeze > requirements.txt
#pip install -r requirements.txt
ENV_FILE = ".env"
BACKUP_FILE = ".env.backup"

# 如果 .env.backup 不存在但 .env 存在，创建备份
if not os.path.exists(BACKUP_FILE):
    if os.path.exists(ENV_FILE):
        shutil.copy(ENV_FILE, BACKUP_FILE)
        print(f"[📦] Backup created: '{ENV_FILE}' → '{BACKUP_FILE}'")
    else:
        print("[❌] Neither .env nor .env.backup exists. Nothing to do.")
        exit(1)

# 如果 .env.backup 存在，则恢复为当前 .env
else:
    shutil.copy(BACKUP_FILE, ENV_FILE)
    print(f"[✅] Restored '{BACKUP_FILE}' → '{ENV_FILE}'")

# 加载环境变量
load_dotenv(ENV_FILE)

# 可验证的一些关键变量（请按实际修改）
TEST_KEYS = ["API_KEY", "MQTT_HOST", "MODEL_VERSION"]

print("\n[🔍] Loaded environment variables:")
for key in TEST_KEYS:
    value = os.getenv(key)
    if value:
        print(f"  {key} = {value}")
    else:
        print(f"  {key} is NOT set ❌")
