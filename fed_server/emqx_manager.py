import os
import argparse
import platform
import subprocess
import time
import requests
import zipfile
import tarfile
INSTALL_DIR="c:\\emqt\\"
VERSION="5.0.26"
class EMQXServer:
    def __init__(self, version=None,install_dir=None):
        self.version = version
        self.install_dir=install_dir   #"c:\\emqt\\"
        self.system = platform.system().lower()
        self.arch = "amd64" if platform.machine().endswith('64') else "386"
        #self.install_dir = os.path.join(os.getcwd(), f"emqx-{self.version}")
        self.install_dir = os.path.join(self.install_dir, f"emqx-{self.version}-windows-amd64\\")
        self.bin_path0 = os.path.join(self.install_dir, "bin", "emqx")
        self.conf_path = os.path.join(self.install_dir, "etc", "emqx.conf")
        self.bin_path1 = self.bin_path0+" start -c "+self.conf_path
        self.bin_path  = os.path.join(self.install_dir, "bin", "emqx")

    def download_emqx(self):
        base_url = "https://www.emqx.com/en/downloads/broker"

        if self.system == "windows":
            url = f"{base_url}/{self.version}/emqx-{self.version}-windows-{self.arch}.zip"
            local_file = f"emqx-{self.version}-windows.zip"
        else:
            url = f"{base_url}/{self.version}/emqx-{self.version}-{self.system}-{self.arch}.tar.gz"
            local_file = f"emqx-{self.version}-{self.system}.tar.gz"

        print(f"Downloading EMQX {self.version} from {url}")
        response = requests.get(url, stream=True)
        with open(local_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting...")
        if self.system == "windows":
            with zipfile.ZipFile(local_file, 'r') as zip_ref:
                zip_ref.extractall()
        else:
            with tarfile.open(local_file, 'r:gz') as tar_ref:
                tar_ref.extractall()

        os.remove(local_file)
        print(f"EMQX installed to {self.install_dir}")

    def start_server(self):
        if self.system == "windows":
            command = f"{self.bin_path1}"
        else:
            command = f"chmod +x {self.bin_path} && {self.bin_path} start"

        print("Starting EMQX server...")
        print(self.bin_path1)
        subprocess.Popen(command, shell=True, cwd=self.install_dir)

        # Wait for startup
        time.sleep(5)
        print(f"EMQX started. Dashboard: http://localhost:18083 (admin/public)")

    def stop_server(self):
        print("Stopping EMQX server...")
        subprocess.run(f"{self.bin_path} stop", shell=True, cwd=self.install_dir)
        print("EMQX stopped")

    def check_status(self):
        try:
            result = subprocess.run(
                f"{self.bin_path} ping",
                shell=True,
                cwd=self.install_dir,
                capture_output=True,
                text=True
            )
            return "pong" in result.stdout
        except:
            return False


def main(args):
    # 更新全局变量
    global  INSTALL_DIR , VERSION 
    INSTALL_DIR = args.install_dir 
    VERSION = args.version 
    try:
        emqx = EMQXServer(version=args.version,install_dir=args.install_dir)  # 可修改版本号
        emqx.start_server()
        time.sleep(4)
        if emqx.check_status():
            print("Server is running")
        else:
            print("Server uncheck!!!")
            emqx.stop_server()

        # 保持运行直到用户中断
        print("netstat -ano | findstr :18083 Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        emqx.stop_server()


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--install_dir", type=str, default=INSTALL_DIR) 
    parser.add_argument("--version", type=str, default=VERSION) 
    args = parser.parse_args()
     
    
    main(args)
    #if not os.path.exists(emqx.install_dir):
    #    emqx.download_emqx()

    