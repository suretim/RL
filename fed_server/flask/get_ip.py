import socket
import psutil  # pip install psutil

def get_local_ip():
    candidates = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ip = addr.address
                # 过滤掉 127.x.x.x 和 169.254.x.x
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                # 只收集常见内网网段
                if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
                    candidates.append(ip)
    return candidates[0] if candidates else "127.0.0.1"

if __name__ == "__main__":
    ip = get_local_ip()
    print(f"💡 你的局域网 IP 是: {ip}")
    print(f"👉 ESP32 OTA URL 示例: http://{ip}:5001/ota_model")
    print(f"👉 ESP32 MD5 URL 示例: http://{ip}:5001/ota_model_md5")
