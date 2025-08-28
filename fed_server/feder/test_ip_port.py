import socket

host = "192.168.0.57"  # 使用 IP 地址
port = 1883

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to {host}:{port}")
except socket.gaierror as e:
    print(f"getaddrinfo failed: {e}")
except socket.error as e:
    print(f"Connection error: {e}")
