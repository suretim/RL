import json
import numpy as np
import grpc

# 引入 proto 定义
import model_pb2
import model_pb2_grpc

# ======= 模拟的 _on_message =========
def _on_message(client, userdata, msg_payload):
    try:
        # 解码 JSON
        message = json.loads(msg_payload)

        # 提取字段
        client_request = int(message.get("client_request", 0))
        client_id = str(message.get("client_id", "1"))

        print(f"[TEST] client_request={client_request}, client_id={client_id}")

        if client_request == 1:
            print("[TEST] client_request=1, 模拟 publish ACK")
            return

        fea_weights = message.get("fea_weights", [])
        fea_labels = message.get("fea_labels", [])

        # 确保都是 list
        if not isinstance(fea_weights, list):
            fea_weights = [fea_weights]
        if not isinstance(fea_labels, list):
            fea_labels = [fea_labels]

        # 拼接 + flatten
        fea_vec = np.array(fea_labels + fea_weights, dtype=float).flatten().tolist()

        print(f"[TEST] fea_vec 长度 = {len(fea_vec)}")
        print(f"[TEST] 前 5 个值 = {fea_vec[:5]}")

        # ====== gRPC 上传 ======
        grpc_channel = grpc.insecure_channel("localhost:50051")  # 改成你的 GRPC_SERVER
        stub = model_pb2_grpc.FederatedLearningStub(grpc_channel)

        request = model_pb2.ModelParams(client_id=int(client_id), values=fea_vec)
        response = stub.UploadModelParams(request)

        print(f"[TEST] gRPC server response: {response.message}")

    except Exception as e:
        print(f"[ERROR] {e}")


# ======= 主程序，模拟 MQTT 消息 =======
if __name__ == "__main__":
    # 这是你日志里的 JSON
    fake_msg = """{
        "fea_weights":[0.360249,0.316236,0.323515,0,0.0600093,0,0.0352879,0,0,0.00656125,
                       0.040056,0,0,0,0,0,0.0140805,0,0,0.0584615,0,0,0,0,0.0472893,
                       0,0,0.061304,0.00401666,0.0320163,0,0,0.108493,-0.0218137,0.00094423,
                       0,0,0,0,0,0,0,0,0,0,0.0133623,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "fea_labels":[0],
        "client_request":0,
        "client_id":46087134
    }"""

    # 调用测试函数
    _on_message(client=None, userdata=None, msg_payload=fake_msg)
