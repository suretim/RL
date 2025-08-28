import grpc
from concurrent import futures
import time

import model_pb2
import model_pb2_grpc


# === 实现 FederatedLearning 服务 ===
class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def UploadModelParams(self, request, context):
        print(f"[SERVER] 收到请求: client_id={request.client_id}, "
              f"param_type={request.param_type}, values_len={len(request.values)}")

        # 随便返回一个结果
        return model_pb2.UploadReply(
            update_successful=True,
            message=f"收到 {len(request.values)} 个参数，来自 client {request.client_id}"
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)
    server.add_insecure_port("[::]:50051")  # 本地 50051 端口
    server.start()
    print("[SERVER] Mock FederatedLearning gRPC server 已启动，监听 :50051")
    try:
        while True:
            time.sleep(86400)  # 一天
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
