import flwr as fl
import matplotlib.pyplot as plt
def main():


    # 记录每轮聚合后的准确率
    accuracies = []

    def on_round_end(server_round, results, failures):
        # 获取每轮聚合后的评估结果
        accuracy = np.mean([res.metrics["accuracy"] for res in results])
        accuracies.append(accuracy)

        print(f"Round {server_round} - Average accuracy: {accuracy:.4f}")
        # 可视化结果
        plt.plot(range(len(accuracies)), accuracies, label="Accuracy")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")
        plt.title("Federated Learning Accuracy")
        plt.pause(0.01)  # 更新图形

    # 使用 FedAvg 聚合策略
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,  # 50% 的客户端参与每轮训练
        fraction_evaluate=0.5,  # 50% 的客户端评估
        min_fit_clients=2,  # 至少 2 个客户端进行训练
        min_eval_clients=2,  # 至少 2 个客户端进行评估
        min_available_clients=2,  # 至少 2 个客户端可用
        evaluate_metrics=["accuracy"],
        on_round_end=on_round_end  # 每轮聚合结束时调用
    )

    # 启动 Flower 服务器
    #fl.server.start_server(server_address="localhost:8081", strategy=strategy)

    #strategy = fl.server.strategy.FedAvg(
    #    min_fit_clients=2,
    #    min_available_clients=2,
    #    #eval_fn=None,
    #    on_fit_config_fn=lambda rnd: {"rnd": rnd},
    #)
    from flwr.server import ServerConfig

    fl.server.start_server(
        server_address="localhost:8081",
        config=ServerConfig(num_rounds=5),  # ✅ 使用 ServerConfig 对象
        strategy=strategy
    )
    #fl.server.start_server(server_address="localhost:8080", config={"num_rounds": 5}, strategy=strategy)

if __name__ == "__main__":
    main()
