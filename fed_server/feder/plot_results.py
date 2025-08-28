import matplotlib.pyplot as plt
import json

def plot_accuracy(log_file="log.json"):
    with open(log_file, "r") as f:
        data = json.load(f)
    rounds = list(range(1, len(data["accuracy"]) + 1))
    plt.plot(rounds, data["accuracy"], marker='o')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Test Accuracy per Round")
    plt.grid()
    plt.show()
