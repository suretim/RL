# client.py
import flwr as fl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(1):  # local epoch
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                output = self.model(images)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()

    def test(self):
        self.model.eval()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                loss += F.cross_entropy(outputs, labels, reduction='sum').item()
                correct += (outputs.argmax(1) == labels).sum().item()
        return loss / len(self.test_loader.dataset), correct / len(self.test_loader.dataset)

# 启动客户端
model = SimpleCNN()
train_loader = DataLoader(client_datasets[0], batch_size=32, shuffle=True)
test_loader = DataLoader(client_datasets[0], batch_size=32)
fl.client.start_numpy_client(server_address="localhost:8080", client=CifarClient(model, train_loader, test_loader))
