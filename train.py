# train_cnn.py – PyTorch 2.0.1 compatible, genera 1 solo modelo .onnx
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

print(f"PyTorch version: {torch.__version__}")

# 1. Cargar FashionMNIST
def load_data(batch_size=64):
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 2. Modelo CNN
class CNNFashion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # (1,28,28) -> (16,28,28)
        self.pool = nn.MaxPool2d(2, 2)                # (16,28,28) -> (16,14,14)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # (16,14,14) -> (32,14,14)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)              # (32,7,7)
        x = x.view(-1, 32 * 7 * 7)    # Aplanar
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Entrenamiento
def train(model, loader, epochs=3):
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in loader:
            opt.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

# 4. Exportar a ONNX
def export_onnx(model, path):
    print(f"Exportando modelo a {path}...")
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=15,
    )
    print("✅ Modelo ONNX exportado con éxito.")

# 5. Main
if __name__ == "__main__":
    model_path = "cnn_fashion.onnx"
    loader = load_data()
    model = CNNFashion()
    print("Entrenando modelo CNN...")
    train(model, loader, epochs=3)
    export_onnx(model, model_path)

    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"\nArchivo generado: {model_path} ({size_kb:.2f} KB)")
        if os.path.exists(model_path + ".data"):
            print("⚠️ Se generó un .data (error).")
        else:
            print("✅ Confirmado: solo 1 archivo .onnx.")
