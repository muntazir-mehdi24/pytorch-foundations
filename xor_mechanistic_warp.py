import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from auto_logger import log_message

class XOR_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

torch.manual_seed(42)
model = XOR_MLP()
print(model)
    
# Dataset
inputs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

targets = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Dual-Plot Live Visualization Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

# Left grid: Input space (-0.2 to 1.2)
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100), np.linspace(-0.2, 1.2, 100))
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Training Loop with Dual Live Animation
for epochs in range(2000):
    optimizer.zero_grad()
    preds = model(inputs)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    
    if epochs % 1000 == 0:
        print(f"Epoch {epochs} || Total Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            # 1. Left plot data (Input space decision boundary)
            Z = model(grid_tensor).numpy().reshape(xx.shape)
            
            # 2. Right plot data (Hidden layer feature space coordinates)
            #hidden_coords = torch.sigmoid(model.fc1(inputs)).numpy()
            
        # --- Update Left Subplot (Input Space) ---
        ax.clear()
        ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.8)
        # Color points by target: Target 0 = Blue, Target 1 = Red
        colors = ['blue' if t.item() == 0 else 'red' for t in targets]
        ax.scatter(inputs[:, 0].numpy(), inputs[:, 1].numpy(), c=colors, s=120, edgecolors='black', linewidth=1.5)
        ax.set_title(f"Input Space (Epoch {epochs})")
        ax.set_xlabel("Input x1")
        ax.set_ylabel("Input x2")

                
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        log_message(model, loss, epochs)

plt.ioff()

print("="*150)

# get weights and biases from the first layer
weights = model.fc2.weight.data.numpy()
biases = model.fc2.bias.data.numpy()


# plug them in the form of equations
for i in range(weights.shape[0]):
    w1, w2 = weights[i]
    b = biases[i]
    print(f"Equation for Neuron {i}: x2 = {-(w1/w2):.2f} * x1 + {-(b/w2):.2f}")


print("Final Evaluation & Internal Inspection Complete.")
plt.show()