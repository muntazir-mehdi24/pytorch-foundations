import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class XOR_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# Left grid: Input space (-0.2 to 1.2)
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100), np.linspace(-0.2, 1.2, 100))
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Training Loop with Dual Live Animation
for epochs in range(10000):
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
            hidden_coords = torch.sigmoid(model.fc1(inputs)).numpy()
            
        # --- Update Left Subplot (Input Space) ---
        ax1.clear()
        ax1.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.8)
        # Color points by target: Target 0 = Blue, Target 1 = Red
        colors = ['blue' if t.item() == 0 else 'red' for t in targets]
        ax1.scatter(inputs[:, 0].numpy(), inputs[:, 1].numpy(), c=colors, s=120, edgecolors='black', linewidth=1.5)
        ax1.set_title(f"Input Space (Epoch {epochs})")
        ax1.set_xlabel("Input x1")
        ax1.set_ylabel("Input x2")

        # --- Update Right Subplot (Hidden Space Warp) ---
        ax2.clear()
        ax2.scatter(hidden_coords[:, 0], hidden_coords[:, 1], c=colors, s=120, edgecolors='black', linewidth=1.5)
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.1, 1.1)
        ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title(f"Hidden Space Warp [h1 vs h2]")
        ax2.set_xlabel("Hidden Neuron 1 Output")
        ax2.set_ylabel("Hidden Neuron 2 Output")
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

plt.ioff()

print("="*150)

plt.show()