import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class XOR_MLP(nn.Module):

	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(2,2)
		self.fc2 = nn.Linear(2,1)

	def forward(self,x):
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		return x

torch.manual_seed(42)
model = XOR_MLP()
print(model)
	
# dataset

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

# loss fn and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Live plotting setup
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100), np.linspace(-0.2, 1.2, 100))
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# training loop

for epochs in range(10000):
    optimizer.zero_grad()
    preds = model(inputs)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    
    if epochs % 2000 == 0:
        print(f"Epoch {epochs} || Total Loss: {loss.item():.4f}")
        
        # Live boundary visualization update
        with torch.no_grad():
            Z = model(grid_tensor).numpy().reshape(xx.shape)
            
        ax.clear()
        ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.8)
        ax.scatter([0, 1], [0, 1], color='blue', s=100, label='Target 0')
        ax.scatter([0, 1], [1, 0], color='red', s=100, label='Target 1')
        ax.set_title(f"Epoch {epochs} | Loss: {loss.item():.4f}")
        ax.legend(loc='upper right')
        plt.draw()
        plt.pause(0.01)

plt.ioff()

print("="*150)
print("Final Eval.")
with torch.no_grad():
    final_preds = model(inputs)
    for i in range(len(inputs)):
        print(f"Inputs: {inputs[i].tolist()} | Target: {targets[i].item()} | Guess: {final_preds[i].item():.4f} (Rounded: {round(final_preds[i].item())})")

print("="*150)
print(f"Hidden layer")
print(f"weights \n{model.fc1.weight.data}")
print(f"Biases \n{model.fc1.bias.data}")

print("\n--- Hidden Layer Activations (The Warp) ---")
with torch.no_grad():
    hidden_out = torch.sigmoid(model.fc1(inputs))
    for i in range(len(inputs)):
        print(f"Input: {inputs[i].tolist()} --> Hidden Space Coordinates: [{hidden_out[i][0]:.4f}, {hidden_out[i][1]:.4f}] --> Guess: {final_preds[i].item():.4f}")

print("\n--- Layer 2 (Output Weights) ---")
print("Weights:", model.fc2.weight.data)
print("Bias:", model.fc2.bias.data)

print("\n--- SURGERY: Killing Neuron 1 ---")
with torch.no_grad():
    # Force all weights and biases of the first hidden neuron to zero
    model.fc1.weight.data[0] = 0.0
    model.fc1.bias.data[0] = 0.0

    # Test the lobotomized model
    surgery_preds = model(inputs)
    for i in range(len(inputs)):
        print(f"Input: {inputs[i].tolist()} | Target: {targets[i].item()} | Lobotomized Guess: {surgery_preds[i].item():.4f}")

# Keep the final plot window open until manually closed
plt.show()