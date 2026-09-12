import torch
import torch.nn as nn
import torch.optim as optim

class XOR_MLP(nn.Module):

	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(2,2)
		self.fc2 = nn.Linear(2,1)

	def forward(self,x):
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		return x

model =  XOR_MLP()
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

# training loop

for epochs in range(100000):
    optimizer.zero_grad()
    preds = model(inputs)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    
    if epochs % 20000 == 0:
        print(f"Epoch {epochs} || Total Loss: {loss.item():.4f}")


print("="*150)
print("Final Eval.")
with torch.no_grad():
    final_preds = model(inputs)
    for i in range(len(inputs)):
        print(f"Inputs: {inputs[i].tolist()} | Target: {targets[i].item()} | Guess: {final_preds[i].item():.4f} (Rounded: {round(final_preds[i].item())})")
		