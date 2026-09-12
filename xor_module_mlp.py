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

dataset = [
	(torch.tensor([0.0, 0.0]), torch.tensor([0.0])),
	(torch.tensor([1.0, 0.0]), torch.tensor([1.0])),
	(torch.tensor([0.0, 1.0]), torch.tensor([1.0])),
	(torch.tensor([1.0, 1.0]), torch.tensor([0.0]))
]

# loss fn and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.11)

# training loop

for epochs in range(50000):

	epoch_loss = 0.0

	for inputs, target in dataset:
		
		# zero out gradients from previous step
		optimizer.zero_grad()

		# forward pass 
		pred = model(inputs)

		# loss 
		loss = criterion(pred, target)
		epoch_loss += loss.item()

		# backward pass
		loss.backward()

		# optimizer step
		optimizer.step()
	
	if epochs % 5000 == 0:
		print(f"Epoch {epochs} || Total Loss: {epoch_loss:.4f}")

print("="*150)

print("Final Eval.")
with torch.no_grad():
	for inputs, target in dataset:
		pred = model(inputs)

		print(f"Inputs: {inputs.tolist()} | Target: {target.item()} | Guess: {pred.item():.4f}")
		