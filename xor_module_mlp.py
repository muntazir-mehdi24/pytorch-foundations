import torch
import torch.nn as nn

class XOR_MLP(nn.Module):

	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(2,2)
		self.fc1 = nn.Linear(2,1)

	def forward(self,x):
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc1(x))
		return x

model =  XOR_MLP()
print(model)
	