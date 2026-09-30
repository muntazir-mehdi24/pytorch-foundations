import torch
import torch.nn as nn

def sep():
	print("-"*50)

sep()
sep()

# create model
model = nn.Sequential(
	nn.Linear(in_features = 3, out_features = 7),
	nn.ReLU(),
	nn.Linear(in_features = 7, out_features = 2)
)

# create dummy input
x = torch.randn(3)

# input pass through model
output = model(x)

print(f"input matrix : \n{x}")
print(f"Output matrix : \n{output}")
print(f"Model structure : \{model}")

sep()
sep()