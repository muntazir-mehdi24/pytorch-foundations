import torch

def sep():
	print("="*50)

# create tensor
x = torch.tensor(2.0, requires_grad = True)

y = x ** 3

# triger chain rule
y.backward()

# check result
print(f"Value of x : {x.item()}")
print(f"gradient of x (dy/dx): {x.grad.item()}")