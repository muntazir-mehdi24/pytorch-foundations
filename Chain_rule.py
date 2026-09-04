import torch

def sep():
	print("="*50)

# create tensor
x = torch.tensor(8.0, requires_grad = True)

y = x ** 3

z = 1/y

l = z**2

# triger chain rule
l.backward()

# check result
print(f"Value of x : {x.item()}")
print(f"gradient of x (dl/dz * dz/dy * dy/dx): {x.grad.item()}")