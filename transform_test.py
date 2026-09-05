import torch

print("-"*50)
print("-"*50)

# write the original vectors
i_hat = torch.tensor([1.0, 0.0])
j_hat = torch.tensor([0.0, 1.0])

# create trannsformation
T = torch.tensor([[2.0, 1.0],[1.0,2.0]])

# apply transformation using multiplication
new_i = T @ i_hat
new_j = T @ j_hat

print(f"Original i_hat = {i_hat}")
print(f"Original j_hat = {j_hat}")
print(f"Original new_i = {new_i}")
print(f"Original new_j = {new_j}")

print("-"*50)
print("-"*50)