import torch

def sep():
	print("="*50)

sep()

# create 1d tensor
token_vec = torch.tensor([1, 2, 3])

# check physical address
print(f"Starting Device : {token_vec.device}")

# move to gpu
gpu_vec = token_vec.to("cuda")
print(f"New Device : {gpu_vec.device}")

# check shape
print(f"gpu_vec shape : {gpu_vec.shape}")
sep()