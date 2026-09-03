# test_gpu.py
# my first pytorch GPU test script

import torch

print("=" * 50)
print("pytorch version: ", torch.__version__)

# check cuda if available
cuda_available = torch.cuda.is_available()
print("CUDA available: ", cuda_available)

if cuda_available:
	# get name
	gpu_name = torch.cuda.get_device_name(0)
	print("gpu_name: ", gpu_name)

	# get gpu memory
	total_mem = torch.cuda.get_device_properties(0).total_memory
	print(f"GPU Memory : {total_mem/1e9:.1f} GB")
	
	# create two small tensors
	a = torch.tensor([1, 2, 3], device = "cuda")
	b = torch.tensor([1, 2, 3], device = "cuda")

	# do a compute
	c = a + b
	print("Tensor a : ", a)
	print("Tensor b : ", b)
	print("a + b = ", c)
	print("GPU Compute test PASSED")
else:
	print("warning: cuda not available")
print("=" * 50)