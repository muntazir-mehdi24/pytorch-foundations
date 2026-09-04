# check_env.py
# Morning health check for my ML workspace

import sys
import torch

def sep():
	print("-" * 50)

def main():
	sep()
	print("ENVIROMENT CHECK")
	sep()

	# check python version
	print(f"Python Version : {sys.version}")

	# check if venv is active
	if "venv" in sys.prefix:
		print(f"Virtual enviroment : active ({sys.prefix})")
	else:
		print(f"Warning : Virtual Env not active!")

	# check pytorch
	print(f"Pytorch version : {torch.__version__}")

	# check Cuda
	if torch.cuda.is_available():
		print("Cuda : Available")
		gpu_name = torch.cuda.get_device_name(0)
		print(f"GPU : {gpu_name}")

		# Memory
		total = torch.cuda.get_device_properties(0).total_memory/1e9
		allocated = torch.cuda.memory_allocated(0)
		free = total - allocated

		print(f"GPU Total : {total:.1f} GB")
		print(f"GPU Allocated : {allocated:.1f} GB")
		print(f"GPU Free : {free:.1f} GB")

	else:
		print(f"CUDA Not Available!")

	sep()

if __name__ == "__main__":
	main()