import os
import shutil

# sepertor
def sep():
	print("=" * 50)

# keep the record of how many files are deleted
deleted_count = 0

sep()

# iterate over all dirs and files in root folder
for root, dirs, files in os.walk("."):
	
	# checks if direct name of folder is there
	if "__pycache__" in dirs:

		# join the path to make full path 
		cache_path = os.path.join(root, "__pycache__")

		# delete every cache file and folder using shutil
		shutil.rmtree(cache_path)
		print(f"Deleted : {cache_path}")
		deleted_count += 1

print(f"Cleanup Completed, Total removed files : {deleted_count}")
sep()
