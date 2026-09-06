import time 

print("-"*50)
print("-"*50)

# Start time
start = time.perf_counter()

# Heavy compute
total = 0
for i in range (10000):
	total += i

# End time
end = time.perf_counter()

# elapsed time
elapsed = end - start

print(f"Computation : {total}")
print(f"Execution Time : {elapsed}")

print("-"*50)
print("-"*50)