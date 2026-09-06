# initial setting 

def f_str(a1, a2):
	print(f"{a1} : {a2}")

# input
x = 3.0

# target
y = 6.0

# initial wight
w = 0.0

# learning rate
l = 0.01

# training loop
for epochs in range(50):
	# forward pass
	guess = x*w
	
	# loss fn
	loss = (guess - y) ** 2

	# calculating  gradient
	gradient = 2*x*(x*w -y)

	# update weight
	w = w - (l*gradient)	

	# training matadata
	print(f"epochs : {epochs} || weight : {w:.4f} || loss : {loss:.4f}")

