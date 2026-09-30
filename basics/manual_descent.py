# initial setting 

def f_str(a1, a2):
	print(f"{a1} : {a2}")

# input
x = float(input("Enter a Number : "))

# target
y = 5.0

# initial wight
w = 0.0

# learning rate
l = 0.01

# bias
b = 0.0

# training loop
for epochs in range(50):
	# forward pass
	guess = x*w + b
	
	# loss fn
	loss = (guess - y) ** 2

	# calculating  gradient
	grad_w = 2 * x * (guess - y)
	grad_b = 2 * (guess - y)

	# update weight
	w = w - (l*grad_w)
	b = b - (l*grad_b)

	# training matadata
	print(f"epochs : {epochs} || weight : {w:.4f} || loss : {loss:.4f} || bias : {b:.4f}")

