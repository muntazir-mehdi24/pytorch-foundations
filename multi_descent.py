# initial setting 

def f_str(a1, a2):
	print(f"{a1} : {a2}")

# input
x1 = 2.0
x2 = 3.0

# target
y = 13.0

# initial wight
w1 = 0.0
w2 = 0.0

# learning rate
l = 0.01

# bias
b = 0.0

# training loop
for epochs in range(50):
	# forward pass
	guess = x1*w1 + x2*w2 + b
	
	# loss fn
	loss = (guess - y) ** 2

	# calculating  gradient
	grad_w1 = 2 * x1 * (guess - y)
	grad_w2 = 2 * x2 * (guess - y)
	grad_b = 2 * (guess - y)

	# update weight
	w1 = w1 - (l*grad_w1)
	w2 = w2 - (l*grad_w2)
	b = b - (l*grad_b)

	# training matadata
	print(f"epochs : {epochs} || weight_1 : {w1:.4f} || weight_2 : {w2:.4f} || loss : {loss:.4f} || bias : {b:.4f}")

