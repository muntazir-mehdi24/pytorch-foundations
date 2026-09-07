# initial setting

def sep():
	print("="* 50)

sep()

x1 = [2.0, 1.0, 4.0]
x2 = [3.0, 1.0, 2.0]

y = [13.0, 5.0, 14.0]

l = 0.005

w1 = 0.0
w2 = 0.0
b = 0.0



for epochs in range(1000):

	total_grad_w1 = 0.0
	total_grad_w2 = 0.0
	total_grad_b = 0.0
	total_loss = 0.0

	for i in range(3):
		guess = (x1[i] * w1 + x2[i] * w2) + b
		loss = (guess - y[i])**2 
		grad_w1 = 2 * x1[i] * (guess - y[i])
		grad_w2 = 2 * x2[i] * (guess - y[i])
		grad_b = 2 * (guess - y[i])

		total_grad_w1 += grad_w1
		total_grad_w2 += grad_w2
		total_grad_b += grad_b
		total_loss += loss

	avg_grad_w1 = total_grad_w1 / 3
	avg_grad_w2 = total_grad_w2 / 3
	avg_grad_b = total_grad_b / 3
	avg_loss = total_loss / 3

	# take step
	w1 = w1 - (l * avg_grad_w1)
	w2 = w2 - (l * avg_grad_w2)
	b = b - (l * avg_grad_b)
	
	print(f"eopoch = {epochs} || weight_1 = {w1:.4f} || weight_2 = {w2:.4f} || bias = {b:.4f} || loss = {avg_loss:.4f}")
sep()
