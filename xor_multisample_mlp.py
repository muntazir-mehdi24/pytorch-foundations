import math
import random

# seperator
def sep():
	print("="*150)

# sigmoid 
def sigmoid(x):
	return 1/(1 + math.exp(-x))

# derivative of sigmoid
def d_sigma(output):
	return output*(1 - output)

dataset = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0)
]

w11 = random.uniform(-1.0, 1.0)
w12 = random.uniform(-1.0, 1.0)
w21 = random.uniform(-1.0, 1.0)
w22 = random.uniform(-1.0, 1.0)
w31 = random.uniform(-1.0, 1.0)
w32 = random.uniform(-1.0, 1.0)
b1 = random.uniform(-1.0, 1.0)
b2 = random.uniform(-1.0, 1.0)
b3 = random.uniform(-1.0, 1.0)
l = 0.9

for epochs in range(100000):

	epoch_loss = 0.0

	for inputs, targets in dataset:

		x1 = inputs[0]
		x2 = inputs[1]
		y = targets

		z11 = x1*w11 + x2*w21 + b1
		z12 = x1*w12 + x2*w22 + b2
		o11 = sigmoid(z11)
		o12 = sigmoid(z12)
		z21 = o11 * w31 + o12 * w32 + b3
		o21 = sigmoid(z21)

		loss = (o21 - y)**2
		epoch_loss += loss

		grad_w11 = 2 * (o21 - y) * d_sigma(o21)* w31 *d_sigma(o11) * x1
		grad_w12 = 2 * (o21 - y) * d_sigma(o21)* w32 *d_sigma(o12) * x1
		grad_w21 = 2 * (o21 - y) * d_sigma(o21)* w31 *d_sigma(o11) * x2
		grad_w22 = 2 * (o21 - y) * d_sigma(o21)* w32 *d_sigma(o12) * x2
		grad_w31 = 2 * (o21 - y) * d_sigma(o21)* o11
		grad_w32 = 2 * (o21 - y) * d_sigma(o21)* o12
		grad_b3 = 2 * (o21 - y) * d_sigma(o21)
		grad_b2 = 2 * (o21 - y) * d_sigma(o21) * w32 * d_sigma(o12)
		grad_b1 = 2 * (o21 - y) * d_sigma(o21) * w31 * d_sigma(o11)

		w11 = w11 - l * grad_w11
		w12 = w12 - l * grad_w12
		w21 = w21 - l * grad_w21
		w22 = w22 - l * grad_w22
		w31 = w31 - l * grad_w31
		w32 = w32 - l * grad_w32
		b1 = b1 - l * grad_b1
		b2 = b2 - l * grad_b2
		b3 = b3 - l * grad_b3

	if epochs % 5000 == 0:
		print(f"epoch : {epochs} || total loss : {epoch_loss:.4f} || w11 : {w11:.1f} || w12 : {w12:.1f} || w21 : {w21:.1f} || w22 : {w22:.1f} || w31 : {w31:.1f} || w32 : {w32:.1f} || b1 : {b1:.1f} || b2 : {b2:.1f} || b3 : {b3:.1f}")

sep()
print("Final Evaluation on XOR Truth Table:")
for inputs, targets in dataset:
    x1 = inputs[0]
    x2 = inputs[1]
    
    z11 = x1*w11 + x2*w21 + b1
    z12 = x1*w12 + x2*w22 + b2
    o11 = sigmoid(z11)
    o12 = sigmoid(z12)
    z21 = o11 * w31 + o12 * w32 + b3
    o21 = sigmoid(z21)
    
    print(f"Inputs: {inputs} | Target: {targets} | Model Guess: {o21:.4f} (Rounded: {round(o21)})")
sep()