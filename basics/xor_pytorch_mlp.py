import torch
import time

dataset = [
    (torch.tensor([0.0, 0.0]), torch.tensor(0.0)),
    (torch.tensor([0.0, 1.0]), torch.tensor(1.0)),
    (torch.tensor([1.0, 0.0]), torch.tensor(1.0)),
    (torch.tensor([1.0, 1.0]), torch.tensor(0.0))
]

w11 = torch.randn(1, requires_grad=True)
w12 = torch.randn(1, requires_grad=True)
w21 = torch.randn(1, requires_grad=True)
w22 = torch.randn(1, requires_grad=True)
w31 = torch.randn(1, requires_grad=True)
w32 = torch.randn(1, requires_grad=True)
b1 = torch.zeros(1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
b3 = torch.zeros(1, requires_grad=True)

l = 0.9

start_time = time.perf_counter()

for epochs in range(100000):

	epoch_loss = 0.0

	for inputs, target in dataset:
		x1, x2 = inputs[0], inputs[1]

		# forward pass
		z11 = x1 * w11 + x2 * w21 + b1
		z12 = x1 * w12 + x2 * w22 + b2
		o11 = torch.sigmoid(z11)
		o12 = torch.sigmoid(z12)
		z21 = o11 * w31 + o12 * w32 + b3
		o21 = torch.sigmoid(z21)
		
		# loss calculation
		loss = (o21 - target)**2
		epoch_loss += loss.item()

		# gradient calculation
		loss.backward()

		# weight update using pytorch
		with torch.no_grad():
			w11 -= l * w11.grad
			w12 -= l * w12.grad
			w21 -= l * w21.grad
			w22 -= l * w22.grad
			w31 -= l * w31.grad
			w32 -= l * w32.grad
			b1 -= l * b1.grad
			b2 -= l * b2.grad
			b3 -= l * b3.grad

			# clear gradients for next iteration
			w11.grad.zero_()
			w12.grad.zero_()
			w21.grad.zero_()
			w22.grad.zero_()
			w31.grad.zero_()
			w32.grad.zero_()
			b1.grad.zero_()
			b2.grad.zero_()
			b3.grad.zero_()

	if epochs % 20000 == 0:
		print(f"Epoch {epochs} || Total Loss: {epoch_loss:.4f}")

end_time = time.perf_counter()
total_time = end_time - start_time 
print(f"time_taken = {total_time}")
print("="*100)
print("Final Evaluation on XOR Truth Table:")
with torch.no_grad():
	for inputs, target in dataset:
		x1, x2 = inputs[0], inputs[1]
		z11 = x1 * w11 + x2 * w21 + b1
		z12 = x1 * w12 + x2 * w22 + b2
		o11 = torch.sigmoid(z11)
		o12 = torch.sigmoid(z12)
		z21 = o11 * w31 + o12 * w32 + b3
		o21 = torch.sigmoid(z21)
		print(f"Inputs: {inputs.tolist()} | Target: {target.item()} | Model Guess: {o21.item():.4f} (Rounded: {round(o21.item())})")