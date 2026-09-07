w1 = 0.0
w2 = 0.0
b =  0.0
l = 0.1

x1 = [0.0, 0.0, 1.0, 1.0]
x2 = [0.0, 1.0, 0.0, 1.0]
y = [0.0, 1.0, 1.0, 1.0]

for eopochs in range(10):

	for i in range(4):
		raw = x1[i]*w1 + x2[i]*w2 + b 
	
		if raw > 0:
			guess = 1
		else :
			guess = 0
		error = y[i] - guess

		w1 = w1 + (l*error*x1[i])
		w2 = w2 + (l*error*x2[i])
		b = b + (l*error)
		print(f"Inputs: {x1[i]}, {x2[i]} || Target: {y[i]} || Guess: {guess} || Error: {error}")
	