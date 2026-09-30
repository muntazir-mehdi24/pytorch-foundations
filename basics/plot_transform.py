import torch
import matplotlib.pyplot as plt

# 1. Your exact PyTorch math
i_hat = torch.tensor([1.0, 0.0])
j_hat = torch.tensor([0.0, 1.0])
T = torch.tensor([[2.0, 3.0], [3.0, 0.0]])

new_i = T @ i_hat
new_j = T @ j_hat

# 2. Setup the canvas
plt.figure(figsize=(6, 6))
plt.axhline(0, color='grey', linewidth=1)
plt.axvline(0, color='grey', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.6)

# 3. Plot Original Vectors (Faded)
plt.quiver(0, 0, i_hat[0], i_hat[1], angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.3, label='Original i-hat')
plt.quiver(0, 0, j_hat[0], j_hat[1], angles='xy', scale_units='xy', scale=1, color='red', alpha=0.3, label='Original j-hat')

# 4. Plot Transformed Vectors (Solid)
plt.quiver(0, 0, new_i[0], new_i[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Transformed i-hat')
plt.quiver(0, 0, new_j[0], new_j[1], angles='xy', scale_units='xy', scale=1, color='red', label='Transformed j-hat')

# 5. Format and show
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.legend()
plt.title("How Matrix T warped your 2D space")
plt.show()