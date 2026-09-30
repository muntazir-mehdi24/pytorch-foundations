# import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

# load the dataset
dataset = datasets.make_moons(n_samples=100, noise=0.1)
X, y = dataset
x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# make the blueprint of the model
class Moon_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Loss function and optimizer
model = Moon_MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# visualization setup
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
loss_history = []

# Create a fixed grid for the decision boundary background once
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100), torch.linspace(y_min, y_max, 100), indexing='ij')
grid_tensor = torch.stack([xx.ravel(), yy.ravel()], dim=1)


# training loop for 2000 epochs
for epoch in range(2000):
    # forward pass
    y_pred = model(x)

    # compute the loss
    loss = criterion(y_pred, y)

    # zero the gradients, backward pass, update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # track loss
    loss_history.append(loss.item())

    # print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} || Total Loss: {loss.item():.4f}")

    # update live plots every 50 epochs to keep it smooth
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            preds = model(grid_tensor).reshape(xx.shape)
        model.train()

        # 1. Plot Decision Boundary
        ax1.clear()
        ax1.contourf(xx.numpy(), yy.numpy(), preds.numpy(), levels=50, cmap=plt.cm.Spectral, alpha=0.8)
        ax1.scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=y.squeeze().numpy(), cmap=plt.cm.Spectral, edgecolors='k')
        ax1.set_title(f"Epoch {epoch} - Decision Boundary")

        # 2. Plot Loss Curve
        ax2.clear()
        ax2.plot(loss_history, color='red', lw=2)
        ax2.set_title("Loss Curve")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")

        plt.draw()
        plt.pause(0.001)

# Keep the plot open at the end
plt.ioff()
plt.show()