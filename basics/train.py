import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.datasets as datasets
from neuroplot import LiveVisualizer  # <--- Updated import

# 1. Load dataset
dataset = datasets.make_moons(n_samples=100, noise=0.1)
X, y = dataset
x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 2. Define model
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

model = Moon_MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize NeuroPlot dashboard
viz = LiveVisualizer(
    plots=["boundary", "latent", "loss", "accuracy"], 
    model=model, 
    data=(x, y), 
    update_every=50,
    save_gif=True,
    gif_name="neuroplot_demo.gif",
    save_best_model=True,
    checkpoint_path="neuroplot_best_model.pth"
)

# 3. Training loop
for epoch in range(3000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    preds = (y_pred >= 0.5).float()
    acc = (preds == y).float().mean().item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    viz.step(epoch, loss=loss.item(), accuracy=acc)

viz.close()