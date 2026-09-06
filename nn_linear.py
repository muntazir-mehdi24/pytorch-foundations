import torch
import torch.nn as nn

# Create the layer
layer = nn.Linear(in_features = 3, out_features = 2)

# inspect parameter
print(f"Automatically generated weight matrix : {layer.weight}")
print(f"Weight shape : {layer.weight.shape}")

print(f"Automatically generated bias vector : {layer.bias}")
print(f"Bias shape : {layer.bias.shape}")

# Dummy input 
x = torch.tensor([4.0, 7.3, 3.3])

# pass through layer
output = layer(x)

print(f"Input shape : {x.shape}")
print(f"Final Output : {output}")
print(f"Output shape : {output.shape}")
