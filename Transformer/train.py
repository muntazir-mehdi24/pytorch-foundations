# in this file we implement training and evaluation functions for the transformer model. We will define a function to train the model for one epoch, and a function to evaluate the model on a validation set. We will also define a function to calculate the loss and accuracy of the model on a given dataset. These functions will be used in the main training loop to train and evaluate the transformer model.

import torch
import torch.nn as nn
from architecture import Transformer
from masking import create_masks

# define the loss function, ignoring the padding tokens in the target sequence. The padding tokens are typically represented by 0 in the target sequence. The loss function is used to calculate the difference between the predicted output and the actual output, and is used to update the model's parameters during training.
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

def calculate_loss(predictions, targets):
    # flatten the 3d predictions to 2d (batch_size *  seq_len, vocab_size)
    predictions = predictions.view(-1, predictions.size(-1))

    # flatten the 2d targets to 1d (batch_size * seq_len)
    targets = targets.view(-1)

    # claculate and return the loss
    return loss_fn(predictions, targets)