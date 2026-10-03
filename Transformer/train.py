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

def train_epoch(model, dataloader, optimizer, device):
    # Put the model in training mode (turns on Dropout)
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # 1. Unpack your batch (assuming a dictionary format) and move to GPU/CPU
        src = batch['src'].to(device)
        raw_tgt = batch['tgt'].to(device)
        
        # 2. Slice the target for Teacher Forcing
        tar_input = raw_tgt[:, :-1]
        tar_real = raw_tgt[:, 1:]
        
        # 3. Generate the masks (using the shifted tar_input!)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(src, tar_input)
        
        # 4. Clear old gradients from the last step
        optimizer.zero_grad()
        
        # 5. Forward Pass
        predictions = model(src, tar_input, enc_padding_mask, combined_mask, dec_padding_mask)
        
        # 6. Calculate Loss against the shifted true targets
        loss = calculate_loss(predictions, tar_real)
        
        # 7. Backward Pass (calculate gradients) and Optimizer Step (update weights)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Return the average loss for this epoch
    return total_loss / len(dataloader)