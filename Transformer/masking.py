# in this file we will define the masking functions that are used in the transformer model. These functions are used to create masks for the input and output sequences, which are used to prevent the model from attending to certain positions in the sequence. The masks are used in the attention mechanism to ensure that the model only attends to relevant positions in the sequence.
import torch    
import torch.nn as nn
from 

# class PaddingMask
def PaddingMask(seq):
    # seq is the input sequence, which is a tensor of shape (batch_size, seq_len)

    mask = (seq != 0) # create a mask where the padding tokens (0) are marked as False and the non-padding tokens are marked as True
    mask = mask.unsqueeze(1).unsqueeze(2) # add two dimensions to the mask tensor to make it compatible with the attention mechanism. The resulting shape is (batch_size, 1, 1, seq_len)

    return mask  # return the mask tensor, which can be used in the attention mechanism to prevent the model from attending to padding tokens.  
