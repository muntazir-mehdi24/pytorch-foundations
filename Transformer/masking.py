# in this file we will define the masking functions that are used in the transformer model. These functions are used to create masks for the input and output sequences, which are used to prevent the model from attending to certain positions in the sequence. The masks are used in the attention mechanism to ensure that the model only attends to relevant positions in the sequence.
import torch    
import torch.nn as nn
from 

# PaddingMask function is used to create a mask for the input sequence, which is used to prevent the model from attending to padding tokens. The padding tokens are typically represented by 0 in the input sequence. The mask is created by checking if each token in the input sequence is not equal to 0, and then unsqueezing the mask tensor to make it compatible with the attention mechanism. The resulting mask tensor has a shape of (batch_size, 1, 1, seq_len), where seq_len is the length of the input sequence. This mask can be used in the attention mechanism to ensure that the model only attends to non-padding tokens in the input sequence.
def PaddingMask(seq):
    # seq is the input sequence, which is a tensor of shape (batch_size, seq_len)

    mask = (seq != 0) # create a mask where the padding tokens (0) are marked as False and the non-padding tokens are marked as True
    mask = mask.unsqueeze(1).unsqueeze(2) # add two dimensions to the mask tensor to make it compatible with the attention mechanism. The resulting shape is (batch_size, 1, 1, seq_len)

    return mask  # return the mask tensor, which can be used in the attention mechanism to prevent the model from attending to padding tokens.  


# look_ahead_mask function is used to create a mask for the output sequence, which is used to prevent the model from attending to future tokens in the sequence. The mask is created by generating a lower triangular matrix of ones, which is then unsqueezed to make it compatible with the attention mechanism. The resulting mask tensor has a shape of (batch_size, 1, seq_len, seq_len), where seq_len is the length of the output sequence. This mask can be used in the attention mechanism to ensure that the model only attends to previous tokens in the output sequence.
def look_ahead_mask(size):
    # size is the length of the output sequence, which is an integer

    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(1) # create a lower triangular matrix of ones, which is then unsqueezed to make it compatible with the attention mechanism. The resulting shape is (1, 1, seq_len, seq_len)

    return mask  # return the mask tensor, which can be used in the attention mechanism to prevent the model from attending to future tokens in the output sequence.

# create_masks function is used to create the padding and look-ahead masks for the input and output sequences, respectively. The function takes in the input and output sequences as arguments, and returns the padding mask for the input sequence, the look-ahead mask for the output sequence, and the padding mask for the output sequence. The padding mask for the output sequence is created by calling the PaddingMask function on the output sequence. The look-ahead mask for the output sequence is created by calling the look_ahead_mask function on the length of the output sequence. The resulting masks can be used in the attention mechanism to ensure that the model only attends to relevant positions in the input and output sequences.
def create_masks(inp, tar):
    # inp is the input sequence, which is a tensor of shape (batch_size, seq_len)
    # tar is the output sequence, which is a tensor of shape (batch_size, seq_len)

    enc_padding_mask = PaddingMask(inp)  # create the padding mask for the input sequence
    dec_padding_mask = PaddingMask(tar)  # create the padding mask for the output sequence
    look_ahead_mask = look_ahead_mask(tar.size(1))  # create the look-ahead mask for the output sequence

    return enc_padding_mask, look_ahead_mask, dec_padding_mask  # return the masks, which can be used in the attention mechanism to ensure that the model only attends to relevant positions in the input and output sequences.

