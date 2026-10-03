# in this file we will define the masking functions that are used in the transformer model. These functions are used to create masks for the input and output sequences, which are used to prevent the model from attending to certain positions in the sequence. The masks are used in the attention mechanism to ensure that the model only attends to relevant positions in the sequence.
import torch    
import torch.nn as nn

# class PaddingMask
class PaddingMask(seq):
    pass