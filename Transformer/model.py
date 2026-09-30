import torch
import torch.nn as nn
import math


# class input_embeddings
# Here we are initializing the input embeddings for the transformer model. The input embeddings are used to convert the input tokens into dense vectors of a specified dimension (d_model). The embedding layer is created using nn.Embedding, which takes the vocabulary size and the embedding dimension as parameters.
class input_embeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # according to the "Attention is All You Need" paper, we need to scale the embeddings by the square root of the model dimension (d_model) to prevent the dot products from growing too large in the attention mechanism. This scaling helps stabilize the training process.
        return self.embedding(x) * math.sqrt(self.d_model)


# class position_embeddings



# class layer_norm

# class feed_forward_block

# class multi_head_attention