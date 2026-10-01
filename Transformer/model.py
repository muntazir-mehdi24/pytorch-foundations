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
# Here we are initializing the position embeddings for the transformer model. The position embeddings are used to encode the positional information of the input tokens, allowing the model to capture the order of the tokens in the sequence. The position embeddings are created using a sinusoidal function, which generates a unique embedding for each position in the sequence. The dropout layer is applied to the position embeddings to prevent overfitting during training.
class position_embeddings(nn.Module):

    def __init__(self, d_model, dropout, max_len=50000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Create the position embeddings using a sinusoidal function
        pe = torch.zeros(max_len, d_model) # this is the master sheet filled with zeros, which will be filled with the position embeddings for each position in the sequence. The shape of the tensor is (max_len, d_model), where max_len is the maximum length of the input sequence and d_model is the dimension of the embeddings.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # this creates a tensor of shape (max_len, 1) containing the position indices from 0 to max_len-1. The unsqueeze(1) operation adds an extra dimension to the tensor, making it compatible for broadcasting with the div_term tensor in the next step.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # this creates a tensor of shape (d_model/2,) containing the scaling factors for the sinusoidal function. The scaling factors are calculated using the formula exp(-log(10000) * (2i/d_model)), where i is the index of the embedding dimension. The scaling factors are used to control the frequency of the sine and cosine functions, allowing the model to capture different positional patterns in the input sequence.

        pe[:, 0::2] = torch.sin(position * div_term) # this fills the even indices of the position embeddings tensor (pe) with the sine values of the product of the position indices and the scaling factors. The sine function is applied to the even dimensions of the embeddings, allowing the model to capture periodic patterns in the input sequence.
        pe[:, 1::2] = torch.cos(position * div_term) # this fills the odd indices of the position embeddings tensor (pe) with the cosine values of the product of the position indices and the scaling factors. The cosine function is applied to the odd dimensions of the embeddings, allowing the model to capture periodic patterns in the input sequence.
        pe = pe.unsqueeze(0)  # Add a batch dimension

        self.register_buffer('pe', pe) # this registers the position embeddings tensor (pe) as a buffer in the module. Buffers are tensors that are not considered model parameters and are not updated during training. By registering the position embeddings as a buffer, we ensure that they are saved and loaded along with the model's state_dict, allowing for consistent positional encoding across different training and inference sessions.

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] # this adds the positional embeddings to the input embeddings. the positional embeddings are sliced to match the length of the input sequence (x.size(1)) and then added to the input embeddings (x). This allows the model to incorporate positional information into the input representations, enabling it to capture the order of the tokens in the sequence.
        # we are not using .to(x.device) here because the position embeddings are already registered as a buffer in the module, which means they will automatically be moved to the same device as the input tensor (x) during training and inference. This ensures that the positional embeddings are always on the correct device without needing to explicitly call .to(x.device).
        return self.dropout(x) # this applies dropout to the combined input and position embeddings, randomly setting a fraction of the elements to zero during training. Dropout helps prevent overfitting by encouraging the model to learn more robust representations that do not rely on specific features in the input data.

# class layer_norm

# class feed_forward_block

# class multi_head_attention