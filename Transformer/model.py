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

# class multi_head_attention

class multi_head_attention(nn.Module):
    def __init__(self, d_model, num_heads, dropouts):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropouts)
        self.head_dim = d_model // num_heads 
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)  
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # now here we initiate the linear projections for the query, key, and value tensors. The linear layers (self.w_q, self.w_k, self.w_v) are applied to the input tensors (query, key, value) to transform them into new representations with the same dimensionality (d_model). This allows the model to learn different representations for the query, key, and value inputs, which are essential for the attention mechanism.
        q = self.w_q(query)  # (batch_size, seq_len, d_model)
        k = self.w_k(key)    # (batch_size, seq_len, d_model
        v = self.w_v(value)  # (batch_size, seq_len, d_model)

        # spitting the query, key, and value tensors into multiple heads. The view operation reshapes the tensors to have a shape of (batch_size, num_heads, seq_len, head_dim), where head_dim is the dimension of each attention head (d_model / num_heads). This allows the model to compute attention in parallel across multiple heads, enabling it to capture different aspects of the input sequence.
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # calculating the attention scores using the scaled dot-product attention mechanism. The attention scores are computed by taking the dot product of the query and key tensors, scaling them by the square root of the head dimension (self.head_dim), and applying a softmax function to obtain a probability distribution over the keys. This allows the model to focus on different parts of the input sequence based on the query.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Apply the mask to the attention scores, setting the masked positions to negative infinity. This ensures that the model does not attend to the masked positions during the attention computation.

        # applying the softmax function to the attention scores to obtain the attention weights. The softmax function normalizes the scores along the last dimension (seq_len), converting them into a probability distribution that sums to 1. This allows the model to weigh the importance of each key when computing the output representation.
        attn_weights = torch.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        # apply dropout to the attention weights to prevent overfitting during training. The dropout layer randomly sets a fraction of the attention weights to zero, encouraging the model to learn more robust representations that do not rely on specific attention patterns.
        attn_weights = self.dropout(attn_weights)
        # multiplying the attention weights with the value tensor to obtain the weighted sum of the values. This operation computes the output representation for each query by aggregating information from the values based on the attention weights, allowing the model to focus on relevant parts of the input sequence.
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # concatenating the outputs from all attention heads and applying a linear transformation to obtain the final output representation. The transpose and contiguous operations rearrange the tensor dimensions, and the view operation reshapes the tensor to have a shape of (batch_size, seq_len, d_model). The linear layer (self.w_o) is then applied to transform the concatenated outputs into the desired output dimension (d_model).
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.w_o(attn_output)  # (batch_size, seq_len, d_model)
        return output, attn_weights  # returning the final output representation and the attention weights for further analysis or visualization.




# class layer_norm

class layer_norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))  # learnable scaling parameter
        self.bias = nn.Parameter(torch.zeros(d_model))  # learnable bias parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # compute the mean of the input tensor along the last dimension
        std = x.std(dim=-1, keepdim=True)    # compute the standard deviation of the input tensor along the last dimension
        normalized_x = (x - mean) / (std + self.eps)  # normalize the input tensor using the computed mean and standard deviation
        return self.alpha * normalized_x + self.bias  # apply the learnable scaling and bias parameters to the normalized tensor


# class feed_forward_block

class feed_forward_block(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # first linear layer to project the input to a higher-dimensional space
        self.dropout = nn.Dropout(dropout)  # dropout layer to prevent overfitting
        self.linear2 = nn.Linear(d_ff, d_model)  # second linear layer to project back to the original dimension

    def forward(self, x):
        x = self.linear1(x)  # apply the first linear transformation
        x = torch.relu(x)  # apply the ReLU activation function
        x = self.dropout(x)  # apply dropout
        x = self.linear2(x)  # apply the second linear transformation
        return x  # return the output of the feed-forward block

