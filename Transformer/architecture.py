# in this file class we define the architecture of the transformer model.
# we define the stacking of the encoder and decoder layers, as well as the input and output embeddings.
# these are essential wrapers that repeat the encoder and decoder layers N times to form the complete transformer model. 
import torch
import torch.nn as nn
from model import encoder_layer, decoder_layer

# class Encoder
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout):
        super().__init__()
        # we first pass 4 parameters down into the layer , and use num_layers to repeat the encoder layer N times.
        self.layers = nn.ModuleList([encoder_layer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        # add final layer normalization to the output of the encoder stack.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # we pass the input through each encoder layer in the stack, applying the mask at each layer.
        for layer in self.layers:
            x = layer(x, mask)
        # we apply final layer normalization to the output of the encoder stack.
        return self.norm(x)


# class Decoder
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout):
        super().__init__()
        # we first pass 4 parameters down into the layer , and use num_layers to repeat the decoder layer N times.
        self.layers = nn.ModuleList([decoder_layer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        # add final layer normalization to the output of the decoder stack.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, padding_mask, look_ahead_mask):
        # we pass the input through each decoder layer in the stack, applying the look ahead mask and padding mask at each layer.
        for layer in self.layers:
            x = layer(x, enc_output, padding_mask, look_ahead_mask)
        # we apply final layer normalization to the output of the decoder stack.
        return self.norm(x)

# the transformer class is the main class that combines the encoder and decoder stacks, as well as the input and output embeddings.
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout):
        super().__init__()
        # we first define the input and output embeddings.
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        # we then define the encoder and decoder stacks.
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, dropout)
        # we then define the final linear layer that maps the decoder output to the target vocabulary size.
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # we first pass the input through the encoder embedding and add positional encoding.
        enc_output = self.encoder(self.encoder_embedding(inp), enc_padding_mask)
        # we then pass the target through the decoder embedding and add positional encoding.
        dec_output = self.decoder(self.decoder_embedding(tar), enc_output, dec_padding_mask, look_ahead_mask)
        # we then pass the decoder output through the final linear layer to get the logits for each token in the target vocabulary.
        final_output = self.final_layer(dec_output)
        return final_output