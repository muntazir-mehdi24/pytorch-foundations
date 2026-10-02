# in this file class we define the architecture of the transformer model.
# we define the stacking of the encoder and decoder layers, as well as the input and output embeddings.
# these are essential wrapers that repeat the encoder and decoder layers N times to form the complete transformer model. 
import torch
import torch.nn as nn
from model import encoder_layer, decoder_layer, input_embeddings, position_embeddings

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
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout, max_len=5000):
        super().__init__()
        # Use the custom input_embeddings to ensure math.sqrt(d_model) scaling is applied
        self.encoder_embedding = input_embeddings(d_model, input_vocab_size)
        self.decoder_embedding = input_embeddings(d_model, target_vocab_size)
        
        # Instantiate the positional embeddings you built
        self.pos_embed = position_embeddings(d_model, dropout, max_len)
        
        # Define the encoder and decoder stacks
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, dropout)
        
        # Define the final linear layer
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # Apply embeddings AND positional encodings
        enc_input = self.pos_embed(self.encoder_embedding(inp))
        enc_output = self.encoder(enc_input, enc_padding_mask)
        
        # Apply embeddings AND positional encodings for the target sequence
        dec_input = self.pos_embed(self.decoder_embedding(tar))
        dec_output = self.decoder(dec_input, enc_output, dec_padding_mask, look_ahead_mask)
        
        # Map to vocabulary space
        final_output = self.final_layer(dec_output)
        return final_output