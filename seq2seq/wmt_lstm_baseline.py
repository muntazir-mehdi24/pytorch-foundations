import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.datasets import Multi30k
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
from neuroplot import LiveVisualizer

# ==========================================
# 1. Tokenizers and Vocabularies
# ==========================================
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenizer_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

def tokenizer_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

def yield_tokens(data_iter, language_index, tokenizer):
    for data_sample in data_iter:
        yield tokenizer(data_sample[language_index])

def build_vocab(language_index, tokenizer):
    train_iter = Multi30k(split='train', language_pair=('de', 'en'))
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, language_index, tokenizer), 
        specials=["<unk>", "<pad>", "<bos>", "<eos>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

source_vocab = build_vocab(0, tokenizer_de)
target_vocab = build_vocab(1, tokenizer_en)

# ==========================================
# 2. Data Pipeline (Collate Function)
# ==========================================
def collate_fn(batch):
    src_list = []
    trg_list = []
    for src_sample, trg_sample in batch:
        src_tensor = torch.tensor([source_vocab["<bos>"]] + [source_vocab[token] for token in tokenizer_de(src_sample)] + [source_vocab["<eos>"]], dtype=torch.long)
        trg_tensor = torch.tensor([target_vocab["<bos>"]] + [target_vocab[token] for token in tokenizer_en(trg_sample)] + [target_vocab["<eos>"]], dtype=torch.long)
        src_list.append(src_tensor)
        trg_list.append(trg_tensor)
        
    src_batch = pad_sequence(src_list, padding_value=source_vocab["<pad>"])
    trg_batch = pad_sequence(trg_list, padding_value=target_vocab["<pad>"])
    return src_batch, trg_batch

# ==========================================
# 3. Model Architecture
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, d):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(d)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=d)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, d, attention):
        super(Decoder, self).__init__()
        self.attention = attention
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(d)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=d)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        embedded = embedded.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        targ_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(targ_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)

        input = trg[0, :]
        for t in range(1, targ_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs

class attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs.permute(1, 0, 2)), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# ==========================================
# 4. Training Function
# ==========================================
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / max(num_batches, 1)  # Prevent division by zero
# ==========================================
# 5. Inference & Evaluation Utilities
# ==========================================
def translate_sentence(model, sentence, device, max_length=50):
    model.eval()
    
    tokens = tokenizer_de(sentence)
    tokens = ["<bos>"] + tokens + ["<eos>"]
    src_indexes = [source_vocab[token] for token in tokens]
    
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor)
        
    trg_indexes = [target_vocab["<bos>"]]
    
    for _ in range(max_length):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
            
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == target_vocab["<eos>"]:
            break
            
    trg_tokens = target_vocab.lookup_tokens(trg_indexes)
    return trg_tokens[1:] # Exclude <bos> for the printout format

def calculate_bleu(data_iterator, model, device, max_length=50):
    trgs = []
    pred_trgs = []
    
    for i, (src_sample, trg_sample) in enumerate(data_iterator):
        if i >= 500: # Evaluate subset to save time
            break
        pred_trg = translate_sentence(model, src_sample, device, max_length)
        # Slicing [:-1] removes <eos> token for BLEU comparison
        pred_trgs.append(pred_trg[:-1])
        trgs.append([tokenizer_en(trg_sample)])
        
    return bleu_score(pred_trgs, trgs) * 100

# ==========================================
# 6. Hyperparameters & Initialization
# ==========================================
batch_size = 64
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim_encoder = len(source_vocab)
output_dim = len(target_vocab)
n_layers = 3
enc_emb_dim = 256
dec_emb_dim = 256
hid_dim = 512
enc_dropout = 0.5
dec_dropout = 0.5

train_iter = Multi30k(split='train', language_pair=('de', 'en'))
train_loader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

enc = Encoder(input_dim_encoder, enc_emb_dim, hid_dim, n_layers, enc_dropout)
attn = attention(hid_dim, hid_dim)
dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout, attn)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = target_vocab["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# ==========================================
# 7. Final Training Loop & Telemetry 
# ==========================================
num_epochs = 20
test_sentence = "Ich werde der Armee beitreten."

viz = LiveVisualizer(plots=["loss"], model=model, update_every=1)

print("Starting training...")
for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion, clip=1)
    
    viz.step(epoch=epoch+1, loss=loss)
    
    # Run live inference checkpoint
    translated_tokens = translate_sentence(model, test_sentence, device)
    
    print(f"Epoch [ {epoch+1} / {num_epochs} ]")
    print("=> Saving checkpoint")
    print(f"Translated example sentence\n {translated_tokens}")

viz.close()
plt.show()

# Calculate Final Baseline Performance
print("\nEvaluating BLEU Score...")
test_iter = Multi30k(split='valid', language_pair=('de', 'en'))
final_bleu = calculate_bleu(test_iter, model, device)
print(f"Final BLEU Score on subset: {final_bleu:.2f}")