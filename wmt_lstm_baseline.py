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
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, d):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(d)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=d)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
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
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs

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
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
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
dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = target_vocab["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# ==========================================
# 7. Final Training Loop & Telemetry 
# ==========================================
num_epochs = 20
test_sentence = "Ich werde der Armee beitreten."

# Initialize interactive matplotlib window
plt.ion()
fig, ax = plt.subplots()
loss_history = []

print("Starting training...")
for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion, clip=1)
    
    # Update matplotlib graph dynamically
    loss_history.append(loss)
    ax.clear()
    ax.plot(range(1, epoch + 2), loss_history, marker='o', color='blue', label="Train Loss")
    ax.set_title("Seq2Seq LSTM Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.pause(0.1)  # Pauses execution just long enough to draw the frame
    
    # Run live inference checkpoint
    translated_tokens = translate_sentence(model, test_sentence, device)
    
    print(f"Epoch [ {epoch+1} / {num_epochs} ]")
    print("=> Saving checkpoint")
    print(f"Translated example sentence\n {translated_tokens}")

# Keep the plot window open at the end
plt.ioff()
plt.show(block=False)

# Calculate Final Baseline Performance
print("\nEvaluating BLEU Score...")
test_iter = Multi30k(split='valid', language_pair=('de', 'en'))
final_bleu = calculate_bleu(test_iter, model, device)
print(f"Final BLEU Score on subset: {final_bleu:.2f}")