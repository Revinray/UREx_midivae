# from marcus_preprocessing import midi_data_loader
import torch
import torch.optim as optim
from torch.utils.data import DataLoader#, random_split
# from datasetcls import TokenizedMidiDataset
from vae import BetaVAE
# import os
from tqdm import tqdm

from miditok import REMI
from pathlib import Path
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.nn.utils.rnn import pad_sequence

import torch.nn as nn

def midi_data_loader(folder, shuffle=True):
    tokenizer = REMI()  # using defaults parameters (constants.py)
    midi_paths = [path.resolve() for path in Path(folder).rglob("*.mid")][:10]

    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=32, shuffle=shuffle)

    return data_loader

def collate_fn(batch):
    # Pad sequences to the same length
    batch = pad_sequence(batch, batch_first=True, padding_value=0)
    # Add a feature dimension
    batch = batch.unsqueeze(-1)
    return batch

# Define model parameters
input_dim = 128           # Each time step has 128 features (piano roll)
hidden_dim = 256          # GRU hidden dimension
latent_dim = 64           # Size of the latent space
beta = 4.0                # Adjust beta for stronger disentanglement

batch_size = 32

model = BetaVAE(input_dim, hidden_dim, latent_dim, beta=beta)

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10

train_data_loader = midi_data_loader("dataset_train", shuffle=True)
test_data_loader = midi_data_loader("dataset_test", shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)

# Create an embedding layer (vocab_size depends on your tokenizer)
vocab_size = len(train_data_loader.dataset.tokenizer.vocab)
embedding_layer = nn.Embedding(vocab_size, input_dim).to(device)

model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0.0
    
    for batch in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        optimizer.zero_grad()
        
        tokens = batch["input_ids"].to(device)
        embedded = embedding_layer(tokens.long())  # shape: (batch, seq_len, input_dim)
        
        # Forward pass through VAE
        recon_x, mu, logvar = model(embedded.float())
        
        # Compute loss
        loss, _, _ = model.loss_function(recon_x, embedded.float(), mu, logvar)
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_data_loader)
    print(f"Average training loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    total_test_loss = 0.
    for batch in test_data_loader:
        tokens = batch["input_ids"].to(device)
        embedded = embedding_layer(tokens.long())  # (batch, seq_len, input_dim)

        recon_x, mu, logvar = model(embedded.float())
        loss, _, _ = model.loss_function(recon_x, embedded.float(), mu, logvar)
        total_test_loss += loss.item()
        
    avg_test_loss = total_test_loss / len(test_data_loader)
    print(f"Average test loss: {avg_test_loss:.4f}")

# Generate a random latent vector z and decode it to a sequence of length seq_len
model.eval()
with torch.no_grad():
    seq_len = 256
    z = torch.randn((1, latent_dim)).to(device)
    decoded = model.decode(z, seq_len)  # shape: (1, seq_len, input_dim)

    # Convert probabilities to token IDs
    predicted_tokens = torch.argmax(decoded, dim=-1)  # shape: (1, seq_len)
    generated_tokens = predicted_tokens.squeeze().tolist()
    print(generated_tokens)

    tokenizer = REMI()
    midi_obj = tokenizer.tokens_to_midi(generated_tokens)

    # save the generated MIDI file
    midi_obj.write("generated.mid")
