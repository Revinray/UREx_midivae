import torch.optim as optim
from torch.utils.data import DataLoader
from datasetcls import MidiDataset
from vae import BetaVAE

# Suppose we have a list of MIDI file paths in midi_files
# midi_files = ['file1.mid', 'file2.mid', ...]
dataset = MidiDataset(midi_files, fs=100, segment_duration=5)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model parameters
input_dim = 128           # Each time step has 128 features (piano roll)
hidden_dim = 256          # GRU hidden dimension
latent_dim = 64           # Size of the latent space
beta = 4.0                # Adjust beta for stronger disentanglement

model = BetaVAE(input_dim, hidden_dim, latent_dim, beta=beta)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        # batch: (batch_size, seq_len, input_dim)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss, recon_loss, kld_loss = model.loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")
