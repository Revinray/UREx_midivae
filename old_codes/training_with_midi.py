import torch.optim as optim
from torch.utils.data import DataLoader
from old_codes.datasetcls import MidiDataset
from vae import BetaVAE
import os
import wandb
from tqdm import tqdm

# Set log_to_wandb to True or False
log_to_wandb = False  # Set to True to enable wandb logging, False to disable

# Initialize wandb if enabled
if log_to_wandb:
    wandb.init(project="midi-vae-training", entity="your_wandb_username")

# Read all MIDI files in the folder "./POP909_MIDIs"
midi_folder = "./POP909_MIDIs"
midi_files = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith('.mid')]

# limit the number of MIDI files for testing
midi_files = midi_files[:100]

dataset = MidiDataset(midi_files, fs=100, segment_duration=5)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model parameters
input_dim = 128           # Each time step has 128 features (piano roll)
hidden_dim = 256          # GRU hidden dimension
latent_dim = 64           # Size of the latent space
beta = 4.0                # Adjust beta for stronger disentanglement

model = BetaVAE(input_dim, hidden_dim, latent_dim, beta=beta)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Log model parameters to wandb if enabled
if log_to_wandb:
    wandb.config.update({
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "beta": beta,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "num_epochs": 1
    })

num_epochs = 1
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        # batch: (batch_size, seq_len, input_dim)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss, recon_loss, kld_loss = model.loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Log metrics to wandb if enabled
        if log_to_wandb:
            wandb.log({
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "kld_loss": kld_loss.item()
            })
        
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")
    if log_to_wandb:
        wandb.log({"epoch_loss": total_loss/len(dataset)})

# Save the trained model
model.save_weights("vae_model.pth")
print("Model saved!")

# Finish the wandb run if enabled
if log_to_wandb:
    wandb.finish()
