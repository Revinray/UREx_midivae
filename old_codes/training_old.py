from marcus_preprocessing import midi_data_loader
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from old_codes.datasetcls import TokenizedMidiDataset
from vae import BetaVAE
# import os
import wandb
from tqdm import tqdm

# Set log_to_wandb to True or False
log_to_wandb = False  # Set to True to enable wandb logging, False to disable

# # Initialize wandb if enabled
# if log_to_wandb:
#     wandb.init(project="midi-vae-training", entity="your_wandb_username")

# Define model parameters
input_dim = 128           # Each time step has 128 features (piano roll)
hidden_dim = 256          # GRU hidden dimension
latent_dim = 64           # Size of the latent space
beta = 4.0                # Adjust beta for stronger disentanglement

# Load and preprocess tokenized data
tokenized_folder = "./tokens"
# dataset = TokenizedMidiDataset(tokenized_folder, limit_files=200, token_size=input_dim) # num_files limits the number of files to load, for testing purposes
dataset = midi_data_loader()

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BetaVAE(input_dim, hidden_dim, latent_dim, beta=beta)

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10

# # Log model parameters to wandb if enabled
# if log_to_wandb:
#     wandb.config.update({
#         "input_dim": input_dim,
#         "hidden_dim": hidden_dim,
#         "latent_dim": latent_dim,
#         "beta": beta,
#         "learning_rate": learning_rate,
#         "batch_size": batch_size,
#         "num_epochs": num_epochs
#     })

model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        # batch: (batch_size, seq_len, input_dim)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss, recon_loss, kld_loss = model.loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # # Log metrics to wandb if enabled
        # if log_to_wandb:
        #     wandb.log({
        #         "loss": loss.item(),
        #         "recon_loss": recon_loss.item(),
        #         "kld_loss": kld_loss.item()
        #     })
        
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")
    if log_to_wandb:
        wandb.log({"epoch_loss": total_loss/len(train_dataloader)})

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            recon_batch, mu, logvar = model(batch)
            loss, recon_loss, kld_loss = model.loss_function(recon_batch, batch, mu, logvar)
            test_loss += loss.item()
    
    test_loss /= len(test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")
    # if log_to_wandb:
    #     wandb.log({"test_loss": test_loss})
    model.train()

# # Save the trained model
# model.save_weights("vae_model.pth")
# print("Model saved!")

# # Finish the wandb run if enabled
# if log_to_wandb:
#     wandb.finish()
