import torch.nn as nn
import torch.nn.functional as F
import torch

class BetaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=1.0):
        """
        Args:
            input_dim: Dimensionality of the input at each time step (e.g., 128 for piano roll).
            hidden_dim: Hidden size for GRU layers.
            latent_dim: Dimensionality of the latent space.
            beta: Weight for the KL divergence term.
        """
        super(BetaVAE, self).__init__()
        self.input_dim = input_dim
        self.beta = beta
        
        # Encoder: Using a 2-layer GRU
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Map latent vector to initial hidden state for GRU decoder
        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        # x: (batch, seq_len, input_dim)
        _, h_n = self.encoder_gru(x)
        # Take the last layer's hidden state: (batch, hidden_dim)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        batch_size = z.size(0)
        # Convert latent vector to initial hidden state for decoder GRU
        hidden = self.fc_latent_to_hidden(z)
        # Repeat for num_layers (here, 2 layers)
        hidden = hidden.unsqueeze(0).repeat(2, 1, 1)  # (num_layers, batch, hidden_dim)
        # For simplicity, use a zero input sequence (alternatively, use teacher forcing)
        decoder_input = torch.zeros(batch_size, seq_len, self.input_dim, device=z.device)
        out, _ = self.decoder_gru(decoder_input, hidden)
        reconstructed = self.output_layer(out)
        return reconstructed
    
    def forward(self, x):
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # Decode (assume reconstruction has same seq_len as input)
        recon_x = self.decode(z, x.size(1))
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss: mean squared error over all time steps and features
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # KL divergence loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld, recon_loss, kld
