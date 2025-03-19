from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


from miditok import REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator

from vae import BetaVAE

def midi_data_loader(folder, shuffle=True):
    tokenizer = REMI()  # using defaults parameters
    midi_paths = [path.resolve() for path in Path(folder).rglob("*.mid")][:100]

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


# Example function to generate from an existing tokenized MIDI file
def generate_from_token_file(file_path):
    # Create a small dataset/loader from the single file
    # tokenizer = REMI()
    single_dataset = DatasetMIDI(
        files_paths=[Path(file_path)],
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    single_loader = DataLoader(single_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    model.eval()
    with torch.no_grad():
        batch = next(iter(single_loader))
        tokens = batch["input_ids"].to(device)  # shape: (1, seq_len)
        
        # Embed tokens
        embedded = embedding_layer(tokens.long())  # shape: (1, seq_len, input_dim)
        
        # Encode to latent
        mu, logvar = model.encode(embedded.float())
        z = model.reparameterize(mu, logvar)
        
        # Decode back to feature vectors
        decoded = model.decode(z, seq_len=256)  # pick a sequence length
        predicted_tokens = torch.argmax(decoded, dim=-1)  # shape: (1, seq_len)

        # convert predicted tokens to a plain Python list, so that __ids_to_tokens can read it
        predicted_tokens = predicted_tokens.squeeze().tolist()
        
        # Convert integers to token strings
        token_strings = tokenizer._ids_to_tokens(predicted_tokens)
        # Convert token strings back to MIDI
        generated_midi = tokenizer([token_strings])
        # print(len(tokens))
        generated_midi.dump_midi(Path("trained_decoded_estimate.mid"))


test_midi_file_path = "dataset_valid/001_t0_0.mid"
generate_from_token_file(test_midi_file_path)









# import miditoolkit
# from miditoolkit import MidiFile
# # Generate a random latent vector z and decode it to a sequence of length seq_len
# model.eval()
# with torch.no_grad():
#     ## generate a random latent vector z
#     seq_len = 256
#     z = torch.randn((1, latent_dim)).to(device)
#     decoded = model.decode(z, seq_len)  # shape: (1, seq_len, input_dim)

#     # Convert probabilities to token IDs
#     predicted_tokens = torch.argmax(decoded, dim=-1)  # shape: (1, seq_len)
#     generated_tokens = predicted_tokens

#     # print shape of generated_tokens
#     print("Shape of generated_tokens:", generated_tokens.shape)

#     ## save to MIDI file
#     tokenizer = REMI()
#     score_object = tokenizer.decode(generated_tokens)
#     print(score_object)
    
#     midi_file = MidiFile()
#     midi_file.ticks_per_beat = score_object.tpq

#     # Add tracks and events from the Score object to the MidiFile object
#     for track in score_object.tracks:
#         print(track.program)
#         midi_track = miditoolkit.midi.containers.Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
#         for note in track.notes:
#             midi_track.notes.append(miditoolkit.midi.containers.Note(
#                 start=note.start,
#                 end=note.end,
#                 pitch=note.pitch,
#                 velocity=note.velocity
#             ))
#         midi_file.instruments.append(midi_track)

#     # Save the MidiFile object to a file
#     midi_file.dump(Path("decoded_midi.mid"))









        # generated_midi = tokenizer(predicted_tokens)
        # generated_midi.dump_midi(Path(f"decoded{Path(file_path).stem}.mid"))
        
        # # Convert tokens back to MIDI
        # generated_tokens = predicted_tokens.squeeze().tolist()
        # score_object = tokenizer.decode([generated_tokens])

        # midi_file = MidiFile()
        # midi_file.ticks_per_beat = score_object.tpq

        # # Add tracks and events from the Score object to the MidiFile object
        # for track in score_object.tracks:
        #     print(track.program)
        #     midi_track = miditoolkit.midi.containers.Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
        #     for note in track.notes:
        #         midi_track.notes.append(miditoolkit.midi.containers.Note(
        #             start=note.start,
        #             end=note.end,
        #             pitch=note.pitch,
        #             velocity=note.velocity
        #         ))
        #     midi_file.instruments.append(midi_track)

        # output_name = f"decoded_midi_{Path(file_path).stem}.mid"
        # # Save the MidiFile object to a file
        # midi_file.dump(Path(output_name))

        # print(f"Wrote generated MIDI to {output_name}")

# print(tokenizer.vocab)
# print("Vocab size:", len(tokenizer.vocab))



# # Generate a random latent vector z and decode it to a sequence of length seq_len
# model.eval()
# with torch.no_grad():
#     ## generate a random latent vector z
#     seq_len = 256
#     z = torch.randn((1, latent_dim)).to(device)
#     decoded = model.decode(z, seq_len)  # shape: (1, seq_len, input_dim)

#     # Convert probabilities to token IDs
#     predicted_tokens = torch.argmax(decoded, dim=-1)  # shape: (1, seq_len)
#     generated_tokens = predicted_tokens

#     # print shape of generated_tokens
#     print("Shape of generated_tokens:", generated_tokens.shape)

#     ## save to MIDI file
#     tokenizer = REMI()
#     score_object = tokenizer.decode(generated_tokens)
#     print(score_object)
    
#     midi_file = MidiFile()
#     midi_file.ticks_per_beat = score_object.tpq

#     # Add tracks and events from the Score object to the MidiFile object
#     for track in score_object.tracks:
#         print(track.program)
#         midi_track = miditoolkit.midi.containers.Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
#         for note in track.notes:
#             midi_track.notes.append(miditoolkit.midi.containers.Note(
#                 start=note.start,
#                 end=note.end,
#                 pitch=note.pitch,
#                 velocity=note.velocity
#             ))
#         midi_file.instruments.append(midi_track)

#     # Save the MidiFile object to a file
#     midi_file.dump(Path("decoded_midi.mid"))

