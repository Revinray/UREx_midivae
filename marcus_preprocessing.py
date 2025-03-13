# Preprocess RAW MIDI files (5-second segments) into tokenized format to be fed directly into the VAE model.
#
# To use miditok's REMI tokenizer
#
# TODO

from miditok import REMI
from miditoolkit import MidiFile
from symusic import Score
from pathlib import Path
from random import shuffle

from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader

# import torch
# def pad_sequences(sequences, max_len, pad_token_id):
#     padded_sequences = []
#     for seq in sequences:
#         if len(seq) < max_len:
#             padded_seq = seq + [pad_token_id] * (max_len - len(seq))
#         else:
#             padded_seq = seq[:max_len]
#         padded_sequences.append(padded_seq)
#     return torch.tensor(padded_sequences)

def midi_generator():
    # Creates the tokenizer and list the file paths
    tokenizer = REMI()  # using defaults parameters
    midi_paths = [path.resolve() for path in Path("POP909_MIDIs").rglob("*.mid")]

    # Builds the vocabulary with BPE
    tokenizer.train(vocab_size=30000, files_paths=midi_paths)


    ## Prepare a dataset before training

    total_num_files = len(midi_paths)
    num_files_valid = round(total_num_files * 0.15)
    num_files_test = round(total_num_files * 0.15)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
    midi_paths_train = midi_paths[num_files_valid + num_files_test:]

    # Chunk MIDIs and perform data augmentation on each subset independently
    for files_paths, subset_name in (
        (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
    ):

        # Split the MIDIs into chunks of sizes approximately about 1024 tokens
        subset_chunks_dir = Path(f"dataset_{subset_name}").resolve()
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=1024,
            num_overlap_bars=2,
        )

        # Perform data augmentation
        augment_dataset(
            subset_chunks_dir,
            pitch_offsets=[-12, 12],
            velocity_offsets=[-4, 4],
            duration_offsets=[-0.5, 0.5],
        )


# from torch import nn
# ## Create a Dataset and collator for training
# def midi_data_loader():
#     tokenizer = REMI()  # using defaults parameters (constants.py)
#     midi_paths = [path.resolve() for path in Path("POP909_MIDIs").rglob("*.mid")]

#     dataset = DatasetMIDI(
#         files_paths=midi_paths,
#         tokenizer=tokenizer,
#         max_seq_len=1024,
#         bos_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer["BOS_None"],
#     )
#     collator = DataCollator(tokenizer.pad_token_id)
#     data_loader = DataLoader(dataset=dataset, collate_fn=collator)

#     embedded_data = []

#     count = 0
#     # Using the data loader in the training loop
#     for batch in data_loader:
#         # print("Train your model on this batch...")
#         pass





    #     # extract the input_ids
    #     input_ids = batch["input_ids"]
    #     # labels = batch["attention_mask"]

    #     # # perform padding if length != 1024
    #     # if input_ids.shape[1] != 1024:
    #     #     input_ids = pad_sequences(input_ids, 1024, tokenizer.pad_token_id)

    #     # perform embedding
    #     vocab_size = 30000
    #     embedding_dim = 128
    #     embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    #     embedded_tokens = embedding_layer(input_ids)

    #     # flatten the input
    #     embedded_tokens = embedded_tokens.view(-1, embedded_tokens.size(-1))

    #     # if dimension not [1022, 128], then pad
    #     if embedded_tokens.shape[0] != 1022:
    #         padding = nn.ConstantPad1d((0, 0, 0, 1022 - embedded_tokens.shape[0]), 0)
    #         embedded_tokens = padding(embedded_tokens)

    #     print(embedded_tokens.shape)
    #     # print(embedded_tokens)

    #     embedded_data.append(embedded_tokens)

    #     count +=1

    #     if count > 10:
    #         break


    #     # # print size of the batch
    #     # print(batch["input_ids"].shape)
    #     # print(batch)

    #     # raise SystemExit

    # return embedded_data
    

if __name__ == "__main__":
    # midi_generator() # this converts the raw MIDI, augments them and splits them to test, train and valid sets
    # midi_data_loader() # this loads the data for training (to be used in the training loop)
    pass
