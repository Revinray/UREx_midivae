from pathlib import Path
from miditok import REMI
tokenizer = REMI()  # using defaults parameters

import numpy as np

# # Tokenize a MIDI file
# tokens = tokenizer(Path("dataset_valid", "001_t0_0.mid"))  # automatically detects Score objects, paths, tokens
# # print shape of tokens
# print(tokens[0][0:10])

# Convert to MIDI and save it
tokens = [4, 214, 45, 121, 125, 216, 47, 120, 125, 218, 50, 122, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 126, 198, 47, 122, 127, 202, 52, 121, 136, 4, 190, 52, 120, 127, 194, 49, 121, 126, 198, 45, 121, 126, 202, 50, 122, 133, 214, 45, 120, 125, 216, 47, 120, 125, 218, 50, 121, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 127, 198, 47, 121, 126, 202, 52, 121, 130, 210, 45, 121, 126, 214, 52, 121, 126, 218, 50, 122, 139, 4, 4, 190, 50, 121, 129, 198, 50, 121, 128, 206, 49, 121, 126, 210, 50, 121, 125, 212, 49, 121, 127, 4, 190, 49, 120, 126, 194, 50, 121, 125, 196, 49, 121, 125, 200, 45, 121, 125, 204, 47, 121, 131, 218, 47, 121, 125, 220, 49, 120, 125, 4, 190, 50, 121, 130, 198, 50, 121, 129, 206, 49, 121, 126, 210, 50, 120, 125, 212, 49, 121, 126, 216, 47, 120, 127, 220, 45, 121, 140, 4, 4, 190, 50, 122, 128, 198, 50, 122, 129, 206, 52, 120, 126, 210, 54, 120, 125, 212, 52, 122, 126, 4, 190, 52, 120, 126, 194, 54, 121, 125, 196, 52, 121, 126, 200, 49, 121, 125, 204, 50, 122, 134, 218, 45, 121, 125, 220, 47, 121, 125, 4, 190, 50, 122, 125, 194, 54, 121, 125, 196, 52, 120, 125, 200, 50, 121, 125, 206, 52, 120, 126, 210, 54, 120, 125, 212, 52, 121, 126, 216, 50, 121, 126, 4, 190, 50, 121, 137, 214, 45, 121, 125, 216, 47, 120, 125, 218, 50, 122, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 126, 198, 47, 122, 127, 202, 52, 121, 136, 4, 190, 52, 120, 127, 194, 49, 121, 126, 198, 45, 121, 126, 202, 50, 122, 133, 214, 45, 120, 125, 216, 47, 120, 125, 218, 50, 121, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 127, 198, 47, 121, 126, 202, 52, 121, 130, 210, 45, 121, 126, 214, 52, 121, 126, 218, 54, 122, 139, 4, 214, 45, 121, 125, 216, 47, 120, 125, 218, 50, 122, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 126, 198, 47, 122, 127, 202, 52, 121, 136, 4, 190, 52, 120, 127, 194, 49, 121, 126, 198, 45, 121, 126, 202, 50, 122, 133, 214, 45, 120, 125, 216, 47, 120, 125, 218, 50, 121, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 127, 198, 47, 121, 126, 202, 52, 121, 130, 210, 45, 121, 126, 214, 52, 121, 126, 218, 50, 122, 139, 4, 4, 4, 4, 4, 4, 190, 50, 121, 129, 198, 50, 121, 128, 206, 49, 121, 126, 210, 50, 121, 125, 212, 49, 121, 127, 4, 190, 49, 120, 126, 194, 50, 121, 125, 196, 49, 121, 125, 200, 45, 121, 125, 204, 47, 121, 131, 218, 47, 121, 125, 220, 49, 120, 125, 4, 190, 50, 121, 130, 198, 50, 121, 129, 206, 49, 121, 126, 210, 50, 120, 125, 212, 49, 121, 126, 216, 47, 120, 127, 220, 45, 121, 140, 4, 4, 190, 50, 122, 128, 198, 50, 122, 129, 206, 52, 120, 126, 210, 54, 120, 125, 212, 52, 122, 126, 4, 190, 52, 120, 126, 194, 54, 121, 125, 196, 52, 121, 126, 200, 49, 121, 125, 204, 50, 122, 134, 218, 45, 121, 125, 220, 47, 121, 125, 4, 190, 50, 122, 125, 194, 54, 121, 125, 196, 52, 120, 125, 200, 50, 121, 125, 206, 52, 120, 126, 210, 54, 120, 125, 212, 52, 121, 126, 216, 50, 121, 126, 4, 190, 50, 121, 137, 214, 45, 121, 125, 216, 47, 120, 125, 218, 50, 122, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 126, 198, 47, 122, 127, 202, 52, 121, 136, 4, 190, 52, 120, 127, 194, 49, 121, 126, 198, 45, 121, 126, 202, 50, 122, 133, 214, 45, 120, 125, 216, 47, 120, 125, 218, 50, 121, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 127, 198, 47, 121, 126, 202, 52, 121, 130, 210, 45, 121, 126, 214, 52, 121, 126, 218, 54, 122, 139, 4, 214, 45, 121, 125, 216, 47, 120, 125, 218, 50, 122, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 126, 198, 47, 122, 127, 202, 52, 121, 136, 4, 190, 52, 120, 127, 194, 49, 121, 126, 198, 45, 121, 126, 202, 50, 122, 133, 214, 45, 120, 125, 216, 47, 120, 125, 218, 50, 121, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 127, 198, 47, 121, 126, 202, 52, 121, 130, 210, 45, 121, 126, 214, 52, 121, 126, 218, 50, 122, 139, 4, 4, 4, 4, 4, 4, 4, 4, 4, 214, 45, 121, 125, 216, 47, 120, 125, 218, 50, 122, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 126, 198, 47, 122, 127, 202, 52, 121, 136, 4, 190, 52, 120, 127, 194, 49, 121, 126, 198, 45, 121, 126, 202, 50, 122, 133, 214, 45, 120, 125, 216, 47, 120, 125, 218, 50, 121, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 127, 198, 47, 121, 126, 202, 52, 121, 130, 210, 45, 121, 126, 214, 52, 121, 126, 218, 54, 122, 139, 4, 214, 45, 121, 125, 216, 47, 120, 125, 218, 50, 122, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 126, 198, 47, 122, 127, 202, 52, 121, 136, 4, 190, 52, 120, 127, 194, 49, 121, 126, 198, 45, 121, 126, 202, 50, 122, 133, 214, 45, 120, 125, 216, 47, 120, 125, 218, 50, 121, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 127, 198, 47, 121, 126, 202, 52, 121, 130, 210, 45, 121, 126, 214, 52, 121, 126, 218, 50, 122, 139, 4, 4, 4, 4, 4, 214, 45, 121, 125, 216, 47, 120, 125, 218, 50, 122, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 126, 198, 47, 122, 127, 202, 52, 121, 136, 4, 190, 52, 120, 127, 194, 49, 121, 126, 198, 45, 121, 126, 202, 50, 122, 133, 214, 45, 120, 125, 216, 47, 120, 125, 218, 50, 121, 125, 220, 52, 120, 125, 4, 190, 54, 120, 125, 194, 50, 120, 127, 198, 47, 121, 126, 202, 52, 121, 130, 210, 45, 121, 126, 214, 52, 121, 126, 218, 50, 122, 139]  # MidiTok can handle PyTorch/Numpy/Tensorflow tensors
# Convert integers to token strings
token_strings = tokenizer._ids_to_tokens(tokens)
# Convert token strings back to MIDI
generated_midi = tokenizer([token_strings])
# print(len(tokens))
generated_midi.dump_midi(Path("decoded_ints.mid"))