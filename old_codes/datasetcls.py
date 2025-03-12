import os
import json
import torch
from torch.utils.data import Dataset
from utils import midi_to_pianoroll, segment_pianoroll
import numpy as np

class TokenizedMidiDataset(Dataset):
    def __init__(self, tokenized_folder, limit_files=-1, token_size=128):
        # this param limits the number of files to load, for testing purposes
        self.token_size = token_size
        self.data = self.load_and_preprocess_data(tokenized_folder, limit_files=limit_files)

    def load_and_preprocess_data(self, tokenized_folder, limit_files=-1):
        tokenized_data = []
        file_count = 0
        # get file names
        file_names = os.listdir(tokenized_folder)

        for file_name in file_names:
            if file_name.endswith('.json'):
                with open(os.path.join(tokenized_folder, file_name), 'r') as f:
                    data = json.load(f)

                    # Split the tokenized data into sequences of token_size
                    # Note that ids contains 3 arrays (representing the 3 parts of the MIDI file)
                    for i in range(0, len(data['ids'][0]), self.token_size):
                        segment = [seq[i:i+self.token_size] for seq in data['ids']]
                    # Ensure all segments have the same length
                    if all(len(s) == self.token_size for s in segment):
                        tokenized_data.append(segment)
                    else:
                        # pad the last segment if it's shorter than token_size
                        padded_segment = [np.pad(s, (0, self.token_size - len(s))) for s in segment]
                        tokenized_data.append(padded_segment)

                    file_count += 1
                    if limit_files > 0 and file_count >= limit_files:
                        break


        print(f"Loaded {file_count} files")
        
        # Flatten the list of lists and convert to PyTorch tensor
        # flattened_data = [item for sublist in tokenized_data for item in sublist]
        flattened_data = tokenized_data
        tensor_data = [torch.tensor(seq, dtype=torch.float) / (self.token_size - 1.0) for seq in flattened_data]  # Normalize to [0, 1]
        # print dimensions of the tensor
#         print(tensor_data[0].shape)
#         print(len(tensor_data))
        return tensor_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MidiDataset(Dataset):
    def __init__(self, midi_files, fs=100, segment_duration=5):
        self.segments = []
        for midi_file in midi_files:
            pr = midi_to_pianoroll(midi_file, fs=fs)
            segs = segment_pianoroll(pr, segment_duration=segment_duration, fs=fs)
            # Normalize velocities to [0,1] (assuming 0-127)
            segs = [np.clip(seg / 127.0, 0, 1) for seg in segs]
            # Transpose so that each segment is (time_steps, 128)
            segs = [seg.T for seg in segs]
            self.segments.extend(segs)
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        seg = self.segments[idx]
        seg = torch.FloatTensor(seg)  # shape: (seq_len, 128)
        return seg





#     def load_and_preprocess_data(self, tokenized_folder, limit_files=-1):
#         tokenized_data = []
#         file_count = 0
#         # get file names
#         file_names = os.listdir(tokenized_folder)

#         for file_name in file_names:
#             if file_name.endswith('.json'):
#                 with open(os.path.join(tokenized_folder, file_name), 'r') as f:
#                     data = json.load(f)

#                     # Split the tokenized data into sequences of token_size
#                     # Note that ids contains 3 arrays (representing the 3 parts of the MIDI file)
#                     for i in range(0, len(data['ids'][0]), self.token_size):
#                         segment = [seq[i:i+self.token_size] for seq in data['ids']]
#                     # Ensure all segments have the same length
#                     if all(len(s) == self.token_size for s in segment):
#                         tokenized_data.append(segment)
#                     else:
#                         # pad the last segment if it's shorter than token_size
#                         padded_segment = [np.pad(s, (0, self.token_size - len(s))) for s in segment]
#                         tokenized_data.append(padded_segment)

#                     file_count += 1
#                     if limit_files > 0 and file_count >= limit_files:
#                         break


#         print(f"Loaded {file_count} files")
        
#         # Flatten the list of lists and convert to PyTorch tensor
#         # flattened_data = [item for sublist in tokenized_data for item in sublist]
#         flattened_data = tokenized_data
#         tensor_data = [torch.tensor(seq, dtype=torch.float) / (self.token_size - 1.0) for seq in flattened_data]  # Normalize to [0, 1]
#         # print dimensions of the tensor
# #         print(tensor_data[0].shape)
# #         print(len(tensor_data))
#         return tensor_data
    

