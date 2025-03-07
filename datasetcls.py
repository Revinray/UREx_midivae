import torch
from torch.utils.data import Dataset
from utils import midi_to_pianoroll, segment_pianoroll
import numpy as np

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
