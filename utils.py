import pretty_midi
import numpy as np

def midi_to_pianoroll(midi_path, fs=100):
    """
    Convert a MIDI file into a piano roll of shape [128, T].
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return piano_roll

def segment_pianoroll(piano_roll, segment_duration=5, fs=100):
    """
    Segment a piano roll into non-overlapping 5-second snippets.
    """
    segment_length = segment_duration * fs
    num_segments = piano_roll.shape[1] // segment_length
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments.append(piano_roll[:, start:end])
    return segments



