import miditok

def augment_midi(midi_path, output_dir, transpositions=[-2, -1, 1, 2], tempo_changes=[0.9, 1.1]):
    """
    Augment a MIDI file by transposing and changing tempo.
    Saves augmented MIDI files to output_dir.
    """
    tokenizer = miditok.MIDITokenizer()
    midi = miditok.MidiFile(midi_path)
    tokens = tokenizer.midi_to_tokens(midi)
    
    for i, semitones in enumerate(transpositions):
        augmented_tokens = miditok.data_augmentation.transpose(tokens, semitones)
        augmented_midi = tokenizer.tokens_to_midi(augmented_tokens)
        augmented_midi.save(f"{output_dir}/transposed_{i}_{semitones}.mid")
    
    for i, tempo_factor in enumerate(tempo_changes):
        augmented_tokens = miditok.data_augmentation.change_tempo(tokens, tempo_factor)
        augmented_midi = tokenizer.tokens_to_midi(augmented_tokens)
        augmented_midi.save(f"{output_dir}/tempo_changed_{i}_{tempo_factor}.mid")
