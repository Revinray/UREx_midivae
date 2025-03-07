import miditok

def augment_midi(midi_path, output_dir, transpositions=[-2, -1, 1, 2], tempo_changes=[0.9, 1.1], dynamics_changes=[0.9, 1.1], mode_changes=['aeolian', 'dorian']):
    """
    Augment a MIDI file by transposing and changing tempo.
    Saves augmented MIDI files to output_dir.
    """
    tokenizer = miditok.MusicTokenizer()
    midi = miditok.MidiFile(midi_path)
    tokens = tokenizer.midi_to_tokens(midi)
    
    # # don't use transpose cos ram said so
    # for i, semitones in enumerate(transpositions):
    #     augmented_tokens = miditok.data_augmentation.transpose(tokens, semitones)
    #     augmented_midi = tokenizer.tokens_to_midi(augmented_tokens)
    #     augmented_midi.save(f"{output_dir}/transposed_{i}_{semitones}.mid")
    
    # Transpose the MIDI file
    for i, tempo_factor in enumerate(tempo_changes):
        augmented_tokens = miditok.data_augmentation.change_tempo(tokens, tempo_factor)
        augmented_midi = tokenizer.tokens_to_midi(augmented_tokens)
        augmented_midi.save(f"{output_dir}/tempo_changed_{i}_{tempo_factor}.mid")

    # Transpose the Dynamics 
    for i, dynamics_factor in enumerate(dynamics_changes):
        augmented_tokens = miditok.data_augmentation.change_dynamics(tokens, dynamics_factor)
        augmented_midi = tokenizer.tokens_to_midi(augmented_tokens)
        augmented_midi.save(f"{output_dir}/dynamics_changed_{i}_{dynamics_factor}.mid")

    # Change the mode (e.g. aeolian, dorian)
    for i, mode in enumerate(mode_changes):
        augmented_tokens = miditok.data_augmentation.change_mode(tokens, mode)
        augmented_midi = tokenizer.tokens_to_midi(augmented_tokens)
        augmented_midi.save(f"{output_dir}/mode_changed_{i}_{mode}.mid")


if __name__ == "__main__":
    import os
    midi_folder_path = "./POP909_MIDIs"
    output_folder_path = "./augmented"
    os.makedirs(output_folder_path, exist_ok=True)

    for midi_file in os.listdir(midi_folder_path):
        midi_path = os.path.join(midi_folder_path, midi_file)
        augment_midi(midi_path, output_folder_path)
        raise Exception("stop")