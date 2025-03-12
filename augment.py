# import os
from pathlib import Path
from miditok import REMI
from miditok.data_augmentation import augment_dataset
import time

# # A validation function to discard undesired MIDIs.
# def midi_valid(midi) -> bool:
#     # Only accept MIDIs with a 4-beat bar (4/* time signatures).
#     if any(ts.numerator != 4 for ts in midi.time_signature_changes):
#         return False
#     return True

def main():
    # Define folder paths.
    midi_folder_path = Path("./POP909_MIDIs").resolve()  # Original MIDI files.
    
    # Folders for augmented outputs.
    augmented_pitch_path = Path("./augmented_pitch").resolve()      # Augmented for key changes (pitch shifts).
    augmented_dynamics_path = Path("./augmented_dynamics").resolve()  # Augmented for dynamics (velocity changes).
    augmented_note_duration_path = Path("./augmented_note_duration").resolve()        # Augmented for note duration changes. (i.e. staccato-ness and legato-ness)

    # Folder where tokenized JSON files will be saved.
    tokens_output_path = Path("./tokens").resolve()

    # Create output folders.
    for folder in [augmented_pitch_path, augmented_dynamics_path, augmented_note_duration_path, tokens_output_path]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # ## 1. Data Augmentation
    # print("Starting pitch augmentation (key changes)...")
    # augment_dataset(
    #     data_path=midi_folder_path,
    #     pitch_offsets=[-12, 12],    # Transpose down/up one octave.
    #     velocity_offsets=[],        # No velocity change.
    #     duration_offsets=[],        # No duration change.
    #     out_path=augmented_pitch_path,
    #     copy_original_in_new_location=False,
    # )
    # print(f"Pitch augmentation complete. Files saved in: {augmented_pitch_path}")
    
    # print("Starting dynamics augmentation (velocity changes)...")
    # augment_dataset(
    #     data_path=midi_folder_path,
    #     pitch_offsets=[],           # No pitch change.
    #     velocity_offsets=[-30, 25],   # Adjust velocities.
    #     duration_offsets=[],        # No duration change.
    #     out_path=augmented_dynamics_path,
    #     copy_original_in_new_location=False,
    # )
    # print(f"Dynamics augmentation complete. Files saved in: {augmented_dynamics_path}")
    
    # print("Starting note duration augmentation (duration changes)...")
    # augment_dataset(
    #     data_path=midi_folder_path,
    #     pitch_offsets=[],           # No pitch change.
    #     velocity_offsets=[],        # No velocity change.
    #     duration_offsets=[-0.5, 1], # Alter note durations. (i.e. staccato-ness and legato-ness)
    #     out_path=augmented_note_duration_path,
    #     copy_original_in_new_location=False,
    # )
    # print(f"Note Duration augmentation complete. Files saved in: {augmented_note_duration_path}")



    ## 2. Tokenization of one augmented dataset (as an example).
    # Here we tokenize the pitch-augmented dataset.
    tokenizer = REMI()
    
    print("Starting tokenization of the pitch-augmented dataset...")
    tokenizer.tokenize_dataset(
        augmented_pitch_path,      # Input dataset: the augmented pitch folder.
        tokens_output_path,        # Output folder for tokenized JSON files.
        # midi_valid                 # Validation function.
    )
    print(f"Pitch Augment Tokenization complete. Token files saved in: {tokens_output_path}")
    
    print("Starting tokenization of the note duration-augmented dataset...")
    tokenizer.tokenize_dataset(
        augmented_note_duration_path,      # Input dataset: the augmented tempo folder.
        tokens_output_path,        # Output folder for tokenized JSON files.
        # midi_valid                 # Validation function.
    )
    print(f"Note Duration Augment Tokenization complete. Token files saved in: {tokens_output_path}")
    
    print("Starting tokenization of the dynamics-augmented dataset...")
    tokenizer.tokenize_dataset(
        augmented_dynamics_path,      # Input dataset: the augmented dynamics folder.
        tokens_output_path,        # Output folder for tokenized JSON files.
        # midi_valid                 # Validation function.
    )
    print(f"Dynamics Tokenization complete. Token files saved in: {tokens_output_path}")
    
    print(f"🎉 Magnificent! The grand tokenization process has reached its glorious conclusion. All token files have been meticulously saved in: {tokens_output_path}")

if __name__ == "__main__":
    main()
