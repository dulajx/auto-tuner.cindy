import librosa
import soundfile as sf
import numpy as np
import os

# Define the musical notes
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def pitch_to_note(pitch):
    """Convert pitch (in Hz) to closest musical note."""
    # Standard tuning frequency for A4 is 440 Hz
    A4 = 440.0
    # Calculate the semitone distance from A4
    semitone_distance = round(12 * np.log2(pitch / A4))
    # Find the closest note
    note_index = semitone_distance % 12
    return notes[note_index]

def correct_pitch(audio_path, output_path, target_note="A4"):
    """Correct the pitch of the audio to match the target note."""
    
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Get the pitch of the input audio using librosa's `piptrack`
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    
    # Extract the dominant pitch (index of the maximum magnitude)
    dominant_pitch = []
    for t in range(magnitudes.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        dominant_pitch.append(pitch)
    
    # Find the average pitch
    avg_pitch = np.mean(dominant_pitch)
    print(f"Detected pitch: {avg_pitch:.2f} Hz")

    # Convert target note to frequency
    target_freq = librosa.note_to_hz(target_note)
    print(f"Target frequency for {target_note}: {target_freq:.2f} Hz")
    
    # Use librosa's `resample` to adjust the pitch of the entire audio
    # For simplicity, we're scaling the pitch by the ratio of the target frequency to the detected pitch
    pitch_ratio = target_freq / avg_pitch
    y_tuned = librosa.effects.pitch_shift(y, sr, n_steps=12 * np.log2(pitch_ratio))

    # Save the tuned audio to the output path
    sf.write(output_path, y_tuned, sr)
    print(f"Pitch corrected and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    input_audio = "input_audio.wav"  # Path to input audio file (must be .wav)
    output_audio = "output_tuned_audio.wav"  # Path to save corrected audio
    correct_pitch(input_audio, output_audio, target_note="A4")
