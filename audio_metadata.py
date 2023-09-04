from pydub import AudioSegment, silence
import os
import numpy as np
import aubio

def calculate_pitch_in_memory(audio_data, sample_rate):
    samplerate = audio.frame_rate
    pitch_o = aubio.pitch("yin", samplerate)
    
    # Open the WAV file
    audio_file = aubio.source(temp_wav_file)
    
    # Initialize variables
    pitch_values = []
    
    # Process the audio file and calculate pitch
    total_frames = 0
    while True:
        samples, read = audio_file()
        pitch = pitch_o(samples)[0]
        pitch_values.append(pitch)
        total_frames += read
        if read < samplerate:
            break
    
    # Clean up
    audio_file.close()
    
    # Calculate average pitch
    average_pitch = sum(pitch_values) / len(pitch_values)
    
    return average_pitch

def gender_identification(audio_data, sample_rate):
    # Calculate the average pitch of the audio
    average_pitch = calculate_pitch_in_memory(audio_data, sample_rate)
    
    # Define pitch thresholds for gender identification
    male_threshold = 120.0  # Adjust as needed
    female_threshold = 220.0  # Adjust as needed
    
    # Determine gender based on pitch
    if average_pitch < male_threshold:
        return "Male"
    elif average_pitch > female_threshold:
        return "Female"
    else:
        return "Undetermined"

# Usage
import soundfile as sf

# Load audio data (you can replace this with your own audio loading logic)
audio_path = "your_audio_file.wav"
audio_data, sample_rate = sf.read(audio_path)

# Perform gender identification
gender = gender_identification(audio_data, sample_rate)
print("Gender:", gender)


def calculate_dynamic_threshold(audio, window_size=1000):
    """
    Calculate a dynamic silence threshold based on a moving average of amplitude levels.

    Args:
    - audio (AudioSegment): The audio segment to analyze.
    - window_size (int): The size of the sliding window for calculating the threshold.

    Returns:
    - float: The dynamic silence threshold.
    """
    audio = audio.set_channels(1)  # Convert to mono for analysis
    audio_samples = audio.get_array_of_samples()
    num_samples = len(audio_samples)
    dynamic_thresholds = []

    for i in range(0, num_samples, window_size):
        window = audio_samples[i:i + window_size]
        rms = np.sqrt(np.mean(np.square(window)))
        dynamic_thresholds.append(rms)

    # Calculate the threshold as a percentile (e.g., 10th percentile)
    threshold = np.percentile(dynamic_thresholds, 10)

    return threshold

def detect_clipping(audio, threshold=0.99):
    max_amplitude = audio.max
    threshold_amplitude = max_amplitude * threshold

    clipped_samples = [i for i, sample in enumerate(audio) if abs(sample) >= threshold_amplitude]

    if clipped_samples:
        print("Clipping detected in samples:", clipped_samples)
        return len(clipped_samples) / len(audio)
    else:
        print("No clipping detected.")
        return 0.0

def get_audio_metadata(audio_file_path):
    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(audio_file_path)

        # Calculate the dynamic silence threshold
        silence_threshold = calculate_dynamic_threshold(audio)

        # Detect audio clipping and calculate the percentage of clipped frames
        clipped_percentage = detect_clipping(audio)

        # Split audio into non-silent chunks using the dynamic threshold
        non_silent_chunks = silence.split_on_silence(audio, silence_thresh=silence_threshold)

        # Calculate ambient noise level (average RMS amplitude of silent chunks)
        silent_chunks_rms = [chunk.rms for chunk in non_silent_chunks]
        ambient_noise_level = np.mean(silent_chunks_rms)

        # Extract metadata
        duration_in_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds
        channels = audio.channels
        sample_width = audio.sample_width
        frame_rate = audio.frame_rate

        return {
            'File Name': os.path.basename(audio_file_path),
            'Duration (s)': duration_in_seconds,
            'Channels': channels,
            'Sample Width (bytes)': sample_width,
            'Frame Rate (Hz)': frame_rate,
            'Ambient Noise Level': ambient_noise_level,
            'Dynamic Silence Threshold': silence_threshold,
            'Percentage of Clipped Frames': clipped_percentage * 100  # Convert to percentage
        }
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Specify the path to your audio file
    audio_file_path = "path/to/your/audio/file.mp3"  # Change this to your audio file's path

    # Get and print audio metadata
    metadata = get_audio_metadata(audio_file_path)
    if isinstance(metadata, dict):
        print("Audio Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        print("Error:", metadata)
