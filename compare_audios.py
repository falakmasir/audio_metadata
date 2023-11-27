import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import webrtcvad

# Function to perform Voice Activity Detection (VAD) on audio file using WebRTC VAD
def perform_vad_with_webrtc(waveform, sample_rate):
    vad = webrtcvad.Vad(2)  # Using aggressive mode (level 2)
    frame_length = 30  # Frame length in milliseconds
    non_silent_segments = []

    # Convert the waveform to bytes (16-bit PCM)
    audio_bytes = waveform.numpy().tobytes()

    # Split audio into frames for VAD
    frame_duration = frame_length * sample_rate // 1000  # Duration of each frame
    for i in range(0, len(audio_bytes), frame_duration):
        frame = audio_bytes[i:i + frame_duration]
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            non_silent_segments.append(frame)

    # Combine non-silent segments into a single waveform tensor
    non_silent_waveform = torch.tensor(b''.join(non_silent_segments)).view(1, -1)

    return non_silent_waveform

# Function to calculate aggregated similarity score for two audio files
def calculate_aggregated_similarity(file_path1, file_path2):
    # Load audio files
    waveform1, sample_rate1 = torchaudio.load(file_path1)
    waveform2, sample_rate2 = torchaudio.load(file_path2)

    # Perform Voice Activity Detection (VAD) using WebRTC VAD
    non_silent_waveform1 = perform_vad_with_webrtc(waveform1, sample_rate1)
    non_silent_waveform2 = perform_vad_with_webrtc(waveform2, sample_rate2)

    # Ensure both non-silent waveforms have the same sample rate for compatibility
    if sample_rate1 != sample_rate2:
        raise ValueError("Sample rates of non-silent audio segments do not match.")

    # Initialize the Wav2Vec2 processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Extract embeddings from non-silent segments of both waveforms
    with torch.no_grad():
        input_values1 = processor(non_silent_waveform1.squeeze(), return_tensors="pt", padding=True).input_values
        input_values2 = processor(non_silent_waveform2.squeeze(), return_tensors="pt", padding=True).input_values

        embeddings1 = model(input_values1).last_hidden_state
        embeddings2 = model(input_values2).last_hidden_state

    # Calculate similarity scores for non-silent chunks
    similarity_scores = torch.nn.functional.cosine_similarity(embeddings1.mean(dim=1),
                                                             embeddings2.mean(dim=1),
                                                             dim=-1)

    # Aggregate similarity scores into a single score (using mean)
    aggregated_similarity_score = similarity_scores.mean().item()

    return aggregated_similarity_score

# Example usage:
similarity_score = calculate_aggregated_similarity("path/to/audio_file1.wav", "path/to/audio_file2.wav")
print("Aggregated Similarity Score:", similarity_score)
