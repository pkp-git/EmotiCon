import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    HubertForSequenceClassification, Wav2Vec2FeatureExtractor
)
import assemblyai as aai
import soundfile as sf
from dotenv import load_dotenv
import os

load_dotenv()

# Text emotion model setup
TEXT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)

# Audio emotion model setup (HuBERT)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
audio_model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")

# AssemblyAI API key from .env
aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
if not aai.settings.api_key:
    raise RuntimeError("ASSEMBLYAI_API_KEY not set in environment or .env file.")
transcriber = aai.Transcriber()

def get_audio_emotions(audio, sr, window_size=5.0):
    """Segment audio into non-overlapping windows and get emotion for each window using HuBERT."""
    input_segments = []
    timestamps = []
    total_length = len(audio) / sr
    num_segments = int(total_length // window_size)
    for i in range(num_segments):
        start = i * window_size
        end = start + window_size
        s = int(start * sr)
        e = int(end * sr)
        segment = audio[s:e]
        if len(segment) < int(window_size * sr):
            break
        input_segments.append(segment)
        timestamps.append((start, end))
    # Batch process for efficiency
    features = feature_extractor(input_segments, sampling_rate=sr, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = audio_model(**features).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_ids = torch.argmax(probs, dim=-1)
    emotions = []
    for i, idx in enumerate(predicted_ids.tolist()):
        label = audio_model.config.id2label[idx]
        confidence = probs[i][idx].item()
        emotions.append((timestamps[i][0], timestamps[i][1], label, confidence))
    return emotions

def merge_segments(emotions):
    """Merge consecutive segments with same emotion."""
    merged = []
    if not emotions:
        return merged
    prev = emotions[0]
    for curr in emotions[1:]:
        if curr[2] == prev[2]:
            prev = (prev[0], curr[1], prev[2], max(prev[3], curr[3]))
        else:
            merged.append(prev)
            prev = curr
    merged.append(prev)
    return merged

def transcribe_segment(audio, sr, start, end, temp_wav_path="temp.wav"):
    """Transcribe audio segment using AssemblyAI."""
    segment = audio[int(start*sr):int(end*sr)]
    sf.write(temp_wav_path, segment, sr)
    transcript = transcriber.transcribe(temp_wav_path)
    return transcript.text

def get_text_emotion(text):
    """Get emotion from text."""
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = text_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        label_id = torch.argmax(probs).item()
        label = text_model.config.id2label[label_id]
        confidence = probs[label_id].item()
    return label, confidence

def main(wav_path, segments_csv="segments.csv", plot_png="emotion_segments.png"):
    import os
    audio, sr = librosa.load(wav_path, sr=16000)
    # Use the directory of segments_csv for all outputs
    data_dir = os.path.dirname(os.path.abspath(segments_csv)) or "."
    temp_wav_path = os.path.join(data_dir, "temp.wav")
    plot_png = os.path.join(data_dir, os.path.basename(plot_png))
    emotions = get_audio_emotions(audio, sr)
    segments = merge_segments(emotions)
    rows = []
    colors = []
    emotion_color_map = {}
    color_palette = plt.cm.get_cmap('tab10', 10)
    for idx, (start, end, audio_emotion, audio_conf) in enumerate(segments):
        text = transcribe_segment(audio, sr, start, end, temp_wav_path=temp_wav_path)
        text_emotion, text_conf = get_text_emotion(text)
        rows.append({
            "Segment": f"{start:.2f}-{end:.2f}",
            "Audio Emotion": f"{audio_emotion} ({audio_conf:.2f})",
            "Transcript": text,
            "Text Emotion": f"{text_emotion} ({text_conf:.2f})"
        })
        if audio_emotion not in emotion_color_map:
            emotion_color_map[audio_emotion] = color_palette(len(emotion_color_map))
        colors.append(emotion_color_map[audio_emotion])
    df = pd.DataFrame(rows)
    df.to_csv(segments_csv, index=False)
    # Plot
    fig, ax = plt.subplots(figsize=(12, 2))
    for i, (start, end, emotion, _) in enumerate(segments):
        ax.axvspan(start, end, color=emotion_color_map[emotion], label=emotion if i == 0 else "")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    handles = [plt.Rectangle((0,0),1,1, color=emotion_color_map[e]) for e in emotion_color_map]
    ax.legend(handles, emotion_color_map.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_png)
    print(f"Saved {segments_csv}, {plot_png}, and {temp_wav_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("Participant 9/Participant Audio 9.wav")