import torch
import torch.nn.functional as F
import torchaudio
import argparse

import pandas as pd

from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor
from pathlib import Path

model_id = 'Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition'

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model_ = AutoModel.from_pretrained(model_id, trust_remote_code=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_.to(device)

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)

    # Mix to mono if stereo.
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict_segment(speech, sampling_rate):
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model_(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    return {config.id2label[i]: float(score) for i, score in enumerate(scores)}

def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    scores = predict_segment(speech, sampling_rate)
    outputs = [{"Emotion": k, "Score": f"{round(v * 100, 3):.1f}%"} for k, v in scores.items()]
    return outputs

def extract_timeseries(path, window_size=5.0, stride=2.0) -> pd.DataFrame:
    sampling_rate = 16000
    speech = speech_file_to_array_fn(path, sampling_rate)

    results = []
    window_samples = int(window_size * sampling_rate)
    stride_samples = int(stride * sampling_rate)

    # Iterate over the audio array with a sliding window
    # Use shape[-1] to get time dimension for both mono (N,) and stereo (2, N)
    for i in range(0, speech.shape[-1] - window_samples, stride_samples):
        # Use ... to slice the last dimension (time)
        chunk = speech[..., i : i + window_samples]

        scores = predict_segment(chunk, sampling_rate)

        # Create a row for the dataframe
        row = {
            "timestamp": i / sampling_rate,
            **scores  # Unpack emotion scores directly into the row
        }
        results.append(row)

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/audio/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/audio_features/emotions/csv"))

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    files_to_process = []
    if args.input.is_dir():
        files_to_process = list(args.input.glob("*.wav"))
    elif args.input.is_file() and args.input.suffix == ".wav":
        files_to_process = [args.input]
    else:
        print(f"No valid input found at {args.input}")
        return

    for wav_file in files_to_process:
        print(f"Processing {wav_file}...")
        try:
            ts_dataframe = extract_timeseries(wav_file, window_size=5, stride=2)

            output_csv = args.output / f"{wav_file.stem}.csv"
            ts_dataframe.to_csv(output_csv, index=False)
            print(f"Saved to {output_csv}")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

if __name__ == "__main__":
    main()