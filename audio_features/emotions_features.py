import torch
import torch.nn.functional as F
import torchaudio

from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

model_id = 'Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition'
model_id = 'Aniemore/wavlm-bert-fusion-s-emotion-russian-resd'
model_id = 'Aniemore/wavlm-emotion-russian-resd'

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model_ = AutoModel.from_pretrained(model_id, trust_remote_code=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_.to(device)

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    # Ensure we resample to the target sampling_rate (16000)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict_segment(speech, sampling_rate):
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model_(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    # Return a dictionary of {label: score}
    return {config.id2label[i]: float(score) for i, score in enumerate(scores)}


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    scores = predict_segment(speech, sampling_rate)
    outputs = [{"Emotion": k, "Score": f"{round(v * 100, 3):.1f}%"} for k, v in scores.items()]
    return outputs


def process_long_audio(path, window_size=5.0, stride=2.0):
    """
    Process audio with a sliding window to create a time series of emotions.
    window_size: length of each segment in seconds
    stride: step size in seconds
    """
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

        top_emotion = max(scores, key=scores.get)

        print(top_emotion)

        results.append({
            "start": i / sampling_rate,
            "end": (i + window_samples) / sampling_rate,
            "emotion": top_emotion,
            "scores": scores
        })

    return results

# Example usage:
# result = predict("sample1.wav", 16000)
# print(result)

# Process a long file
timeline = process_long_audio("/home/trsuser/Downloads/эмоции/norm/4729977882178254307_EDIT.wav", window_size=5, stride=2)
for entry in timeline:
    print(f"{entry['start']:.1f}s - {entry['end']:.1f}s: {entry['emotion']} ({entry['scores'][entry['emotion']]:.2f})")