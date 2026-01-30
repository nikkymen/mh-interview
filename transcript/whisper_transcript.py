import torch
import argparse
import soundfile as sf

from pathlib import Path

from transformers.pipelines import pipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def has_ampere_or_newer() -> bool:
    """Check if GPU has compute capability 8.0+ (Ampere or newer)."""
    if not torch.cuda.is_available():
        return False

    # Get compute capability of the first GPU
    major, minor = torch.cuda.get_device_capability(0)
    compute_capability = major + minor / 10

    # Ampere architecture starts at compute capability 8.0
    return compute_capability >= 8.0

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if torch.cuda.is_available() and has_ampere_or_newer():
    torch_dtype = torch.bfloat16
elif torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

device = torch.device(device)

attn_impl = "flash_attention_2" if (device.type == "cuda" and has_ampere_or_newer()) else "eager"

def format_timestamp(seconds: float) -> str:
    """Converts seconds to MM:SS.ss format."""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02}:{remaining_seconds:05.2f}"

def extract_transcription(file_path: Path, model_path = Path("models/whisper-large-v3-russian")) -> str:
    whisper = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=attn_impl,
        local_files_only=True,
    )

    processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        batch_size=16,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )

    audio_array, sample_rate = sf.read(file_path)

    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = audio_array[:, 0]

    # Get the transcription
    asr = asr_pipeline(audio_array, generate_kwargs={"language": "russian", "max_new_tokens": 256}, return_timestamps=True)

    result_chunks = []

    for chunk in asr['chunks']:
        start = chunk['timestamp'][0]
        end = chunk['timestamp'][1]
        text = chunk['text']
        result_chunks.append(f"[{format_timestamp(start)} - {format_timestamp(end)}] {text.strip()}")

    return '\n'.join(result_chunks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default='data/audio/wav/norm')
    parser.add_argument("--output", type=Path, default=Path("data/transcript"))

if __name__ == "__main__":
    main()