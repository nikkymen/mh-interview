import torch
import soundfile as sf
import os
import time
import re

from transformers.pipelines import pipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def format_timestamp(seconds: float) -> str:
    """Converts seconds to MM:SS.ss format."""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02}:{remaining_seconds:05.2f}"

torch_dtype = torch.bfloat16 # set your preferred type here

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
    setattr(torch.distributed, "is_initialized", lambda : False) # monkey patching
device = torch.device(device)

print(f"Using device: {device}")

whisper = WhisperForConditionalGeneration.from_pretrained(
    "antony66/whisper-large-v3-russian", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    attn_implementation="flash_attention_2"
)

processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=whisper,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
   # chunk_length_s=30, # Этот параметр портит длинные записи
    batch_size=16,
    return_timestamps="word",
    torch_dtype=torch_dtype,
    device=device,
)

# Directory containing WAV files
input_dir = 'data_new'
# Directory to save transcriptions (same as input directory)
output_dir = input_dir

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all WAV files in the input directory
wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]

if not wav_files:
    print(f"No WAV files found in {input_dir}")
    exit(1)

print(f"Found {len(wav_files)} WAV files to process")

# Process each WAV file
for i, wav_file in enumerate(wav_files):
    print(f"Processing file {i+1}/{len(wav_files)}: {wav_file}")
    start_time = time.time()

    # Construct full path to the input WAV file
    wav_path = os.path.join(input_dir, wav_file)

    txt_file_raw = os.path.splitext(wav_file)[0] + '.txt'
    txt_file_ts = os.path.splitext(wav_file)[0] + '_ts.txt'

    try:
        # Load the audio
        audio_array, sample_rate = sf.read(wav_path)

        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array[:, 0]

        # Get the transcription
        asr = asr_pipeline(audio_array, generate_kwargs={"language": "russian", "max_new_tokens": 256}, return_timestamps=True)

        with open(os.path.join(output_dir, txt_file_raw), 'w', encoding='utf-8') as f:
            # Replace sentence-ending punctuation followed by space with punctuation and newline
            text = re.sub(r'([.!?])\s+', r'\1\n', asr['text'])
            f.write(text)

        # Save the transcription to a text file
        with open(os.path.join(output_dir, txt_file_ts), 'w', encoding='utf-8') as f:
            for chunk in asr['chunks']:
                start = chunk['timestamp'][0]
                end = chunk['timestamp'][1]
                text = chunk['text']
                f.write(f"[{format_timestamp(start)} - {format_timestamp(end)}] {text.strip()}\n")

        elapsed_time = time.time() - start_time
        print(f"Transcription saved to {txt_file_ts} (took {elapsed_time:.2f} seconds)")

    except Exception as e:
        print(f"Error processing {wav_file}: {str(e)}")

print("All files processed")
