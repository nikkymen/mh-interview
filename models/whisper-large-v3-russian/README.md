---
language:
- ru
library_name: transformers
tags:
- asr
- whisper
- russian
datasets:
- mozilla-foundation/common_voice_17_0
metrics:
- wer
---

# Model Details

This is a version of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) finetuned for better support of Russian language.

Dataset used for finetuning is Common Voice 17.0, Russian part, that contains over 200k rows.

After preprocessing of the original dataset (all splits were mixed and splited to a new train + test split by 0.95/0.05, 
that is 225761/11883 rows respectively) the original Whisper v3 has WER 9.84 while the finetuned version shows 6.39 (so far).

The finetuning process took over 60 hours on dual Tesla A100 80Gb.

## Usage

In order to process phone calls it is highly recommended that you preprocess your records and adjust volume before performing ASR. For example, like this:

```bash
sox record.wav -r 16k record-normalized.wav norm -0.5 compand 0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2
```

Then your ASR code should look somewhat like this:

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

torch_dtype = torch.bfloat16 # set your preferred type here 

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
    setattr(torch.distributed, "is_initialized", lambda : False) # monkey patching
device = torch.device(device)

whisper = WhisperForConditionalGeneration.from_pretrained(
    "antony66/whisper-large-v3-russian", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    # add attn_implementation="flash_attention_2" if your GPU supports it
)

processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=whisper,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# read your wav file into variable wav. For example:
from io import BufferIO
wav = BytesIO()
with open('record-normalized.wav', 'rb') as f:
    wav.write(f.read())
wav.seek(0)

# get the transcription
asr = asr_pipeline(wav, generate_kwargs={"language": "russian", "max_new_tokens": 256}, return_timestamps=False)

print(asr['text'])

```

## Work in progress

This model is in WIP state for now. The goal is to finetune it for speech recognition of phone calls as much as possible. If you want to contribute and you know or have any good dataset please let me know. Your help will be much appreciated.