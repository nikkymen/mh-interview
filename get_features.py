import argparse
import pandas as pd
import tempfile

from pathlib import Path

# from video_features.openface_features import extract_vf_openface
# from video_features.tsfresh_features import extract_vf_tsfresh

from audio_features.extract_audio import extract_wav_from_video
from audio_features.opensmile_features import extract_af_opensmile
# from transcript.whisper_transcript import extract_transcription

# from text_features.llm_features import extract_tf_llm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("./features.parquet"))

    args = parser.parse_args()

    input_path: Path = args.input

    #video_tracks: pd.DataFrame = extract_vf_openface(args.input)
    #vf_tsfresh: pd.DataFrame = extract_vf_tsfresh(video_tracks, 'id')

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    wav_raw, wav_norm = extract_wav_from_video(input_path, temp_path)

    af_opensmile: pd.DataFrame = extract_af_opensmile(wav_raw)

    #transcription: str = extract_transcription(wav_norm)

    temp_dir.cleanup()

    # tf_llm_1: pd.DataFrame = extract_tf_llm(transcription, 'llama-3.3-70b-instruct')
    # tf_llm_2: pd.DataFrame = extract_tf_llm(transcription, 'gpt-oss-120b')
    # tf_llm_3: pd.DataFrame = extract_tf_llm(transcription, 'qwen-2.5-72b-instruct')

    # Concat all columns

    # features = pd.concat([tf_llm_1, tf_llm_2, tf_llm_3, af_opensmile, vf_tsfresh], axis=1)

    features = af_opensmile

    # Export

    features.to_parquet(args.output)

if __name__ == "__main__":
    main()