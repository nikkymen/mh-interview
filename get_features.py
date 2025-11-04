import argparse
import logging
import pandas as pd
import tempfile
import os

from pathlib import Path

# from video_features.openface_features import extract_vf_openface
# from video_features.tsfresh_features import extract_vf_tsfresh

from audio_features.extract_audio import extract_wav_from_video
from audio_features.opensmile_features import extract_af_opensmile
# from transcript.whisper_transcript import extract_transcription

# from text_features.llm_features import extract_tf_llm

logging.basicConfig(format='[%(asctime)s] %(name)-15.15s [%(levelname)-8.8s]  %(message)s')
logger = logging.getLogger('wellbeing')
logger.setLevel('INFO')
logger.info(f'Initialized logger {logger}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=False)
    parser.add_argument("--output", type=Path, default=Path("./features.parquet"), required=False)

    args = parser.parse_args()

    env_input = os.getenv("UNIP_PIPELINE_INPUT")

    input_path = args.input or (Path(env_input) if env_input else None)
    if not input_path:
        logger.info('Входной файл не задан.')
        parser.error("Missing input. Provide --input or set UNIP_PIPELINE_INPUT")

    if not input_path.exists():
        logger.info(f'Файл не существует: {input_path}.')
        return 1

    env_output = os.getenv("UNIP_PIPELINE_OUTPUT")

    if env_output:
        output_path = Path(env_output) / "features.parquet"
    else:
        output_path = args.output

    #video_tracks: pd.DataFrame = extract_vf_openface(args.input)
    #vf_tsfresh: pd.DataFrame = extract_vf_tsfresh(video_tracks, 'id')

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    logger.info('Извлечение аудиодорожек.')

    wav_raw, wav_norm = extract_wav_from_video(input_path, temp_path)

    logger.info('Извлечение аудиопризнаков.')

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

    logger.info('Экспорт.')

    features.to_parquet(output_path)

if __name__ == "__main__":
    main()