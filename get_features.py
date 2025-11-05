import argparse
import logging
import pandas as pd
import tempfile
import os
import psutil
import shutil
import platform
import sys
import traceback
import time
from contextlib import contextmanager

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from pathlib import Path
from datetime import datetime

# from video_features.openface_features import extract_vf_openface
# from video_features.tsfresh_features import extract_vf_tsfresh

from audio_features.extract_audio import extract_wav_from_video
from audio_features.opensmile_features import extract_af_opensmile
# from transcript.whisper_transcript import extract_transcription

# from text_features.llm_features import extract_tf_llm

logging.basicConfig(format='[%(asctime)s] %(name)-15.15s [%(levelname)-8.8s]  %(message)s')

logger = logging.getLogger('wellbeing')
logger.setLevel('INFO')

@contextmanager
def log_timing(task: str):
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error(f"{task} failed after {elapsed:.3f}s")
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(f"{task} completed in {elapsed:.3f}s")

def log_system_info():
    """Log detailed system information"""
    logger.info("="*50)
    logger.info("System Information:")
    logger.info("="*50)

    logger.info(f"Working dir: {os.path.dirname(os.path.realpath(__file__))}")

    # CPU Info
    logger.info(f"CPU Count (logical): {psutil.cpu_count(logical=True)}")
    logger.info(f"CPU Count (physical): {psutil.cpu_count(logical=False)}")
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        logger.info(f"CPU Frequency: Current={cpu_freq.current:.2f}MHz, Min={cpu_freq.min:.2f}MHz, Max={cpu_freq.max:.2f}MHz")
    logger.info(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")

    # RAM Info
    ram = psutil.virtual_memory()
    logger.info(f"RAM Total: {ram.total / (1024**3):.2f} GB")
    logger.info(f"RAM Available: {ram.available / (1024**3):.2f} GB")
    logger.info(f"RAM Used: {ram.used / (1024**3):.2f} GB ({ram.percent}%)")

    # GPU Info
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    logger.info(f"GPU {i}: {gpu.name}")
                    logger.info(f"  Memory Total: {gpu.memoryTotal} MB")
                    logger.info(f"  Memory Used: {gpu.memoryUsed} MB ({gpu.memoryUtil*100:.1f}%)")
                    logger.info(f"  GPU Load: {gpu.load*100:.1f}%")
            else:
                logger.info("No GPU detected")
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
    else:
        logger.info("GPUtil not installed, skipping GPU info")

    # Disk Info (current directory)
    disk = shutil.disk_usage(os.getcwd())
    logger.info(f"Disk Total: {disk.total / (1024**3):.2f} GB")
    logger.info(f"Disk Used: {disk.used / (1024**3):.2f} GB")
    logger.info(f"Disk Free: {disk.free / (1024**3):.2f} GB ({(disk.free/disk.total)*100:.1f}%)")

    # Platform Info
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info("="*50)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=False)
    parser.add_argument("--output", type=Path, default=Path("./"), required=False)

    args = parser.parse_args()

    env_output = os.getenv("UNIP_PIPELINE_OUTPUT")

    if env_output:
        output_path = Path(env_output)
    else:
        output_path = args.output

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = output_path / f"log-{timestamp}.txt"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(name)-15.15s [%(levelname)-8.8s]  %(message)s'))

    logger.addHandler(file_handler)

    logger.info(f'output_path: {str(output_path)}')

    log_system_info()

    env_input = os.getenv("UNIP_PIPELINE_INPUT")

    input_path = args.input or (Path(env_input) if env_input else None)
    if not input_path:
        logger.info('Входной файл не задан.')
        parser.error("Missing input. Provide --input or set UNIP_PIPELINE_INPUT")

    logger.info(f'input_path: {str(input_path)}')

    if not input_path.exists():
        logger.info(f'Файл не существует: {str(input_path)}.')
        return 1

    #video_tracks: pd.DataFrame = extract_vf_openface(args.input)
    #vf_tsfresh: pd.DataFrame = extract_vf_tsfresh(video_tracks, 'id')

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    with log_timing('Извлечение аудиодорожек'):
        wav_raw, wav_norm = extract_wav_from_video(input_path, temp_path)

    with log_timing('Извлечение аудиопризнаков'):
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

    logger.info('Экспорт признаков...')

    features.to_parquet(output_path / f"features-{timestamp}.parquet")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        logger.error("Полная трассировка ошибки:")
        logger.error(traceback.format_exc())
        sys.exit(1)