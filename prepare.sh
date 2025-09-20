#!/bin/bash

# Проверка на наличие аргумента
if [ -z "$1" ]; then
  echo "Использование: $0 <имя_видео_файла>"
  exit 1
fi

# Входной видеофайл
INPUT="$1"

# Получение имени файла без расширения
BASENAME=$(basename "$INPUT")
NAME="${BASENAME%.*}"

# Промежуточный и итоговый файлы
RAW_WAV="${NAME}.wav"
NORM_WAV="${NAME}.norm.wav"

# Извлечение аудио из видео
ffmpeg -i "$INPUT" -vn -acodec pcm_s16le -ar 44100 -ac 2 "$RAW_WAV"

# Нормализация и компрессия
sox "$RAW_WAV" -r 16k "$NORM_WAV" norm -0.5 compand 0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2

echo "Аудио сохранено в: $NORM_WAV"
