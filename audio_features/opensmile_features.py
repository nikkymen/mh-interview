import argparse
import opensmile
import pandas as pd

from pathlib import Path

feature_extractor = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    num_workers=16
)

feature_extractor.process.num_workers = 16
feature_extractor.process.multiprocessing = True

def extract_af_opensmile(input: Path) -> pd.DataFrame:

    features = feature_extractor.process_file(str(input))
    features = features.reset_index()

    del features['file']
    del features['start']
    del features['end']

    return features

def extract_af_opensmile_from_dir(input: Path) -> pd.DataFrame:

    features = feature_extractor.process_folder(str(input))
    features = features.reset_index()

    features['video_id'] = features['file'].apply(lambda x: Path(x).stem)

    del features['file']
    del features['start']
    del features['end']

    features = features.set_index('video_id')

    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/audio/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/features/af_opensmile.parquet"))

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    features: pd.DataFrame = extract_af_opensmile_from_dir(args.input)

    features.to_parquet(args.output)

    print(f'Saved features {features.shape}: {args.output}')

    print(features.columns)

    #o: Path = args.output
   # print(features.columns, file=str(o.with_suffix('.txt')))

if __name__ == "__main__":
    main()