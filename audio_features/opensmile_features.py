import opensmile
from pathlib import Path

fe = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    num_workers=16
)

fe.process.num_workers = 16
fe.process.multiprocessing = True

print(fe.feature_names)

#features = fe.process_file('test.wav')

features = fe.process_folder('/media/trsuser/1E9AAD029AACD793/psyco/norm')

features = features.reset_index()

features['video_id'] = features['file'].apply(lambda x: Path(x).stem)

del features['file']
del features['start']
del features['end']

features = features.set_index('video_id')

features.to_csv('audio_features_normalized.csv', index=True)
features.to_parquet('audio_features_normalized.parquet')