import numpy as np
import csv
from sklearn.utils import Bunch

def load_spotify_dataset():
    with open(r"Spotify Songs' Genre Segmentation/spotify dataset.csv") as csv_file:
        data_reader = csv.reader(csv_file)
        feature_names = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            features = row[:-1]
            features = features[3:4] + features[11:]
            label = row[-1]
            data.append([float(num) for num in features])
            target.append(int(label))
        
        data = np.array(data)
        target = np.array(target)
    return Bunch(data=data, target=target, feature_names=feature_names)