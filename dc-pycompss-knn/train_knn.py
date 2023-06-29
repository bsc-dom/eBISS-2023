import sklearn.datasets

import dislib as ds

from model.knn import DCKNNClassifier

def train():
    data, target = sklearn.datasets.load_digits(return_X_y=True)
    data = ds.array(data, block_size=(400, 64))

    knn = DCKNNClassifier()
    knn.fit(data, target)


if __name__ == "__main__":
    train()
