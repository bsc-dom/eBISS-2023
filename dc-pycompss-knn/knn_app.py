import numpy as np
import sklearn.datasets

from PIL import Image
from PIL.ImageOps import invert

from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.parameter import COLLECTION_IN, FILE_IN
from pycompss.api.task import task
import dislib as ds

from model.knn import DCKNNClassifier

def train():
    data, target = sklearn.datasets.load_digits(return_X_y=True)
    data = ds.array(data, block_size=(400, 64))

    knn = DCKNNClassifier()
    knn.fit(data, target)

    return knn

@task(image_file=FILE_IN)
def prepare_image(image_file, returns=1):
    image = Image.open(image_file).convert("L")
    image = invert(image).resize((8, 8))

    # Grayscale typically goes between 0-255, with 8 bits.
    # But the dataset has 16 levels only, so we match the images
    # to the 0-16 range of the dataset.
    return np.array(image).flatten() // 16


@task(image_list=COLLECTION_IN, returns=1)
def stack_images(image_list):
    return np.vstack(image_list)


if __name__ == "__main__":
    print("Starting train...")
    knn = train()
    compss_barrier()
    print("Train has finished")

    print("Transforming images")
    images = list()
    for i in range(13):
        images.append([prepare_image("../custom-data/digit%02d.png" % i)])

    labels = knn.predict(stack_images(images))

    print("")
    print("********************************************************")
    print("Final labels (result of prediction with kNN classifier):")
    print(compss_wait_on(labels))
    print("********************************************************")
    print("")
