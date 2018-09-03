from datasets.simplydatasetloader import SimpleDatasetLoader
from preprocessing.simplypreprocessor import SimplePreprocessor
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")

imagepath = list(paths.list_images("/home/xiong/PycharmProjects/tf_learning/flower_photos"))
print(imagepath)
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
data, labels = sdl.load(imagepath, verbose=500)
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000)))