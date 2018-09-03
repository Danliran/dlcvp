from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarrarypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplypreprocessor import SimplePreprocessor
from datasets.simplydatasetloader import SimpleDatasetLoader
from nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import  numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
print(trainY)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
print(trainY)

print("[INFO] compiling model....")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=5)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)


print("[INFO] evaluating network...")
prediction = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), prediction.argmax(axis=1), target_names=["daisy", "dandelion", "roses"
                                                                                           , "sunflowers", "tulips"]))