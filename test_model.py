# python .\test_model.py -d .\dataset\ -m .\attempt1.model\

# import the necessary packages
from utilities import ImageToArrayPreprocessor
from utilities import AspectAwarePreprocessor
from utilities import SimpleDatasetLoader
from tensorflow.python.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the arguments and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# initialise the class labels
classLabels = ["Coast", "Forest", "Highway", "Insidecity", "Mountain",
               "Office", "OpenCountry", "Street", "Suburb", "TallBuilding",
               "bedroom", "industrial", "kitchen", "livingroom", "store"]

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(50,))
imagePaths = imagePaths[idxs]

# initialise the image preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
predictions = model.predict(data, batch_size=64).argmax(axis=1)

# loop over the sample images
for (i, imagePaths) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePaths)
    cv2.putText(image, "Label: {}".format(classLabels[predictions[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)