import cv2
import numpy as np
from keras.models import load_model
from features_extractor import getFeatures
from skimage import measure

# Load the trained model
model = load_model('weights_results_5out/mlp_model.h5')

# Class labels
class_labels = ['damaged_grain', 'foreign', 'grain', 'broken_grain', 'grain_cover']
counts = {label: 0 for label in class_labels}

# Ask user for image
image_name = input("Please enter image name with extension: ")
image = cv2.imread(image_name)

if image is None:
    print("❌ Error: Image not found. Please check the file name or path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Label connected components
labels = measure.label(thresh, connectivity=2)
props = measure.regionprops(labels)

# Loop over each detected region
for prop in props:
    minr, minc, maxr, maxc = prop.bbox
    grain_img = image[minr:maxr, minc:maxc]

    # Ignore very small noise
    if grain_img.shape[0] > 20 and grain_img.shape[1] > 20:
        features = getFeatures(grain_img)
        if features is not None:
            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)
            predicted_class = class_labels[np.argmax(prediction)]
            counts[predicted_class] += 1

# Print results
print("\n✅ Classification Summary:")
for label in class_labels:
    print(f"{label}: {counts[label]}")
