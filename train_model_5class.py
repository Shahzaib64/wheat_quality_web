import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle

# Define your labels
labels = ['damaged_grain', 'foreign', 'grain', 'broken_grain', 'grain_cover']

# Path to the folder containing the image(s) for testing
test_image_path = r"C:\Users\Mr Zaibi\Desktop\Wheat-quality-detector-2-master\Wheat-quality-detector-2-master\test_images\test.jpg"
# Path to the saved model
model_path = r"C:\Users\Mr Zaibi\Desktop\Wheat-quality-detector-2-master\Wheat-quality-detector-2-master\weights_results_5out\mlp_model.h5"

# Load the pre-trained model
model = load_model(model_path)


# Function to extract features from image (example: use OpenCV)
def extract_features(image_path):
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)

        # Check if image is loaded correctly
        if img is None:
            print(f"Error: Unable to load image from {image_path}")
            return None

        # Preprocess the image (resize and convert to RGB)
        img = cv2.resize(img, (224, 224))  # Adjust the size based on your model's expected input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (if necessary)

        # Convert the image to an array and scale pixel values
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

        # Add an extra dimension for batch size (required by Keras)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# Function to predict the class and count occurrences of each class in the image
def predict_class(image_path):
    features = extract_features(image_path)

    if features is None:
        return None

    # Predict the class of the image
    predictions = model.predict(features)
    predicted_class = labels[np.argmax(predictions)]

    return predicted_class


# Function to process the image and count occurrences of each class
def process_test_image(test_image_path):
    class_counts = {label: 0 for label in labels}

    # Get the predicted class for the image
    predicted_class = predict_class(test_image_path)

    if predicted_class is not None:
        class_counts[predicted_class] += 1

    return class_counts


# Get the counts of each class for the test image
class_counts = process_test_image(test_image_path)

# Print the final counts for each class
print("\nClass Counts for Test Image:")
for label, count in class_counts.items():
    print(f"{label}: {count}")
