# Step 5 

# HUSSEIN MOHAMMAD - 501098569
# AER850 Section 01 Project 2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the path to the test images
test_images = {
    "crack": "./Data/Test/crack/test_crack.jpg",
    "missing_head": "./Data/Test/missing-head/test_missinghead.jpg",
    "paint_off": "./Data/Test/paint-off/test_paintoff.jpg"
}

# Load the trained model
test_model = tf.keras.models.load_model("model.keras")

# Define the class names based on training (adjust based on your class order in the model)
class_names = ["crack", "missing-head", "paint-off"]

# Function to preprocess and predict on a single image
def process_and_predict(image_path):
    # Load the image
    img = load_img(image_path, target_size=(500, 500))
    
    # Convert the image to an array and normalize
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = test_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class]
    confidence = predictions[0][predicted_class]

    return predicted_class_name, confidence, img

# Loop through each test image, process, predict, and display the results
for label, image_path in test_images.items():
    predicted_class_name, confidence, img = process_and_predict(image_path)
    
    # Display the image and the prediction
    plt.figure()
    plt.imshow(img)
    plt.title(f"Actual: {label.replace('_', ' ').title()}, Predicted: {predicted_class_name.title()} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()
