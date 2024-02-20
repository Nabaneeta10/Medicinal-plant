

import os
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import StandardScaler
# loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits)

# Load the trained model
model = joblib.load('C:/Users/ASUS/Desktop/Dieases/Disease.joblib')  # Replace with the actual path

# Function to preprocess the image
# image_path='C:\Users\ASUS\Desktop\Dieases\healthy.jpg'
# image_path = 'C:\\Users\\ASUS\\Desktop\\Dieases\\healthy.JPG'
image_path = r'C:\Users\ASUS\Desktop\pyth_env\healthy.JPG'

def preprocess_image(image_path):
    # Open and resize the image
    img = Image.open(image_path).resize((224, 224))
    # Convert the image to a NumPy array
    img_array = np.array(img)
    # Flatten the image array
    flattened_img = img_array.flatten().reshape(1, -1)
    # Standardize the features
    scaler = StandardScaler()
    standardized_img = scaler.fit_transform(flattened_img)
    return standardized_img

# Path to the image you want to test
image_path1 = 'healthy.JPG'  # Replace with the actual path
# print(f'Absolute path to image: {os.path.abspath(image_path1)}')

# Preprocess the image
preprocessed_image = preprocess_image(image_path1)

# Perform prediction
prediction = model.predict(preprocessed_image)

print(f'The predicted class is: {prediction}')

