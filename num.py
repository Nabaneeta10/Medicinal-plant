import joblib
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('C:/Users/ASUS/Desktop/Dieases/Disease.joblib')  # Replace with the actual path

# Function to preprocess the image
def preprocess_image(image_path):
    # Open and resize the image
    img = Image.open(image_path).resize((256, 256))
    img = img.convert('RGB')

    # Convert the image to a NumPy array
    img_array = np.array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array

# Path to the image you want to test
image_path = r'C:\Users\ASUS\Desktop\pyth_env\healthy.JPG'  # Replace with the actual path

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Perform prediction
prediction = model.predict(preprocessed_image)

print(f'The predicted class is: {prediction}')
