import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model_path = r"C:\Users\Lanovo\mango_disease_model_new.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please verify the path.")
    
model = tf.keras.models.load_model(model_path)

class_names = ['Anthracnose', 'Bacterial Canker', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew']

def preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at {img_path}. Please verify the path.")
    
    img = image.load_img(img_path, target_size=(150, 150)) 
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

def predict_disease(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]
    confidence = prediction[0][predicted_class] * 100
    return predicted_label, confidence

img_path = r"C:\Users\Lanovo\OneDrive\Desktop\Mango\try4.jpg"

try:
    predicted_label, confidence = predict_disease(img_path)
    print("\nDetection Successful!")
    print(f"Predicted Class: {predicted_label}")
except Exception as e:
    print(f"An error occurred: {e}")
