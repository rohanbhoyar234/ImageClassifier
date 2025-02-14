# Image Classifier using Google's Teachable Machine

## Overview
This project is an Image Classifier built using Google's Teachable Machine. It allows users to classify images based on a trained model without requiring extensive coding knowledge.

## Features
- Easy-to-train model using Google's Teachable Machine
- Image classification in real-time
- Lightweight and efficient implementation

## Prerequisites
Before running the project, ensure you have the following:
- Python (if using a local implementation)
- Required Python libraries: `tensorflow`, `keras`, `numpy`, `opencv-python`, `PIL`

## Steps to Build the Image Classifier

### Step 1: Train the Model
1. Go to [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Select "Image Project."
3. Upload images for different classes.
4. Train the model using the platform.
5. Export the model in TensorFlow format.

### Step 2: Implement the Model
1. Download the exported model.
2. Place the model files in your project directory.
3. Install dependencies using:
   ```sh
   pip install tensorflow keras numpy opencv-python pillow
   ```
4. Load the model in Python:
   ```python
   import tensorflow as tf
   from tensorflow.keras.models import load_model
   import numpy as np
   from PIL import Image
   
   model = load_model('path_to_your_model')
   ```

### Step 3: Run Image Classification
1. Capture or load an image.
2. Preprocess the image (resize, normalize).
3. Make predictions:
   ```python
   def predict_image(image_path, model):
       img = Image.open(image_path).resize((224, 224))
       img_array = np.array(img) / 255.0
       img_array = np.expand_dims(img_array, axis=0)
       prediction = model.predict(img_array)
       return prediction
   ```
4. Display classification results.

## Usage
- Run the script and pass an image to classify.
- Modify the UI to integrate it with a web or desktop application.

## Future Enhancements
- Improve accuracy with more training data.
- Deploy as a web application.
- Integrate with a mobile application.

## Conclusion
This project demonstrates how to build an image classifier using Google's Teachable Machine with minimal coding effort. It is useful for quick prototyping and educational purposes.

