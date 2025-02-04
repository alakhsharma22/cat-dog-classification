# Cat vs Dog Classification

This project is a simple web-based application that allows users to upload an image and classify it as either a cat or a dog using a trained deep learning model.

## Features
- Upload an image via a web interface.
- Predict whether the image is a cat or a dog.
- Display the prediction result with confidence percentage.
- User-friendly interface with HTML & CSS.

## Project Structure

├── app.py # Flask application to handle requests 

├── train.py # Script for training the deep learning model

├── templates/ # Folder for HTML templates 

             ├── index.html # Frontend upload page 
             
             ├── result.html # Prediction result display page 
             
## Technologies Used
- Flask (Backend)
- TensorFlow/Keras (Model Training)
- HTML/CSS (Frontend)
## Deep learning Model Architecture 
The classification model is built using a **Convolutional Neural Network (CNN)** with TensorFlow/Keras.
The model follows a sequential architecture consisting of convolutional layers, batch normalization, max pooling, dropout, and dense layers.

| Layer (type)                   | Output Shape          | Param #      |
|--------------------------------|----------------------|-------------|
| **conv2d (Conv2D)**             | (None, 126, 126, 32) | 896         |
| **batch_normalization (BatchNormalization)** | (None, 126, 126, 32) | 128 |
| **max_pooling2d (MaxPooling2D)** | (None, 63, 63, 32)  | 0           |
| **dropout (Dropout)**           | (None, 63, 63, 32)  | 0           |
| **conv2d_1 (Conv2D)**           | (None, 61, 61, 64)  | 18,496      |
| **batch_normalization_1 (BatchNormalization)** | (None, 61, 61, 64) | 256 |
| **max_pooling2d_1 (MaxPooling2D)** | (None, 30, 30, 64)  | 0           |
| **dropout_1 (Dropout)**         | (None, 30, 30, 64)  | 0           |
| **conv2d_2 (Conv2D)**           | (None, 28, 28, 128) | 73,856      |
| **batch_normalization_2 (BatchNormalization)** | (None, 28, 28, 128) | 512 |
| **max_pooling2d_2 (MaxPooling2D)** | (None, 14, 14, 128) | 0           |
| **dropout_2 (Dropout)**         | (None, 14, 14, 128) | 0           |
| **conv2d_3 (Conv2D)**           | (None, 12, 12, 256) | 295,168     |
| **batch_normalization_3 (BatchNormalization)** | (None, 12, 12, 256) | 1,024 |
| **max_pooling2d_3 (MaxPooling2D)** | (None, 6, 6, 256)  | 0           |
| **dropout_3 (Dropout)**         | (None, 6, 6, 256)  | 0           |
| **flatten (Flatten)**           | (None, 9216)       | 0           |
| **dense (Dense)**               | (None, 512)       | 4,719,104   |
| **batch_normalization_4 (BatchNormalization)** | (None, 512) | 2,048 |
| **dropout_4 (Dropout)**         | (None, 512)       | 0           |
| **dense_1 (Dense)**             | (None, 2)         | 1,026       |

### Total Parameters:
- **Total params:** 5,112,514 (≈19.50 MB)
- **Trainable params:** 5,110,530
- **Non-trainable params:** 1,984

