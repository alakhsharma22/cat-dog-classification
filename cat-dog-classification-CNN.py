"""
CNNs are special types of neural networks used for processing data that has a known grid-like topology, such as the images(2-D).
These algorithms help to reduce the High computation costs for the model and also reduce the loss of important info such as Special arrangement
of pixels.

The image imput is passed through multiple layers of Convolution + ReLU (rectified linear unit => any -ve to 0 and positive remains same).
The task of these layers is to extract primitive features (the multiple edges in an image) from the image. As the tasks increase, the more
complex features are extracted from the image in the succeding layers.

Finally, the whole data is flattened to a 1-D array and is passed to a ANN for giving the output.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
import keras.utils.image_dataset
from keras.utils import load_img , img_to_array
from keras.utils import image_dataset_from_directory
import os
import matplotlib.image as mpimg 
from zipfile import ZipFile

data_path = 'Cats_Dogs_Images.zip'

with ZipFile(data_path, 'r') as zip:
    zip.extractall()
    print('The data set has been extracted.')

path = 'Cats_Dogs_Images' 
classes = os.listdir(path)
print(classes)

fig = plt.gcf()
fig.set_size_inches(16, 16)

cat_dir = os.path.join('Cats_Dogs_Images/Cats_Images')
dog_dir = os.path.join('Cats_Dogs_Images/Dogs_Images')
cat_names = os.listdir(cat_dir)
dog_names = os.listdir(dog_dir)

pic_index = 210

cat_images = [os.path.join(cat_dir, fname)
              for fname in cat_names[pic_index-8:pic_index]]
dog_images = [os.path.join(dog_dir, fname)
              for fname in dog_names[pic_index-8:pic_index]]

for i, img_path in enumerate(cat_images + dog_images):
    sp = plt.subplot(4, 4, i+1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

base_dir = 'Cats_Dogs_Images'

train_datagen = image_dataset_from_directory(base_dir,
                                                  image_size=(200,200),
                                                  subset='training',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= 32)
test_datagen = image_dataset_from_directory(base_dir,
                                                  image_size=(200,200),
                                                  subset='validation',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= 32)

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])
model.summary()
keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(train_datagen,
          epochs=10,
          validation_data=test_datagen)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

test_image = load_img('Cats_Dogs_Images/Dogs_Images/dog.10.jpg',target_size=(200,200))

plt.imshow(test_image)
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

result = model.predict(test_image)

if(result<=0.5):
  print("Dog")
else:
  print("Cat")

test_image2 = load_img('Cats_Dogs_Images/Cats_Images/cat.1009.jpg', target_size=(200, 200))
plt.imshow(test_image2)

test_image2 = img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)
result = model.predict(test_image2)

if(result >= 0.5):
    print("Dog")
else:
    print("Cat")