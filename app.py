import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = 'my_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
input_shape = model.input_shape 
img_height, img_width = input_shape[1], input_shape[2]

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image:
      - Convert to RGB (if needed).
      - Resize to the expected (img_width, img_height) dimensions.
      - Convert to a numpy array and normalize pixel values (0-1).
      - Expand dimensions to add a batch dimension.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((img_width, img_height))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part in the request.", 400
        
        file = request.files['file']
        if file.filename == "":
            return "No file selected.", 400
        
        try:
            image = Image.open(file.stream)
        except Exception as e:
            return f"Error opening image: {e}", 400

        input_data = preprocess_image(image)
        output_data = model.predict(input_data)[0]

        cat_prob, dog_prob = output_data
        if cat_prob > dog_prob:
            predicted_label = "Cat"
            confidence = cat_prob * 100
        else:
            predicted_label = "Dog"
            confidence = dog_prob * 100

        confidence = round(confidence, 2)
        return render_template("result.html", label=predicted_label, accuracy=confidence)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)