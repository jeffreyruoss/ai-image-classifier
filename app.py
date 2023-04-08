from io import BytesIO
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = VGG16(weights='imagenet', include_top=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    img = request.files['image']
    img_bytes = BytesIO(img.read())  # Convert FileStorage to BytesIO
    img = load_img(img_bytes, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_name = decode_predictions(preds, top=1)[0][0][1]

    return jsonify({'class_name': class_name})

if __name__ == '__main__':
    app.run(debug=True)
