from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = Load_model('cnn.model.h5')
def prepare_image(image, target_size):
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    image = Image.open(file)
    
    processed_image = prepare_image(image, target_size=(64, 64))   
    prediction = model.predict(processed_image).tolist()
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)