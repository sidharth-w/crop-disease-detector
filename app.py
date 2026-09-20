from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

app = Flask(__name__)

model = load_model("crop_disease_model.h5")

class_names = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy'
]

treatments = {
    'Potato___Early_blight': 'Apply copper-based fungicide; remove affected leaves.',
    'Potato___Late_blight': 'Apply fungicide immediately; destroy infected plants to prevent spread.',
    'Potato___healthy': 'No disease detected. Plant looks healthy!',
    'Tomato___Early_blight': 'Remove lower infected leaves; apply fungicide, improve air circulation.',
    'Tomato___Late_blight': 'Urgent: remove and destroy infected plants; apply fungicide to nearby plants.',
    'Tomato___healthy': 'No disease detected. Plant looks healthy!'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = img.resize((224, 224))

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred) * 100)

    return jsonify({
        'disease': predicted_class,
        'confidence': round(confidence, 2),
        'treatment': treatments.get(predicted_class, 'No suggestion available.')
    })

if __name__ == '__main__':
    app.run(debug=True)