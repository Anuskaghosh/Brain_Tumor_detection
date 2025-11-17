from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the model
MODEL_PATH = 'brain_tumor_model.h5'
model = None

def load_model():
    global model
    if model is None:
        try:
            model = keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return model

def preprocess_image(image):
    image = image.resize((150, 150))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        model = load_model()
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])

        # TRUE ORDER USED IN TRAINING
        classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

        result_class = classes[class_idx]

        # no_tumor is index 2
        has_tumor = class_idx != 2

        result = {
            'has_tumor': bool(has_tumor),
            'class': result_class,
            'confidence': round(confidence * 100, 2),
            'all_predictions': {
                classes[i]: round(float(prediction[0][i]) * 100, 2)
                for i in range(len(classes))
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
