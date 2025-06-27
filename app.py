from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model (make sure the path is correct)
model = load_model('ecg_cnn_model.h5')

def prepare_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = prepare_image(file_path)
        
        prediction = model.predict(img)[0][0]

        result = "Abnormal" if prediction > 0.5 else "Normal"

        # Clean up the uploaded file
        os.remove(file_path)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

    
