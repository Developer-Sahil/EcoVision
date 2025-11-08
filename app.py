from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'waste_classifier_model.h5'
IMG_SIZE = 96  # Match training image size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class mapping (from training)
# 0 = O (Organic/Biodegradable), 1 = R (Recyclable/Non-biodegradable)
CLASS_NAMES = {
    0: "Biodegradable (Organic)",
    1: "Non-Biodegradable (Recyclable)"
}

# Eco-impact data
ECO_IMPACT = {
    "Biodegradable (Organic)": {
        "score": 2,
        "tip": "Compost organic waste to enrich soil and reduce landfill waste!",
        "decompose_time": "2-6 months"
    },
    "Non-Biodegradable (Recyclable)": {
        "score": 8,
        "tip": "Recycle this item to prevent it from polluting the environment for centuries.",
        "decompose_time": "100-1000 years"
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    # Read image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            img = preprocess_image(filepath)
            prediction = model.predict(img)
            
            # Get class (0 or 1)
            predicted_class = int(prediction[0][0] > 0.5)
            confidence = float(prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0])
            
            # Get class name and eco-impact
            class_name = CLASS_NAMES[predicted_class]
            eco_data = ECO_IMPACT[class_name]
            
            result = {
                'classification': class_name,
                'confidence': f"{confidence * 100:.2f}%",
                'eco_score': f"{eco_data['score']}/10",
                'tip': eco_data['tip'],
                'decompose_time': eco_data['decompose_time']
            }
            
            # Clean up uploaded file (optional)
            # os.remove(filepath)
            
            return jsonify(result), 200
        
        else:
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)