from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load the trained model
MODEL_PATH = os.path.abspath("app/model/my_model.h5")
print(f"🔍 MODEL_PATH: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure the uploads folder exists
UPLOAD_FOLDER = "app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image to match training preprocessing."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decode image
    img = tf.image.per_image_standardization(img)  # Standardization
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize
    img = tf.image.resize(img, (180, 180))  # Resize to match model input
    img = np.expand_dims(img.numpy(), axis=0)  # Convert to NumPy and add batch dimension
    return img

@app.route("/")
def home():
    return "Pneumonia Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process image
        img = preprocess_image(filepath)

        # Make prediction
        prediction = model.predict(img)[0]
        confidence = float(prediction[0]) * 100  # Convert to percentage

        # Convert model output to readable result
        result = "Pneumonia Positive" if prediction[0] > 0.5 else "Normal"

        return jsonify({
            "prediction": result,
            "confidence": f"{confidence:.2f}%"
        }), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure Render's dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
