from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FileValidator:
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in FileValidator.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file(file):
        if file is None or file.filename == "":
            return False, "No file selected"
        if not FileValidator.allowed_file(file.filename):
            return False, "Invalid file format"
        return True, "File is valid"

class ImagePreprocessor:
    @staticmethod
    def preprocess_image(image_path):
        """Preprocess the image to match training preprocessing."""
        try:
            logger.debug(f"Processing image: {image_path}")
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.per_image_standardization(img)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (180, 180))
            img = np.expand_dims(img.numpy(), axis=0)
            logger.debug("Image preprocessing completed successfully")
            return img
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise

app = Flask(__name__)
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "methods": ["POST", "OPTIONS"],
         "allow_headers": ["Content-Type"]
     }})

# Ensure the uploads folder exists
UPLOAD_FOLDER = "app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
try:
    MODEL_PATH = os.path.abspath("model/my_model.h5")
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    try:
        if request.method == "OPTIONS":
            response = jsonify({"message": "OK"})
            response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            return response

        logger.info("Received prediction request")
        
        if model is None:
            logger.error("Model not loaded")
            return jsonify({"error": "Model not initialized"}), 500

        if "file" not in request.files:
            logger.warning("No file in request")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        is_valid, message = FileValidator.validate_file(file)
        
        if not is_valid:
            logger.warning(f"File validation failed: {message}")
            return jsonify({"error": message}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            logger.debug(f"Saving file to: {filepath}")
            file.save(filepath)

            # Process image
            img = ImagePreprocessor.preprocess_image(filepath)
            
            # Make prediction
            logger.debug("Making prediction")
            prediction = model.predict(img)[0]
            confidence = float(prediction[0]) * 100

            result = "Pneumonia Positive" if prediction[0] > 0.5 else "Normal"
            logger.info(f"Prediction complete: {result}")

            return jsonify({
                "message": result,
                "confidence": f"{confidence:.2f}%"
            }), 200

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": "Error processing image"}), 500
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug("Removed temporary file")

    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health")
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "upload_folder": os.path.exists(UPLOAD_FOLDER)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)