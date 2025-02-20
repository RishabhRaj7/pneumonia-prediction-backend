from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from werkzeug.utils import secure_filename
import gc  # Garbage collector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FileValidator:
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    
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
        # Check file size
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > FileValidator.MAX_FILE_SIZE:
            return False, "File size too large (max 5MB)"
        return True, "File is valid"

class ImagePreprocessor:
    TARGET_SIZE = (180, 180)
    
    @staticmethod
    def preprocess_image(image_path):
        """Preprocess the image with memory optimization."""
        try:
            logger.debug(f"Processing image: {image_path}")
            
            # Read image with OpenCV instead of TensorFlow
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, ImagePreprocessor.TARGET_SIZE)
            
            # Convert to float32 and normalize
            img = img.astype(np.float32) / 255.0
            
            # Standardize
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / (std + 1e-7)
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Force garbage collection
            gc.collect()
            
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

# Load the model with memory optimization
def load_model():
    try:
        MODEL_PATH = os.path.abspath("model/my_model.h5")
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        # Configure TensorFlow to be memory-efficient
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load model with memory optimization
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False  # Don't load optimizer state
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

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
            with tf.device('/CPU:0'):  # Force CPU usage
                prediction = model.predict(img, batch_size=1)[0]
            confidence = float(prediction[0]) * 100

            result = "Pneumonia Positive" if prediction[0] > 0.5 else "Normal"
            logger.info(f"Prediction complete: {result}")

            # Force garbage collection
            gc.collect()

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
            gc.collect()  # Force garbage collection

    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health")
def health_check():
    """Simple health check endpoint"""
    memory_info = ""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        memory_info = "unavailable"
        
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "upload_folder": os.path.exists(UPLOAD_FOLDER),
        "memory_usage_mb": memory_info
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)