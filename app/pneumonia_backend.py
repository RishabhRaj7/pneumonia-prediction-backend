from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Explicitly allow CORS for all domains
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

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

    if file:
        return jsonify({"message": "Prediction logic goes here"}), 200

    return jsonify({"error": "Something went wrong"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure Render's dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
