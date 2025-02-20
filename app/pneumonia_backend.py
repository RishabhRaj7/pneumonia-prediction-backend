from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "methods": ["POST", "OPTIONS"],
         "allow_headers": ["Content-Type"]
     }})

# Rest of your imports and setup code remains the same...

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # Handling preflight request
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response

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

        response = jsonify({
            "message": result,
            "confidence": f"{confidence:.2f}%"
        })
        return response, 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)