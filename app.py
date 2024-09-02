from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import base64
from src.Ship_Classifier.pipeline.predict import PredictionPipeline

app = Flask(__name__)
CORS(app)

# Initialize the ClientApp
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"  # Default filename
        self.classifier = None

clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # Check if an image was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    # If the user does not select a file, the browser may also submit an empty part without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Initialize the prediction pipeline with the saved image file
        clApp.classifier = PredictionPipeline(filepath)

        # Perform prediction
        result = clApp.classifier.predict()

        return jsonify(result)

if __name__ == "__main__":
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(host='0.0.0.0', port=8080)
