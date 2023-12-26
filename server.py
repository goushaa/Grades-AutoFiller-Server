from flask import Flask, jsonify, request, redirect, url_for
import cv2
from numpy import asarray
from PIL import Image
import os
from module1.main import process1
from module2.bubbleSheetMain import process2
from flask_cors import CORS  # Import CORS from flask_cors


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes

@app.route('/module1', methods=['POST'])
def module1():
    if 'image' in request.files:
        image = request.files['image']
        send_checkbox_value = request.form.get('sendCheckbox')
        alreadyMadeOCR = send_checkbox_value == 'on'
        image.save('./flask-server/module1/uploaded_image.jpg')
        image = cv2.imread('./flask-server/module1/uploaded_image.jpg')
        process1(image,alreadyMadeOCR)
        return redirect('http://localhost:3000/success')
    else:
        return 'No image uploaded.'

@app.route('/module2', methods=['POST'])
def module2():
    if 'image' in request.files:
        image = request.files['image']
        text_file = request.files['textFile']
        image.save('./flask-server/module2/uploaded_image.jpg')
        image = cv2.imread('./flask-server/module2/uploaded_image.jpg')
        process2(image,text_file)
        return redirect('http://localhost:3000/success')
    else:
        return 'No image uploaded.'            

if __name__ == "__main__":
    app.run(debug=True)
