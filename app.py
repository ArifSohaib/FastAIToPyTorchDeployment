from flask import Flask, jsonify, request, render_template, redirect
from create_model import get_model
from prepare_and_process_image import transform_image
from torchvision import models

import json
import os


app = Flask(__name__)
# model = models.densenet121(pretrained=True)
# model.eval()
model = get_model() 

class_index = json.load(open('./static/traffic_class.json'))
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_index = str(y_hat.item())
    return class_index[predicted_index]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files["file"]
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        print(f"IN FUNCTION PREDICT {class_name}")
        return jsonify({"class_id":class_id,"class_name":class_name})

@app.route("/", methods=["GET","POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files.get("file")
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return render_template("results.html", class_id=class_id, class_name=class_name)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT",5000)))