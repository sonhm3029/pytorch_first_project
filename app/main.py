from flask import Flask, request, jsonify
import json
from app.torch_utils import transform_image, get_prediction
from app.helper import allowed_file

app = Flask(__name__)


@app.route('/')
def server():
    return "HEllo world"

@app.route('/predict', methods=['POST'])
def predict():
    if(request.method == 'POST'):
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({
                'code':400,
                'message':'no file'
            })
        if not allowed_file(file.filename):
            return jsonify({'error':'format not supported'})
        
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction':prediction.item(), 'class_name':str(prediction.item())}
            return jsonify({
                'code':200,
                'data':data
            })
        except:
            return jsonify({'error':'error during prediction'})
    # 1 load image
    # 2 image -> tensor
    # 3 prediction
    # 4 return json
    return jsonify({'result':1})


if __name__ == '__main__':
    app.run(debug=True)