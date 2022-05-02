import joblib
from flask import Flask, jsonify, request

from tools import parse_info

model = joblib.load('model.joblib')
district_model = joblib.load('district_model.joblib')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    file = request.json
    return jsonify(prediction=parse_info(file, district_model, model))


@app.route('/', methods=['GET'])
def index():
    return 'Crime App Machine Learning API'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

