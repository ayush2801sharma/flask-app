from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    humidity = request.form.get('humidity')
    temperature = request.form.get('temperature')
    gas = request.form.get('gas')
    heartrate = request.form.get('heartrate')
    spo2 = request.form.get('spo2')
    pm2 = request.form.get('pm2')
    pm10 = request.form.get('pm10')
    pefr = request.form.get('pefr')

    input_query = np.array([[humidity,temperature,gas,heartrate,spo2,pm2,pm10,pefr]])

    b = np.array(input_query, dtype=float)
    result = model.predict(b)[0]

    return jsonify({'result': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
