import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np

import warnings 
warnings.filterwarnings('ignore')

# Create the app
app = Flask(__name__)
# Load the model
model = pickle.load(open("lr.pkl", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    Make predictions from a json file with the postman API
    '''
    
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))

    # Perform the standardization of the data
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))

    # Perform the prediction
    prediction = model.predict(scaled_data)
    print(prediction[0])

    return jsonify(prediction[0])

@app.route('/predict', methods=['POST'])
def predict():
    '''

    '''

    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    prediction = model.predict(final_input)[0]

    return render_template('home.html', prediction_text='The predicted price of the house is: $ {}'.format(prediction))




if __name__=="__main__":
    app.run(debug=True)


