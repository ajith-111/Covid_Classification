import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
from django.shortcuts import render
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('covid_model_LR.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    return jsonify(output[0])


@app.route('/predict_api',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    output = regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text = "The predictyion is {}".format(output))

if __name__ == "__main__":
    app.run(debug = True)    

