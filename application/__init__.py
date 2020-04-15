from flask import Flask, request, Response, json
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys

app = Flask(__name__)

# Load the model from pickle
regress_model = pickle.load(open('regress_model','rb'))

#create api
@app.route('/api/', methods=['GET', 'POST'])
@app.route('/api',methods=['GET', 'POST'])
def predict():
    # Get the data from POST request
    data = request.get_json(force=True)
    requestData = [data["fixed_acidity"], data["volatile_acidity"], data["citric_acid"], data["residual_sugar"],
    data["chlorides"], data["free_sulfur_dioxide"], data["total_sulfur_dioxide"], data["density"],
    data["pH"], data["sulphates"], data["alcohol"]]
       
    requestData = np.array([requestData]).astype(np.float64)
    requestData = np.ndarray.reshape(requestData, (11, 1) )

    # Make prediction using model 
    prediction = regress_model.predict(requestData)
    return Response(json.dumps(float(prediction[0])))

if __name__ == '__main__':
   app.run()
    