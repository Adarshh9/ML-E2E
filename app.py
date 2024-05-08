from flask import Flask, request ,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelne.predict_pipeline import CustomData ,PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' ,methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')  
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        dataframe = data.get_data_as_dataframe()
        print(dataframe)

        predict_pipeline = PredictPipeline()

        result = predict_pipeline.predict(dataframe)

        return render_template('home.html' ,result = result[0])

if __name__=="__main__":
    app.run(port='5000' ,debug=True)  