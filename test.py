import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

scaler = joblib.load('scaler.joblib')
columns = joblib.load('columns.joblib')

model = load_model('model_lstm.h5')

def preprocess_new_data(new_data):
    new_data_df = pd.DataFrame(new_data, columns=["age", "restingBP", "gender", "chestpain", "serumcholestrol", 
                                                  "fastingbloodsugar", "restingrelectro", "maxheartrate", 
                                                  "exerciseangia", "oldpeak", "slope", "noofmajorvessels"])

    new_data_df = pd.get_dummies(new_data_df, columns=["gender", "chestpain", "fastingbloodsugar", "restingrelectro", "exerciseangia", "slope", "noofmajorvessels"])
    
    for col in columns:
        if col not in new_data_df:
            new_data_df[col] = 0
    new_data_df = new_data_df[columns]
    new_data_df[["age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"]] = scaler.transform(new_data_df[["age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"]])
    return new_data_df

def predict_new_data(new_data):
    preprocessed_data = preprocess_new_data(new_data)
    
    preprocessed_data = preprocessed_data.values.reshape((preprocessed_data.shape[0], 1, preprocessed_data.shape[1]))
    new_pred = model.predict(preprocessed_data)
    new_pred = (new_pred > 0.5).astype(int)
    return new_pred

new_data = np.array([[57, 150, 1, 0, 229, 0, 1, 140, 0, 2.5, 2, 3]])
prediction = predict_new_data(new_data)
print("Prédiction:", prediction)
