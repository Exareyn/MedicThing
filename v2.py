import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Chargement du dataset
data = pd.read_csv('data/dataset.csv')
data.columns = ["patientid", "age", "gender", "chestpain", "restingBP", "serumcholestrol", 
                "fastingbloodsugar", "restingrelectro", "maxheartrate", "exerciseangia", 
                "oldpeak", "slope", "noofmajorvessels", "target"]

data.fillna(data.mean(), inplace=True)

scaler = StandardScaler()
data[['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']] = scaler.fit_transform(data[['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']])
data = pd.get_dummies(data, columns=["gender", "chestpain", "fastingbloodsugar", "restingrelectro", "exerciseangia", "slope", "noofmajorvessels"])

X = data.drop(columns=["patientid", "target"]).astype(np.float32)
y = data["target"].astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy*100:.2f}%')

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

print(y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
model.save('model_lstm2.keras')

joblib.dump(scaler, 'scaler.joblib')