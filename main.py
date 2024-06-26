# train_model.py
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

# Gestion des valeurs manquantes
data.fillna(data.mean(), inplace=True)

# Séparation des caractéristiques et de la cible avant l'encodage
features = data.drop(columns=["patientid", "target"])
target = data["target"]

# Encodage des variables catégorielles
features = pd.get_dummies(features, columns=["gender", "chestpain", "fastingbloodsugar", "restingrelectro", "exerciseangia", "slope", "noofmajorvessels"])

# Normalisation des données numériques
scaler = StandardScaler()
features[["age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"]] = scaler.fit_transform(features[["age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"]])

# Sauvegarde du scaler pour le prétraitement des nouvelles données
joblib.dump(scaler, 'scaler.joblib')

# Sauvegarde des colonnes pour les futures transformations
joblib.dump(features.columns, 'columns.joblib')

# Séparation des features et de la target
X = features.astype(np.float32)  # Forcer les types de données en float32
y = target.astype(np.float32)  # Forcer le type de données en float32

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape des données pour LSTM (samples, timesteps, features)
X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Définition de l'architecture du modèle
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Sauvegarde du modèle
model.save('model_lstm.keras')

# Évaluation sur les données de test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy*100:.2f}%')

# Prédiction sur de nouvelles données
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Évaluation des performances du modèle
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
