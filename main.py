import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Charger les données
data = pd.read_csv('data/dataset.csv')

# Retirer les colonnes non pertinentes

# !!!! verifier les features et ajouter que les plus importantes
data = data.drop(['patientid'], axis=1)

# Séparer les features (X) et le label (y)
X = data.drop('target', axis=1).values
y = data['target'].values

# Normaliser les données
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Définir la longueur des séquences
sequence_length = 10

# Fonction pour créer les séquences
def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Définir le modèle LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, X_seq.shape[2])))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Évaluer le modèle
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Faire des prédictions
predictions = model.predict(X_val)
predictions = (predictions > 0.5).astype(int)

print(predictions)
model.save('model_lstm.keras')
