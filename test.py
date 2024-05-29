import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
model = load_model('model_lstm.h5')

# Charger les nouvelles données
new_data = pd.read_csv('data/dataset.csv')

# Retirer les colonnes non pertinentes
new_data = new_data.drop(['patientid'], axis=1)

# Séparer les features (X)
X_new = new_data.values

# Normaliser les nouvelles données
scaler = MinMaxScaler(feature_range=(0, 1))
X_new_scaled = scaler.fit_transform(X_new)  # Assurez-vous d'utiliser le même scaler que celui utilisé pour les données de formation

# Définir la longueur des séquences
sequence_length = 10

# Fonction pour créer les séquences
def create_sequences(X, sequence_length):
    X_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
    return np.array(X_seq)

# Créer des séquences à partir des nouvelles données
X_new_seq = create_sequences(X_new_scaled, sequence_length)

# Faire des prédictions
predictions = model.predict(X_new_seq)
predictions = (predictions > 0.5).astype(int)  # Convertir les probabilités en classes binaires

# Afficher les prédictions
print(predictions)