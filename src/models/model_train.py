#entrainement du modèle avec les meilleurs paramètres
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Chargement des données prétraitées
X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

# Chargement des meilleurs paramètres
best_params = joblib.load('models/best_params_RFReg.pkl')

# Définition du modèle avec les meilleurs paramètres
model = RandomForestRegressor(**best_params, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train.values.ravel())

#sauvegarde du modèle
joblib.dump(model, 'models/RandomForestRegressor_model.pkl')