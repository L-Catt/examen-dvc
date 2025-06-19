#evaluation du modèle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# Chargement des données prétraitées
X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
# Chargement du modèle entraîné
model = joblib.load('models/RandomForestRegressor_model.pkl')
# Prédictions sur les données de test
y_pred = model.predict(X_test)
# Calcul des métriques d'évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Affichage des résultats
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
# Sauvegarde des résultats dans un fichier json
results = {
    'Mean Squared Error': mse,
    'R^2 Score': r2
}
with open('metrics/evaluation_results.json', 'w') as f:
    import json         
    json.dump(results, f, indent=4)
# Sauvegarde des prédictions dans un fichier csv
predictions_df = pd.DataFrame({'Actual': y_test.values.ravel(), 'Predicted': y_pred})
predictions_df.to_csv('data/processed_data/predictions.csv', index=False)
