#GridSearch des meilleurs paramètres à utiliser pour la modélisation
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib


# Chargement des données prétraitées
X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
# Définition du modèle
model = RandomForestRegressor(random_state=42)
# Définition des paramètres à tester
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Initialisation de GridSearchCV
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)
# Exécution de la recherche des meilleurs paramètres
grid_search.fit(X_train, y_train.values.ravel())
# Affichage des meilleurs paramètres
print("Meilleurs paramètres trouvés :")
print(grid_search.best_params_)
# Sauvegarde des meilleurs paramètres dans un fichier .pkl

joblib.dump(grid_search.best_params_, 'models/best_params_RFReg.pkl')
