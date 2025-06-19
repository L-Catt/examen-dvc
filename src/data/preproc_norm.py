import pandas as pd
from sklearn.preprocessing import StandardScaler

# Chargement des données prétraitées
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')


# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde des données normalisées
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed_data/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed_data/X_test_scaled.csv', index=False)
