import pandas as pd
from sklearn.model_selection import train_test_split

#import de raw.csv
raw_data = pd.read_csv('data/raw_data/raw.csv')

#séparation de la variable cible et des variables explicatives
X = raw_data.drop(columns=['silica_concentrate'])
y = raw_data['silica_concentrate']

#séparation en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
#sauvegarde des données
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

