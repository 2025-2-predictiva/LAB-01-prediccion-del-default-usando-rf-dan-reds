# Importamos librerías necesarias
import os
import pandas as pd
import gzip
import json
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score

# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".


def load_and_process_data():
    train_path = '../files/input/train_data.csv.zip'
    test_path = '../files/input/test_data.csv.zip'
    
    df_train = pd.read_csv(train_path, index_col=False, compression='zip')
    df_test = pd.read_csv(test_path, index_col=False, compression='zip')
    
    # Re nombranndo y removiendo columnas no necesarias
    df_train.rename(columns={'default payment next month': 'default'}, inplace=True)
    df_test.rename(columns={'default payment next month': 'default'}, inplace=True)
    df_train.drop(columns=['ID'], inplace=True)
    df_test.drop(columns=['ID'], inplace=True)

    # removiendo registros con informacion no disponible. Ceros en MARRIAGE y EDUCATION
    df_train = df_train.loc[df_train['EDUCATION'] != 0]
    df_train = df_train.loc[df_train['MARRIAGE'] != 0]
    df_test = df_test.loc[df_test['EDUCATION'] != 0]
    df_test = df_test.loc[df_test['MARRIAGE'] != 0]

    df_train['EDUCATION'] = df_train['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    df_test['EDUCATION'] = df_test['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    return df_train, df_test

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

# Como ya está partido, solo hay que separar features y target
def split_features_target(df_train, df_test):
    x_train = df_train.drop(columns=['default'])
    y_train = df_train['default']
    x_test = df_test.drop(columns=['default'])
    y_test = df_test['default']
    
    return x_train, y_train, x_test, y_test

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#

def create_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), categorical_features),
            ],
            remainder='passthrough'
    )

    pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ],
    )

    return pipeline

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#

def make_grid_search(pipeline, x_train, y_train):
    param_grid = {
    "classifier__n_estimators": [100],
    "classifier__max_depth": [None],
    "classifier__min_samples_split": [10],
    'classifier__min_samples_leaf': [4], 
    "classifier__max_features": [25],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    return grid_search

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
def save_estimator(estimator):
    models_path = "../files/models"
    os.makedirs(models_path, exist_ok=True)

    with gzip.open("../files/models/model.pkl.gz", "wb") as file:
        pickle.dump(estimator, file)  
    
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
def calc_metrics(model, x_train, y_train, x_test, y_test):

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    metrics = [
        {
            'type': 'metrics',
            'dataset': 'train',
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
        },
        {
            'type': 'metrics',
            'dataset': 'test',
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
        },
        {
            'type': 'cm_matrix',
            'dataset': 'train',
            'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
            'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
        },
        {
            'type': 'cm_matrix',
            'dataset': 'test',
            'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
            'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
        }
    ]

    return metrics

# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
def save_metrics(metrics):
    metrics_path = "../files/output"
    os.makedirs(metrics_path, exist_ok=True)
    
    with open("../files/output/metrics.json", "w") as file:
        for metric in metrics:
            file.write(json.dumps(metric, ensure_ascii=False))
            file.write('\n')
            
# Ejecutamos todo con el main
def main():
    df_train, df_test = load_and_process_data()
    x_train, y_train, x_test, y_test = split_features_target(df_train, df_test)
    pipeline = create_pipeline()
    model = make_grid_search(pipeline, x_train, y_train)
    save_estimator(model)
    metrics = calc_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics)

    print(model.best_estimator_)
    print(model.best_params_)

if __name__ == "__main__":
    main()