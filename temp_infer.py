import mlflow.pyfunc
import pandas as pd

model_uri = 'file:///C:/development/ML-System/mlruns/0/models/m-fdde435077a84d50bb5a215cbf6be7a7/artifacts'
print('Loading', model_uri)
model = mlflow.pyfunc.load_model(model_uri)
print('Loaded. Predicting...')
print(model.predict(pd.DataFrame([[6.7,3.1,5.6,2.4]])))
