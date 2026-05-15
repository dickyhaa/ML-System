import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
logged_model = 'runs:/0dff510b5ef34f4f9dacb6d1c573eae7/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
predict = loaded_model.predict(pd.DataFrame([[6.7, 3.1, 5.6, 2.4]]))

print(predict)