import mlflow.pyfunc
import pandas as pd

# URI model registry
model_uri = "models:/iris_classifier/Staging"

# Load model
model = mlflow.pyfunc.load_model(model_uri)

# Data baru
input_data = pd.DataFrame([[6.7, 3.1, 5.6, 2.4]])

# Prediksi
predictions = model.predict(input_data)

print(predictions)