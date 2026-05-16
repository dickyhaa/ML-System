import mlflow.pyfunc
import pandas as pd

# URI model dalam Model Registry
model_uri_testing = "models:/iris_classifier@testingv1" # Update Version
model_uri_evaluasi = "models:/iris_classifier@evaluasi" # Update Version

# Load model
model_test = mlflow.pyfunc.load_model(model_uri_testing)
model_evaluasi = mlflow.pyfunc.load_model(model_uri_evaluasi)

# Gunakan model untuk prediksi
input_data = pd.DataFrame([[6.7, 3.1, 5.6, 2.4]])
predictions1 = model_test.predict(input_data)
print("Hasil prediksi testing: ",predictions1)
predictions2 = model_evaluasi.predict(input_data)
print("Hasil prediksi evaluasi: ",predictions2)