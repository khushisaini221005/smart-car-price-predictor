import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Sample Dataset
data = {
    'Year': [2014, 2013, 2017, 2011, 2014, 2018, 2015, 2020, 2016, 2019],
    'Present_Price': [5.59, 9.54, 9.85, 4.15, 6.87, 14.25, 8.50, 12.00, 7.60, 10.50],
    'Kms_Driven': [27000, 43000, 6900, 52000, 42450, 20000, 35000, 15000, 28000, 18000],
    'Fuel_Type_Diesel': [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    'Fuel_Type_Petrol': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    'Transmission_Manual': [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    'Selling_Price': [3.35, 4.75, 7.25, 2.85, 4.60, 11.50, 5.25, 9.50, 5.80, 8.20]
}

df = pd.DataFrame(data)

# Features & Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save Model
joblib.dump(model, 'car_price_model.pkl')
print("Model saved successfully!")
