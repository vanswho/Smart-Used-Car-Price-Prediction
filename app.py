from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your saved models and encodings
model = joblib.load('car_price_prediction_model (1).pkl')
scaler = joblib.load('scaler.pkl')
brand_mean_price = joblib.load('brand_mean_price.pkl')
model_mean_price = joblib.load('model_mean_price.pkl')

# Mappings for categorical variables
seller_type_mapping = {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}
fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
transmission_mapping = {'Manual': 0, 'Automatic': 1}

@app.route('/')
def home():
    return render_template('index.html')  # Your HTML form file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        vehicle_age = int(request.form['vehicle_age'])
        km_driven = int(request.form['km_driven'])
        mileage = float(request.form['mileage'])
        engine = int(request.form['engine'])
        max_power = float(request.form['max_power'])
        seller_type = request.form['seller_type']
        fuel_type = request.form['fuel_type']
        transmission_type = request.form['transmission_type']
        brand = request.form['brand']
        model_name = request.form['model']

        # Encode categorical features
        seller_type_enc = seller_type_mapping.get(seller_type, 0)
        fuel_type_enc = fuel_type_mapping.get(fuel_type, 0)
        transmission_enc = transmission_mapping.get(transmission_type, 0)

        # Target encode brand and model, fallback to mean if not found
        brand_enc = brand_mean_price.get(brand, brand_mean_price.mean())
        model_enc = model_mean_price.get(model_name, model_mean_price.mean())

        # Prepare data dictionary (no brand/model anymore)
        input_dict = {
            'vehicle_age': vehicle_age,
            'km_driven': km_driven,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seller_type': seller_type_enc,
            'fuel_type': fuel_type_enc,
            'transmission_type': transmission_enc,
            'brand_encoded': brand_enc,
            'model_encoded': model_enc
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Scale features
        scaled_input = scaler.transform(input_df)

        # Predict price
        prediction = model.predict(scaled_input)[0]

        return f"<h2>Predicted Selling Price: ₹{round(prediction)}</h2>"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
