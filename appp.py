# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
df = pd.read_csv('crop_yield_maharashtra.csv')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['District', 'Crop', 'Season', 'Soil_Type']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop('Yield_Quintal_per_Hectare', axis=1)
y = df['Yield_Quintal_per_Hectare']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save the trained model into a .pkl file
with open('crop_yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as 'crop_yield_model.pkl'")

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('crop_yield_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title and Header
st.set_page_config(page_title="Crop Yield Prediction | Maharashtra", layout="wide")

st.title("🌾 Crop Yield Prediction App")
st.subheader("Predict the crop yield (in quintal/hectare) based on conditions")

st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Dropdown options
districts = [
    "Pune", "Nashik", "Nagpur", "Aurangabad", "Solapur", "Amravati", "Kolhapur", "Sangli",
    "Satara", "Ahmednagar", "Dhule", "Jalgaon", "Latur", "Beed", "Osmanabad", "Nanded",
    "Parbhani", "Hingoli", "Buldhana", "Washim", "Akola", "Yavatmal", "Wardha", "Chandrapur",
    "Gadchiroli", "Bhandara", "Gondia", "Raigad", "Ratnagiri", "Sindhudurg", "Palghar", "Thane", "Mumbai"
]

crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Soybean"]
seasons = ["Kharif", "Rabi", "Summer"]
soil_types = ["Black", "Red", "Laterite", "Alluvial"]

# User Inputs
district = st.sidebar.selectbox("District", districts)
crop = st.sidebar.selectbox("Crop", crops)
season = st.sidebar.selectbox("Season", seasons)
soil_type = st.sidebar.selectbox("Soil Type", soil_types)

rainfall = st.sidebar.slider("Rainfall (mm)", 400, 3000, 1000)
temperature = st.sidebar.slider("Temperature (°C)", 20, 40, 30)
fertilizer = st.sidebar.slider("Fertilizer Used (kg/hectare)", 50, 400, 100)
pesticide = st.sidebar.slider("Pesticide Used (liters/hectare)", 1, 10, 3)
area = st.sidebar.slider("Area (hectares)", 0.5, 10.0, 2.0)

# Encode inputs manually (simple encoding like training)
def encode_input(value, options):
    return options.index(value)

district_enc = encode_input(district, districts)
crop_enc = encode_input(crop, crops)
season_enc = encode_input(season, seasons)
soil_type_enc = encode_input(soil_type, soil_types)

# Prepare final input array
input_features = np.array([[district_enc, crop_enc, season_enc, rainfall, temperature, soil_type_enc,
                            fertilizer, pesticide, area]])

# Prediction
if st.sidebar.button('Predict Yield'):
    prediction = model.predict(input_features)[0]
    st.success(f"🌱 Estimated Crop Yield: *{prediction:.2f} quintal/hectare*")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px;">
        Developed with ❤️ by <b>OM choudhari</b> | Maharashtra Agriculture Data
    </div>
    """, unsafe_allow_html=True
)