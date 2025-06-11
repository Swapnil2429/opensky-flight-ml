import streamlit as st
import pandas as pd
import pyodbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 📡 Azure SQL DB connection
# -------------------------------
server = st.secrets["azure_sql"]["server"]
database = st.secrets["azure_sql"]["database"]
username = st.secrets["azure_sql"]["username"]
password = st.secrets["azure_sql"]["password"]
driver = "{ODBC Driver 17 for SQL Server}"

# -------------------------------
# 📥 Load data from Azure SQL
# -------------------------------
@st.cache_data
def load_data():
    try:
        conn = pyodbc.connect(
            f"DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}"
        )
        query = "SELECT * FROM dbo.OpenskyFlights"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()

# -------------------------------
# 🚀 Streamlit UI
# -------------------------------
st.title("✈️ OpenSky Altitude Class Predictor")

df = load_data()

if df.empty:
    st.stop()

st.markdown("This app predicts whether a flight is flying at **LOW**, **MEDIUM**, or **HIGH** altitude.")

# Show data sample
st.subheader("🔍 Sample Data")
st.dataframe(df.head(10))

# -------------------------------
# 🧠 Train ML model
# -------------------------------
features = ['longitude', 'latitude', 'altitude', 'velocity', 'heading']
df_clean = df.dropna(subset=features + ['altitude_class'])

X = df_clean[features]
y = df_clean['altitude_class']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# -------------------------------
# 📊 User Input for Prediction
# -------------------------------
st.subheader("🎯 Predict Altitude Class")

user_input = {
    'longitude': st.slider("Longitude", float(X.longitude.min()), float(X.longitude.max()), float(X.longitude.mean())),
    'latitude': st.slider("Latitude", float(X.latitude.min()), float(X.latitude.max()), float(X.latitude.mean())),
    'altitude': st.slider("Altitude", float(X.altitude.min()), float(X.altitude.max()), float(X.altitude.mean())),
    'velocity': st.slider("Velocity", float(X.velocity.min()), float(X.velocity.max()), float(X.velocity.mean())),
    'heading': st.slider("Heading", float(X.heading.min()), float(X.heading.max()), float(X.heading.mean()))
}

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
predicted_label = le.inverse_transform([prediction])[0]

st.success(f"🛬 Predicted Altitude Class: **{predicted_label}**")
