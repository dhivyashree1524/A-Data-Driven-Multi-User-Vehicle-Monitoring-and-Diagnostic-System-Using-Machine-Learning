# -*- coding: utf-8 -*-

import streamlit as st
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

st.set_page_config(page_title="Vehicle Monitoring System", layout="wide")

st.title("Multi-User Vehicle Monitoring & Analytics System")

# -------------------- LOAD DATA --------------------
with open("vehicle_data.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

vehicles = data["vehicles"]
vehicle_ids = [v["vehicle_id"] for v in vehicles]

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Control Panel")

selected_vehicle_id = st.sidebar.selectbox("Select Vehicle ID", vehicle_ids)

st.sidebar.markdown("### 👤 Select Role")
role = st.sidebar.radio("", ["Driver", "Technician", "Manufacturer"])

st.sidebar.markdown("### 🌐 Select Language")
language = st.sidebar.selectbox("", ["English", "Tamil"])

vehicle = next(v for v in vehicles if v["vehicle_id"] == selected_vehicle_id)

# -------------------- BASIC INFO --------------------
st.subheader("📌 Vehicle Basic Data")

colA, colB = st.columns(2)

with colA:
    st.write(f"**Driver:** {vehicle['driver']}")
    st.write(f"**Location:** {vehicle['location']}")
    st.write(f"**Speed:** {vehicle['speed']} km/h")

with colB:
    st.write(f"**Vehicle ID:** {vehicle['vehicle_id']}")
    st.write(f"**Last Updated:** {vehicle['timestamp']}")
    st.write(f"**Driver Score:** {vehicle['driver_score']}")

# -------------------- COMPLETE DASHBOARD (ALL 15 FACTORS) --------------------

# -------------------- ROLE BASED DASHBOARD --------------------

st.markdown("---")

if role == "Driver":

    st.subheader("👨‍✈️ Driver Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Speed (km/h)", vehicle["speed"])
    col2.metric("Fuel Level (%)", vehicle["gas_level"])
    col3.metric("Engine Temperature (°C)", vehicle["temperature"])

    col1.metric("Oil Level (%)", vehicle["oil_level"])
    col2.metric("Driver Score", vehicle["driver_score"])

    st.info("Simplified driving insights based on vehicle condition.")

# --------------------------------------------------------------

elif role == "Technician":

    st.subheader("🧑‍🔧 Technician Diagnostic Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Engine Temperature (°C)", vehicle["temperature"])
    col2.metric("Fuel Level (%)", vehicle["gas_level"])
    col3.metric("Oil Level (%)", vehicle["oil_level"])

    col1.metric("Battery Voltage (V)", vehicle["battery_voltage"])
    col2.metric("Tyre Pressure (PSI)", vehicle["tyre_pressure"])
    col3.metric("Engine RPM", vehicle["engine_rpm"])

    col1.metric("Brake Health (%)", vehicle["brake_health"])
    col2.metric("Accident Tilt", vehicle["accident_tilt"])
    col3.metric("Fault Code", vehicle["fault_code"])

    st.warning("Technical parameters shown for maintenance and diagnostics.")

# --------------------------------------------------------------

elif role == "Manufacturer":

    st.subheader("🏭 Manufacturer Fleet Analytics Dashboard")

    st.json(vehicle)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Driver Score", vehicle["driver_score"])
        st.metric("Brake Health (%)", vehicle["brake_health"])
        st.metric("Battery Voltage (V)", vehicle["battery_voltage"])

    with col2:
        st.metric("Engine RPM", vehicle["engine_rpm"])
        st.metric("Fuel Level (%)", vehicle["gas_level"])
        st.metric("Temperature (°C)", vehicle["temperature"])

    # Fleet comparison graph
    df = pd.DataFrame(vehicles)

    st.subheader("Fleet Temperature Comparison")

    plt.figure()
    plt.bar(df["vehicle_id"], df["temperature"])
    plt.xticks(rotation=90)
    plt.title("Temperature Across Fleet")
    st.pyplot(plt)

    st.info("Advanced fleet-level insights and performance analytics.")

# -------------------- ALERT SYSTEM --------------------

thresholds = {
    "temperature": 100,
    "gas_level": 20,
    "oil_level": 25
}

alerts = []

if vehicle["temperature"] > thresholds["temperature"]:
    alerts.append("எஞ்சின் வெப்பநிலை அதிகமாக உள்ளது" if language=="Tamil" else "Engine temperature is high")

if vehicle["gas_level"] < thresholds["gas_level"]:
    alerts.append("எரிபொருள் அளவு குறைவாக உள்ளது" if language=="Tamil" else "Fuel level is low")

if vehicle["oil_level"] < thresholds["oil_level"]:
    alerts.append("எண்ணெய் அளவு மிகவும் குறைவாக உள்ளது" if language=="Tamil" else "Oil level is critically low")

if vehicle["brake_health"] < 50:
    alerts.append("பிரேக் நிலை மிகவும் மோசமாக உள்ளது" if language=="Tamil" else "Brake condition is critical")

if vehicle["battery_voltage"] < 11.5:
    alerts.append("பேட்டரி மின்னழுத்தம் குறைவாக உள்ளது" if language=="Tamil" else "Battery voltage is low")

if vehicle["accident_tilt"]:
    alerts.append("விபத்து கண்டறியப்பட்டது" if language=="Tamil" else "Accident detected")

st.markdown("---")

speech_text = ""

if alerts:
    st.subheader("🚨 Vehicle Alerts Detected")
    for alert in alerts:
        st.error(alert)
    speech_text = ". ".join(alerts)
else:
    normal_msg = "வாகன நிலை இயல்பாக உள்ளது. எச்சரிக்கை எதுவும் இல்லை." if language=="Tamil" else "Vehicle status normal. No alerts detected."
    st.success("✅ " + normal_msg)
    speech_text = normal_msg

# -------------------- AUTO SPEECH --------------------

st.components.v1.html(f"""
<script>
window.speechSynthesis.cancel();
var msg = new SpeechSynthesisUtterance("{speech_text}");
msg.lang = "{'ta-IN' if language=='Tamil' else 'en-US'}";
msg.rate = 1;
window.speechSynthesis.speak(msg);
</script>
""", height=0)

# ===================== MACHINE LEARNING SECTION =====================

st.markdown("---")
st.header("🤖 AI / Machine Learning Analysis")

df = pd.DataFrame(vehicles)

feature_cols = ["temperature","gas_level","oil_level",
                "battery_voltage","tyre_pressure","brake_health"]

iso_model = IsolationForest(contamination=0.15, random_state=42)
iso_model.fit(df[feature_cols])

current_features = np.array([[
    vehicle["temperature"],
    vehicle["gas_level"],
    vehicle["oil_level"],
    vehicle["battery_voltage"],
    vehicle["tyre_pressure"],
    vehicle["brake_health"]
]])

if iso_model.predict(current_features)[0] == -1:
    st.error("ML detected abnormal vehicle behavior" if language=="English"
             else "இயந்திர கற்றல் முறையில் அசாதாரண நிலை கண்டறியப்பட்டது")
else:
    st.success("No anomaly detected by ML")

df["risk_label"] = df["driver_score"].apply(lambda x: 1 if x < 70 else 0)

X = df[["speed","temperature","brake_health"]]
y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

risk_pred = rf_model.predict([[
    vehicle["speed"],
    vehicle["temperature"],
    vehicle["brake_health"]
]])

if risk_pred[0] == 1:
    st.error("High Risk Driving Pattern Detected" if language=="English"
             else "உயர் அபாய ஓட்டுநர் நடத்தை கண்டறியப்பட்டது")
else:
    st.success("Driver behavior normal")

accuracy = accuracy_score(y_test, rf_model.predict(X_test))
st.info(f"Model Accuracy: {round(accuracy*100,2)}%")

# -------------------- ML GRAPHS --------------------

# -------------------- ML GRAPHS (DYNAMIC PER VEHICLE) --------------------

st.markdown("---")
st.header("📈 ML Visual Analytics")

df = pd.DataFrame(vehicles)

# -------- 1️⃣ Feature Importance (Same Model, But Highlight Selected Vehicle) --------

st.subheader("Feature Importance (Driver Risk Model)")

plt.figure()
features = ["Speed", "Temperature", "Brake Health"]
importances = rf_model.feature_importances_

plt.bar(features, importances)
plt.title(f"Feature Importance for Risk Prediction - {vehicle['vehicle_id']}")
st.pyplot(plt)
plt.close()

# -------- 2️⃣ Vehicle vs Fleet Temperature Comparison --------

st.subheader("Vehicle Temperature vs Fleet Average")

fleet_avg_temp = df["temperature"].mean()
selected_temp = vehicle["temperature"]

plt.figure()
plt.bar(["Fleet Avg", selected_vehicle_id],
        [fleet_avg_temp, selected_temp])
plt.title("Temperature Comparison")
st.pyplot(plt)
plt.close()

# -------- 3️⃣ Vehicle Health Radar Style Bar --------

st.subheader("Vehicle Health Snapshot")

plt.figure()
health_metrics = [
    vehicle["temperature"],
    vehicle["gas_level"],
    vehicle["oil_level"],
    vehicle["brake_health"],
    vehicle["battery_voltage"] * 10  # scaled for visibility
]

labels = ["Temp", "Fuel", "Oil", "Brake", "Battery(x10)"]

plt.bar(labels, health_metrics)
plt.title(f"Health Metrics - {selected_vehicle_id}")
st.pyplot(plt)
plt.close()

# -------- 4️⃣ Risk Prediction Probability --------

st.subheader("Driver Risk Prediction Confidence")

risk_prob = rf_model.predict_proba([[
    vehicle["speed"],
    vehicle["temperature"],
    vehicle["brake_health"]
]])[0]

plt.figure()
plt.bar(["Safe", "High Risk"], risk_prob)
plt.title(f"Risk Probability - {selected_vehicle_id}")
st.pyplot(plt)
plt.close()

# -------------------- PDF REPORT --------------------

st.markdown("---")
st.subheader("📄 Generate Vehicle Report")

if st.button("Generate PDF Report"):

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Vehicle Monitoring Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    data_table = [[k, str(v)] for k, v in vehicle.items()]
    table = Table(data_table)
    table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black)]))

    elements.append(table)
    elements.append(Spacer(1, 0.4 * inch))

    plt.figure()
    plt.bar(["Temp","Fuel","Oil","Brake"],
            [vehicle["temperature"],
             vehicle["gas_level"],
             vehicle["oil_level"],
             vehicle["brake_health"]])
    graph_path = "graph.png"
    plt.savefig(graph_path)
    plt.close()

    elements.append(RLImage(graph_path, width=4*inch, height=3*inch))
    doc.build(elements)

    with open(temp_file.name, "rb") as f:
        st.download_button("Download Report",
                           f,
                           file_name=f"{vehicle['vehicle_id']}_report.pdf",
                           mime="application/pdf")
