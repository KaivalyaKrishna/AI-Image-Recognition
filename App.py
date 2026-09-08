import os
import datetime
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Industrial Quality Control System",
    page_icon="🏭",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "casting_inspection_model.keras")
LOG_FILE = os.path.join(BASE_DIR, "inspection_records.csv")

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return tf.keras.models.load_model(MODEL_FILE)

def preprocess_image(pil_image):
    img = pil_image.convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype="float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def log_inspection(filename, verdict, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{
        "Timestamp": timestamp,
        "Image Name": filename,
        "Verdict": verdict,
        "Confidence (%)": round(confidence, 2)
    }])
    
    if os.path.exists(LOG_FILE):
        new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(LOG_FILE, mode='w', header=True, index=False)

def get_inspection_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Image Name", "Verdict", "Confidence (%)"])

st.title("Automated Quality Control & Defect Detection System")
st.caption("AI-powered visual inspection for industrial casting manufacturing lines")

model = load_trained_model()

if model is None:
    st.error(f"Model file not found! Please place 'casting_inspection_model.keras' inside this folder:\n{BASE_DIR}")
    st.stop()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Product Inspection Station")
    uploaded_file = st.file_uploader(
        "Upload a product image for inspection",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}")
        
        with st.spinner("Analyzing image..."):
            processed_input = preprocess_image(image)
            raw_prob = float(model.predict(processed_input, verbose=0)[0][0])
            
            if raw_prob >= 0.5:
                verdict = "DEFECTIVE"
                confidence = raw_prob * 100
            else:
                verdict = "GOOD"
                confidence = (1.0 - raw_prob) * 100
        
        st.divider()
        if verdict == "GOOD":
            st.success(f"Verdict: PASSED (GOOD) — Confidence: {confidence:.2f}%")
        else:
            st.error(f"Verdict: REJECTED (DEFECTIVE) — Confidence: {confidence:.2f}%")
            
        if st.button("Log Inspection Record"):
            log_inspection(uploaded_file.name, verdict, confidence)
            st.info("Record saved successfully.")
            st.rerun()

with col_right:
    st.subheader("Live Quality Analytics & Reporting")
    logs = get_inspection_logs()
    
    total_inspections = len(logs)
    good_count = len(logs[logs["Verdict"] == "GOOD"]) if total_inspections > 0 else 0
    defective_count = len(logs[logs["Verdict"] == "DEFECTIVE"]) if total_inspections > 0 else 0
    defect_rate = (defective_count / total_inspections * 100) if total_inspections > 0 else 0.0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Inspected", total_inspections)
    m2.metric("Good Products", good_count)
    m3.metric("Defects Found", defective_count)
    m4.metric("Defect Rate", f"{defect_rate:.1f}%")
    
    st.divider()
    st.write("Recent Inspection Log:")
    st.dataframe(logs.tail(10))
    
    if total_inspections > 0:
        csv_data = logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Quality Audit Report (CSV)",
            data=csv_data,
            file_name=f"qc_report_{datetime.date.today()}.csv",
            mime="text/csv"
        )