# 🏭 AI-Powered Visual Quality Control & Defect Detection System
### *An Industrial Computer Vision Solution for Automated Manufacturing Lines*

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.08%25-brightgreen)
![Precision](https://img.shields.io/badge/Defect%20Precision-97%25-success)

---

## 📸 Live Application Preview

<div align="center">
  <img src="assets/dashboard_overview.png" alt="Live Quality Control Dashboard" width="900"/>
  <p><em>Figure 1: Streamlit Inspection Bay, Real-Time Quality KPI Telemetry, and Audit Log.</em></p>
</div>

---

## 📌 Project Overview & Problem Statement

In industrial metal casting (used for automotive engine blocks, aerospace housings, and high-pressure valves), structural imperfections such as **shrinkage cracks, surface blowholes, and slag inclusions** can cause catastrophic mechanical failure if not caught early.

### The Challenge of Manual Inspection:
- **Human Eye Fatigue**: Drops inspection consistency over an 8-hour shift.
- **Subjective Criteria**: Varies between different shift inspectors.
- **Line Bottlenecks**: Manual checks take 4 to 10 seconds per unit.

### The AI Solution:
An end-to-end automated Computer Vision inspection system that processes product images in **< 50 milliseconds**, delivers an instant **PASSED / REJECTED** verdict with a probability confidence score, updates live production quality metrics, and maintains a downloadable CSV audit trail for ISO 9001 compliance.

---

## 🔍 Visualizing Good vs. Defective Components

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Good Product (PASSED)</b></td>
      <td align="center"><b>Defective Product (REJECTED)</b></td>
    </tr>
    <tr>
      <td><img src="sample_images/sample_good.jpeg" alt="Good Product" width="300"/></td>
      <td><img src="sample_images/sample_defective.jpeg" alt="Defective Product" width="300"/></td>
    </tr>
    <tr>
      <td align="center">Clean bore rim, smooth uniform surface.</td>
      <td align="center">Visible void and irregular surface cracks along the rim.</td>
    </tr>
  </table>
</div>

---

## 🏗️ End-to-End System Flowchart

```text
       Conveyor Inspection Sensor / Camera
                        │
                        ▼
             [ Upload Product Image ]
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │     PREPROCESSING & NORMALIZATION     │
    │  • BGR -> RGB color conversion        │
    │  • Resize: 512x512 -> 224x224 pixels  │
    │  • Scale intensities: [0, 255] -> 0-1 │
    │  • Tensor Batch Reshape: (1, 224,224,3│
    └───────────────────┬───────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │     MOBILENETV2 FEATURE EXTRACTOR     │
    │     (Pretrained on 1.4M Images)       │
    │  • Extracts low, mid, and high-level  │
    │    spatial contours and textures      │
    └───────────────────┬───────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │      CUSTOM CLASSIFICATION HEAD       │
    │  • GlobalAveragePooling2D             │
    │  • Dropout (0.3) Regularization       │
    │  • Dense Layer (128 units, ReLU)      │
    │  • Dense Output (1 unit, Sigmoid)     │
    └───────────────────┬───────────────────┘
                        │
                        ▼
                 SIGMOID OUTPUT
            ┌───────────┴───────────┐
            │                       │
      Score < 0.50            Score >= 0.50
            │                       │
            ▼                       ▼
    🟢 VERDICT: PASSED      🔴 VERDICT: REJECTED
    Confidence: (1 - P)%    Confidence: (P)%
            │                       │
            └───────────┬───────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │      STREAMLIT OPERATOR CONSOLE       │
    │  1. Render Status Alert & Confidence  │
    │  2. Log to inspection_records.csv     │
    │  3. Real-time Defect Rate Telemetry   │
    │  4. Export CSV Quality Audit Reports  │
    └───────────────────────────────────────┘
```

---

## 📊 Dataset & Verified Results

- **Dataset**: Real-Life Industrial Casting Product Dataset (1,300 images: 519 Good, 781 Defective).
- **Split**: Stratified 80% Train (1,040), 10% Validation (130), 10% Test (130).

### Evaluation on 130 Unseen Test Images

```text
Test Accuracy: 93.08%
Test Loss    : 0.1995

Classification Report:
               precision    recall  f1-score   support

     Good (0)       0.88      0.96      0.92        52
Defective (1)       0.97      0.91      0.94        78

     accuracy                           0.93       130
    macro avg       0.92      0.94      0.93       130
 weighted avg       0.93      0.93      0.93       130
```

- **Defect Precision (97%)**: Minimizes false rejections, avoiding unnecessary scrapping of good products.
- **Defect Recall (91%)**: Catches 91% of real defects, preventing compromised parts from reaching customers.
- **Convergence Verification**: Training loss (`0.2365`) and Validation loss (`0.2352`) converged closely with zero divergence, confirming no overfitting.

---

## 💻 Streamlit Web Application Workflow

<div align="center">
  <img src="assets/inspection_demo.png" alt="Inspection Bay Demo" width="850"/>
  <p><em>Figure 2: Real-time inference showing PASSED vs REJECTED indicators and telemetry counters.</em></p>
</div>

1. **Inspection Station**: Upload casting images to receive an instantaneous color-coded verdict and confidence rating.
2. **Audit Logging**: Saves timestamp, filename, verdict, and confidence score to `inspection_records.csv`.
3. **Live Telemetry Cards**: Tracks Total Inspected, Good vs. Defect counts, and line Defect Rate (%).
4. **Export Report**: Download the full shift log as an audit-ready CSV file.

---

## 📁 Project Directory Structure

```text
AI_Image_Recognition_Quality_Control/
│
├── App.py                            # Streamlit web application & inspection station
├── casting_inspection_model.keras    # Saved MobileNetV2 trained model weights
├── inspection_records.csv            # Auto-generated CSV logging database
├── requirements.txt                  # Python library dependencies
├── README.md                         # Detailed project documentation
│
├── assets/                           # Screenshots for documentation
│   ├── dashboard_overview.png
│   └── inspection_demo.png
│
├── sample_images/                    # Verification samples for testing
│   ├── sample_good.jpeg              # Sample non-defective product
│   └── sample_defective.jpeg         # Sample defective product
│
└── notebooks/
    └── Model_Training.ipynb          # Full training & evaluation notebook (Kaggle)
```

---

## 🚀 How to Run the Project Locally

1. Open PowerShell or Command Prompt and navigate to your project directory:
   ```powershell
   cd "C:\Users\Kaivalya Krishna\OneDrive\Desktop\Codes\Projects\AI_Image_Recognition\AI-Image-Recognition"
   ```

2. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```powershell
   streamlit run App.py
   ```

4. The application will open automatically in your browser at `http://localhost:8501`. Upload images from `sample_images/` to test.

---

## 👤 Author

- **Developer**: Kaivalya Krishna
- **Project Domain**: Artificial Intelligence, Computer Vision & Industrial Quality Assurance