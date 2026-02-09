App link https://genai160123748055-kvfaenauqgjyejfssevc34.streamlit.app/
Here is the professional documentation for your project. This is structured to be used as your README.md file on GitHub, which will impress your evaluators for the AI²: Applied Generative AI course.
🧠 AI²: Brain Tumor Diagnostic Lab
Applied Generative AI for Intelligent Applications
This project implements a high-accuracy Deep Learning system to detect and classify brain tumors from MRI scans. It compares a custom baseline architecture with a fine-tuned MobileNetV2 model to demonstrate the power of Transfer Learning in medical diagnostics.
📁 Project Structure
Organize your GitHub repository as follows:
Brain-Tumor-AI-Lab/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation (This file)
├── LICENSE                # MIT License
├── models/
│   └── max_val_brain_model.keras  # The final trained high-accuracy model
└── notebooks/
    └── Training_Notebook.ipynb    # Your Google Colab training code

🚀 How to Follow the App (Steps)
1. Training Phase (Google Colab)
 * Open the provided notebook in Google Colab.
 * Set the Runtime to T4 GPU.
 * Run the cells to clone the Sartaj Brain Tumor Dataset.
 * The script performs Two-Stage Training:
   * Stage 1: Training the classification head (initial weights).
   * Stage 2: Fine-tuning the top 20 layers of MobileNetV2 with a low learning rate (1e-5).
 * Download the resulting max_val_brain_model.keras file.
2. Deployment Phase (Streamlit)
 * Upload the project files to a GitHub repository.
 * Visit Streamlit Cloud.
 * Connect your GitHub account and select this repository.
 * Deploy the app.py file.
🧪 Testing Procedures
To verify the model's reliability, perform the following tests within the application:
| Test Case | Expected Input | Expected Outcome |
|---|---|---|
| Normal Scan | MRI image from 'No Tumor' folder | "Result: Normal" with High Confidence |
| Pathological Scan | MRI image from 'Glioma' folder | "Potential Glioma detected" alert |
| Image Preprocessing | Low-resolution or unscaled image | Model should still predict correctly due to preprocess_input logic |
Verification Metrics:
 * Training Accuracy: ~92%
 * Validation Accuracy: ~90%+
 * Generalization Gap: <5\% (Fixed the previous 85-58 gap using Dropout and Global Average Pooling).
🛠️ Tech Stack
 * Framework: TensorFlow / Keras
 * Architecture: MobileNetV2 (Transfer Learning)
 * Frontend: Streamlit
 * Language: Python 3.10+
 * Deployment: Streamlit Community Cloud
📜 License & Disclaimer
License: Distributed under the MIT License.
⚠️ MEDICAL DISCLAIMER:
This application is developed as a Value Added Course (VAC) project for educational purposes only. It is not intended for clinical use or real-world medical diagnosis. AI predictions should always be verified by a certified Radiologist or Medical Professional.
