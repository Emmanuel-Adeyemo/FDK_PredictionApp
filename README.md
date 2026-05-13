## FDK Prediction App: Deep Learning for Fungal Damage Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dash](https://img.shields.io/badge/Dash-Plotly-008bb4.svg)](https://dash.plotly.com/)
[![Deployment](https://img.shields.io/badge/Deployed-Render-46E3B7.svg)](https://fdk-prediction-app.onrender.com/)

### Overview
This repository contains an end-to-end computer vision service designed to automate the detection and scoring of Fusarium Damaged Kernels (FDK) in wheat. By leveraging deep learning, this application replaces subjective manual assessments with a scalable, data-driven scoring system to assist plant breeders and pathologists in grain quality evaluation.

### Key Features
* Deep Learning Engine: Utilizes a fine-tuned EfficientNet-B2 architecture for disease scoring.
* Explainable AI: Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize and identify the specific morphological and spectral features driving model predictions.
* Interactive Web Service: A full-stack Dash application that enables non-technical stakeholders to upload grain images and receive real-time, automated scores.
* Production Ready: Architected for cloud deployment with robust handling for real-time inference.

### Technical Stack
* **Modeling:** PyTorch, Torchvision
* **Image Processing:** OpenCV
* **Web Framework:** Plotly Dash, Flask
* **Deployment:** Render, Gunicorn
* **Data Handling:** NumPy, Pandas


### Methodology
#### Data & Training
The model was trained on wheat grain images, optimized to distinguish between healthy kernels and various levels of fungal damage.

#### Interpretability
To ensure the model's interpretability, Grad-CAM overlays highlight the model's areas of interest, allowing users to verify that the model is focusing on relevant biological indicators rather than background noise.

### Getting Started
1. Clone the repository.
2. Install dependencies: pip install -r requirements.txt
3. Run the app: python app.py
4. Access the dashboard at http://127.0.0.1:8050/ in your browser.

Experience the application live: https://fdk-prediction-app.onrender.com/
