# Al-for-software-eng_week-5
## Breast Cancer Image Classification & Inference API
This project provides a pipeline for breast cancer image classification using color histogram features and a Random Forest classifier. It also includes a lightweight Flask API for model inference, allowing you to predict the priority (malignant/benign) of new images.

## Table of Contents

## Project Overview

-Features
-Setup Instructions
-Usage
-Jupyter Notebook
-Flask Inference API
-API Example
-Project Structure
-Next Steps
-Project Overview
-Goal: Classify breast cancer images as high (malignant) or low (benign) priority.
-Approach: Extract color histogram features from images and train a Random Forest classifier.
-API: A Flask-based REST API for easy model inference on new images.
-Features
-Data loading and preprocessing
-Feature extraction using color histograms
-Model training and evaluation (accuracy, F1-score, confusion matrix)
-Batch prediction on unseen test images
-Lightweight Flask API for real-time inference
## Setup Instructions
- Clone the repository:
    git clone https://github.com/yourusername/breast-cancer-classification.git
   cd breast-cancer-classification
- Install dependenciesv:
   pip install -r requirements.txt
  
### Train the model (Jupyter Notebook):
- Open breast_cancer_analysis.ipynb and run all cells to train the model and save it as rf_model.joblib.
- Run the Flask API:
Ensure rf_model.joblib is present in the project directory.
- Start the API:
-      python inference_api.py
## Project Structure
.
├── breast_cancer_analysis.ipynb   # Jupyter notebook for training and evaluation
├── inference_api.py               # Flask API for inference
├── rf_model.joblib                # Saved Random Forest model (after training)
├── test_set_predictions.csv       # Predictions on test set
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation  
### Next Steps
- Improve feature extraction (e.g., deep learning, texture analysis)
  Add more advanced model selection and hyperparameter tuning
  Integrate additional clinical data if available
by:
Liso Mlunguza
Email: lisomlunguza8@gmail.com  
   

