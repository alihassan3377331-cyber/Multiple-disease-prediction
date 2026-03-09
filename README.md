# 🩺 Multiple Disease Prediction System

A comprehensive disease prediction system that uses blood sample data to predict multiple diseases using machine learning algorithms.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Machine Learning Models](#machine-learning-models)
- [GUI Application](#gui-application)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Project Overview

This project implements a **Multiple Disease Prediction System** that analyzes blood test results to predict various diseases. The system uses machine learning algorithms (K-Nearest Neighbors and Decision Tree) trained on blood sample datasets to make predictions.

### Key Highlights:
- 📊 Analyzes blood sample data
- 🤖 Uses ML models for disease prediction
- 🖥️ Provides an intuitive GUI for easy interaction
- ⚡ Real-time predictions

---

## ✨ Features

1. **Data Processing**
   - Downloads and combines datasets from Kaggle
   - Handles missing values and data cleaning
   - Feature extraction from blood test parameters

2. **Machine Learning Models**
   - K-Nearest Neighbors (KNN) Classifier
   - Decision Tree Classifier
   - Model accuracy evaluation

3. **Graphical User Interface (GUI)**
   - User-friendly Tkinter interface
   - Dynamic input fields based on dataset features
   - Model selection dropdown
   - Real-time prediction results
   - Progress indicators during training

4. **Data Visualization**
   - Distribution plots
   - Correlation heatmaps
   - Count plots
   - Pair plots

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.x |
| **GUI Framework** | Tkinter |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Data Visualization** | Matplotlib, Seaborn |
| **Data Source** | Kaggle (kagglehub) |

---

## 📂 Project Structure

```
multiple disease/
│
├── # Multiple Disease Prediction System GUI.py   # Main GUI Application
├── multiple_disease_prediction.py                 # ML Model Training & Analysis
├── Blood_samples_dataset_balanced_2(f).csv       # Training Dataset
├── blood_samples_dataset_test.csv                 # Test Dataset
└── README.md                                       # Project Documentation
```

---

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Libraries
Install the required libraries using:

```
bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

### Kaggle Configuration
The application uses `kagglehub` to download datasets. Make sure you have:
- Kaggle account
- Kaggle API token (optional, but may be required for some datasets)

---

## 🚀 Usage

### Option 1: GUI Application

1. Run the GUI application:
```
bash
python "# Multiple Disease Prediction System GUI.py"
```

2. Click on **"📥 Download Data & Train Models"** button
3. Wait for training to complete (progress bar will show)
4. Enter blood test values in the input fields
5. Select a model (Decision Tree or KNN) from dropdown
6. Click **"🔍 Predict Disease"** to see the prediction result

### Option 2: ML Analysis Script

Run the analysis script to see model training and evaluation:
```
bash
python multiple_disease_prediction.py
```

---

## 📊 Datasets

### Source
- **Kaggle Dataset**: `ehababoelnaga/multiple-disease-prediction`
- **Files**:
  - `Blood_samples_dataset_balanced_2(f).csv` - Balanced training data
  - `blood_samples_dataset_test.csv` - Test data

### Features (Blood Test Parameters)
The dataset includes various blood test parameters such as:
- Glucose
- Cholesterol
- White Blood Cells
- Red Blood Cells
- Hemoglobin
- And more...

### Target Variable
- **Disease** - The disease to be predicted (multiple classes)

---

## 🤖 Machine Learning Models

### 1. K-Nearest Neighbors (KNN)
- **Algorithm**: Instance-based learning
- **Parameters**: n_neighbors=1
- **Use Case**: Classification based on similarity

### 2. Decision Tree Classifier
- **Algorithm**: Tree-based model
- **Parameters**: Default settings
- **Use Case**: Interpretable classification rules

### Model Performance
Both models are trained and evaluated on the same dataset:
- Training set: 80% of data
- Test set: 20% of data
- Random state: 42 (for reproducibility)

---

## 🖥️ GUI Application

### Main Features:
1. **Header**: Application title
2. **Control Panel**: 
   - Train Models button
   - Status display
   - Progress bar
3. **Input Panel**: 
   - Scrollable list of blood test parameters
   - Entry fields for each parameter
4. **Prediction Panel**:
   - Model selection dropdown
   - Predict button
   - Result display area

### How to Use:
1. Launch the application
2. Click "Download Data & Train Models"
3. Wait for training to complete
4. Enter patient blood test values
5. Select model type
6. Click "Predict Disease"
7. View prediction result

---

## 📈 Results

### Model Accuracy
The system evaluates models using:
- Accuracy Score
- Mean Absolute Error
- R2 Score

### Output
- Predicted disease name displayed in the GUI
- Model accuracy shown in status bar

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is for educational purposes.

---

## 👨‍💻 Author

Created as a demonstration of machine learning in healthcare applications.

---

## ⚠️ Disclaimer

**Important**: This is a demonstration/educational project. The predictions made by this system should NOT be used for actual medical diagnosis. Always consult healthcare professionals for medical decisions.
