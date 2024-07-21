---

# Diabetes Prediction Model 🩺📊

Welcome to the Diabetes Prediction Model repository! This project is aimed at predicting diabetes using logistic regression on a dataset. Below you'll find details on how to use the model, its performance, and how to get started.

## 📋 Project Overview

This project involves building a logistic regression model to predict diabetes based on various features. The dataset used includes information on demographics, health conditions, and lifestyle factors.

## 🚀 Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Armanx200/Diabetes_Model.git
   cd Diabetes_Model
   ```

2. **Install Dependencies**
   Ensure you have Python 3.12 or later. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Model**
   Execute the following script to run the model:
   ```bash
   python Model.py
   ```

## 📊 Model Performance

The logistic regression model's performance is summarized below:

- **Accuracy**: 96.07%
- **Confusion Matrix**:
  ```
  [[18140   157]
   [  629  1074]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

             0       0.97      0.99      0.98     18297
             1       0.87      0.63      0.73      1703

      accuracy                           0.96     20000
     macro avg       0.92      0.81      0.86     20000
  weighted avg       0.96      0.96      0.96     20000
  ```

## 📈 Visualizations

Here is a visualization of the model's performance:

![Model Performance](https://github.com/Armanx200/Diabetes_Model/blob/main/Figure.png)

## 🔧 Requirements

This project requires the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

These dependencies are listed in `requirements.txt`.

## 📬 Contact

Feel free to reach out if you have any questions or suggestions:

- **GitHub**: [Armanx200](https://github.com/Armanx200)

---
