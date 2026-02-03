# Disease Prediction System 🏥

## Project Overview
The **Disease Prediction System** is a machine learning project aimed at predicting whether a patient is likely to have a specific disease (e.g., heart disease) based on their health-related attributes. This system uses historical patient data to train a model and can provide predictions along with the probability of the disease.

---

## Objective
- To **analyze patient health data** and predict disease occurrence.  
- To **assist healthcare providers** by offering preliminary risk assessments.  
- To **deploy a user-friendly interface** using Streamlit for interactive predictions.

---

## Dataset
The dataset used contains various health parameters of patients, such as:

- Age  
- Sex  
- Blood pressure  
- Cholesterol levels  
- Heart rate  
- Other relevant clinical measurements  

**Source:** Provided in CSV format (`german.data` or similar).  
**Note:** Ensure that all sensitive data is anonymized.

---


---

## Tools & Technologies Used
- **Programming Language:** Python 3.x  
- **Data Analysis & Visualization:** Pandas, NumPy, Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn (Logistic Regression / Random Forest / XGBoost)  
- **Model Persistence:** Pickle  
- **Deployment:** Streamlit  
- **Environment:** Anaconda / Jupyter Notebook  

---

## Methodology

1. **Data Loading & Cleaning:**  
   - Load dataset using Pandas.  
   - Handle missing values and encode categorical variables.  

2. **Exploratory Data Analysis (EDA):**  
   - Visualize distributions of features.  
   - Understand correlations between features and disease occurrence.

3. **Feature Scaling:**  
   - Apply `StandardScaler` to normalize features for model training.

4. **Model Selection & Training:**  
   - Train a classification model (e.g., Logistic Regression).  
   - Evaluate model performance using accuracy, precision, recall, and F1-score.  

5. **Model Saving:**  
   - Save the trained model using `pickle` for later deployment.  
   - Save the scaler and feature columns to ensure consistency during prediction.

---

## Model Performance
| Metric        | Score |
|---------------|-------|
| Accuracy      | 85%   |
| Precision     | 82%   |
| Recall        | 80%   |
| F1-Score      | 81%   |

> These metrics may vary depending on the dataset and model choice.

---



