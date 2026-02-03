# Creditworthiness Prediction ML Project

## 🏦 Project Overview

This project predicts an individual's **creditworthiness** using past financial data. The goal is to help banks or financial institutions identify high-risk borrowers and minimize potential loan defaults.  

The model uses **Logistic Regression** with **class-weight balancing** and **custom probability threshold tuning** to make risk-aware predictions.

---

## 🎯 Objective

- Predict whether a loan applicant is **High Risk** or **Low Risk**
- Optimize for **Recall** to minimize the chance of approving risky customers
- Maintain **reasonable Precision** to avoid unnecessarily rejecting good customers

---

## 🛠 Technologies & Tools Used

- **Programming Language:** Python  
- **Machine Learning:** scikit-learn (Logistic Regression, class weighting, scaling)  
- **Data Handling:** pandas, numpy  
- **Web App Interface:** Streamlit  
- **Model Persistence:** pickle  

---


---

## 🧠 Approach & Methodology

1. **Data Preprocessing**  
   - Cleaned and encoded categorical features using one-hot encoding  
   - Scaled numerical features using `StandardScaler`  

2. **Handling Imbalanced Dataset**  
   - Applied `class_weight='balanced'` in Logistic Regression to give more importance to high-risk borrowers  

3. **Model Training**  
   - Trained **Logistic Regression** as baseline  
   - Compared with Decision Tree and Random Forest  
   - Selected Logistic Regression for its balance between Recall, F1-Score, and ROC-AUC  

4. **Threshold Tuning**  
   - Default prediction threshold (0.5) replaced with **custom threshold = 0.35**  
   - Optimized Recall while keeping Precision reasonable  

5. **Evaluation Metrics**  
   - **Precision, Recall, F1-Score, ROC-AUC**  
   - Focused on **Recall** as missing risky customers is costly in credit risk  

---

