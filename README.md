<p align="center">
  <img width="900" alt="AI Credit Risk Banner" src="https://img.shields.io/badge/Machine%20Learning-Credit%20Risk%20Intelligence-darkblue?style=for-the-badge">
</p>

# AI Credit Risk Intelligence System

## End-to-End Financial Risk Analytics • Explainable AI • Responsible Lending Intelligence

An end-to-end machine learning project for loan default prediction using credit risk analytics, advanced feature engineering, imbalance-aware modeling, SHAP explainability, fairness auditing, and a deployment-ready Streamlit application.

---

## Project Overview

This project predicts whether a loan applicant is likely to be a **Good Credit** or **Bad Credit** customer using the German Credit dataset.  
The goal is not only to build a predictive model, but also to create a **business-focused credit risk intelligence system** that supports transparent and responsible lending decisions.

---

## Key Features

- End-to-end ML pipeline
- Banking-code decoding into readable financial categories
- Exploratory data analysis
- Advanced borrower risk feature engineering
- 10-model benchmark comparison
- Imbalance-aware modeling using SMOTE and Balanced Random Forest
- Threshold optimization for banking loss minimization
- SHAP explainability
- Responsible AI fairness audit
- Streamlit risk prediction application

---

## Dataset

**Source:** UCI Statlog German Credit Data

- Total Applicants: 1000
- Original Features: 20
- Engineered Features: 34
- Target Classes:
  - Good Credit
  - Bad Credit

---

## Machine Learning Workflow

1. Raw banking code decoding  
2. Exploratory Data Analysis (EDA)  
3. Professional feature engineering  
4. Data preprocessing and scaling  
5. Baseline advanced benchmark models  
6. Imbalance-aware specialized risk models  
7. Comprehensive 10-model comparison  
8. Lending threshold optimization  
9. Explainable AI using SHAP  
10. Responsible AI fairness audit  
11. Deployment artifact generation  

---

## Final Model Performance

### Best Risk Ranking Model
**Extra Trees Classifier**  
ROC-AUC: **0.7971**

### Best Bad Borrower Detection Model
**Balanced Random Forest**  
Recall: **0.6667**

### Optimal Lending Threshold
**0.30 threshold** produced the lowest expected bank portfolio loss under asymmetric financial cost assumptions.

---

## Explainable AI Findings

Top global drivers of bad credit prediction:

- Low checking liquidity
- Weak financial stability
- High repayment pressure
- Low savings strength
- Long loan duration
- High loan amount indicators
- Credit history behavior

---

## Responsible AI Findings

Fairness analysis showed that:

- younger applicants were flagged risky more frequently,
- some personal-status borrower groups showed elevated predicted risk rates,
- subgroup fairness monitoring is required before production deployment.

---

## Folder Structure

AI-Credit-Risk-Intelligence-System/  
│  
├── app.py  
├── requirements.txt  
├── README.md  
│  
├── data/  
│   └── german.data  
│  
├── models/  
│   ├── balanced_random_forest_credit_model.pkl  
│   ├── feature_columns.pkl  
│   ├── credit_scaler.pkl  
│   └── credit_label_encoders.pkl  
│  
└── notebooks/  
&nbsp;&nbsp;&nbsp;&nbsp;└── Credit-Risk-Scoring.ipynb  

---

## How to Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py
