# Firm Innovation Analysis

**Identifying Key Predictors of Innovation Using Machine Learning Techniques**

## Problem Statement

Not all firms innovate, even with available resources. This project analyzes firm-level data to identify which characteristics most strongly predict whether a firm will innovate. The goal is to provide data-driven insights useful for research and policy decisions.

## Dataset

9,000 firms with 10 features:

| Feature | Description |
|---------|-------------|
| firm_size | Number of employees |
| firm_age | Years in operation |
| has_rd | R&D spending (Yes/No) |
| is_exporter | Exports products (Yes/No) |
| has_training | Provides employee training (Yes/No) |
| competition | Number of competitors (0-5) |
| foreign_owned | Foreign ownership (Yes/No) |
| quality_cert | Quality certification (Yes/No) |
| finance_obstacle | Access to finance obstacle level (0-4) |
| sector | Manufacturing / Retail / Services |

## Approach

1. **Data Cleaning & EDA**: Distribution analysis, innovation rates by firm characteristics, correlation heatmap
2. **Preprocessing**: Label encoding, train-test split (80/20), SMOTE for class imbalance
3. **Model Training**: Logistic Regression, Random Forest, XGBoost with 5-fold cross-validation
4. **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
5. **Feature Importance**: Ranked predictors of innovation

## Results

| Model | CV Accuracy |
|-------|-------------|
| Logistic Regression | 82% |
| Random Forest | 83% |
| XGBoost | 84% |

**Test Accuracy (XGBoost): ~80%**

## Key Finding

R&D spending is the strongest predictor of firm innovation, followed by firm size and export status.

## Interactive App

A Streamlit web app where you can enter any firm's details and get an instant prediction - Innovator or Non-Innovator with probability score and feature importance chart.

**[Try the Live App →](https://firm-innovation-analysis-kl8miw7ehjywxhgnabgdvc.streamlit.app/)**

## Tech Stack

Python, scikit-learn, XGBoost, Random Forest, Decision Trees, Logistic Regression, SMOTE, Streamlit, Pandas, NumPy, Matplotlib, Seaborn

## Project Structure

```
├── app.py                          # Streamlit interactive app
├── Firm_Innovation_Analysis.ipynb  # Full analysis notebook
├── requirements.txt                # Dependencies
└── README.md
```

