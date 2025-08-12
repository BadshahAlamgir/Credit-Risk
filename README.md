# 💳 Credit Risk Prediction & Management

A **machine learning project** to predict the probability of loan default using **credit and demographic data**.  
This project applies **data preprocessing, EDA, feature engineering, class balancing, and model building** to create an accurate and interpretable credit risk classifier.

---

## 📌 Project Overview

- **Goal:** Predict whether a borrower will default on a loan (`default_on_file`)
- **Techniques Used:**
  - Data Cleaning & Missing Value Handling
  - Exploratory Data Analysis (EDA) with visualizations
  - Class balancing using **SMOTE**
  - Feature scaling and transformation
  - Model training with **Random Forest**
  - Model evaluation using **ROC-AUC** and classification metrics

---

## 📂 Dataset

- **Rows:** ~28,638  
- **Columns:** 12 (age, income, loan details, credit history, etc.)
- **Target:** `default_on_file` (1 = default, 0 = no default)

**Example Features:**
| Feature | Description |
|---------|-------------|
| `age` | Age of borrower |
| `income` | Annual income |
| `home_ownership` | Home ownership status |
| `emp_length` | Employment length (years) |
| `loan_intent` | Purpose of loan |
| `loan_grade` | Credit grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as % of income |
| `cred_hist_length` | Length of credit history |

---

## 🛠 Tech Stack

- **Language:** Python (Pandas, NumPy)
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, RandomForestClassifier
- **Class Balancing:** imbalanced-learn (SMOTE)

---

## 🔍 Exploratory Data Analysis (EDA)

**Target Variable Distribution (Before Balancing)**  
![Target Variable Distribution](images/target_distribution.png)

**Feature Distributions**  
- Age Distribution  
![Age Distribution](images/age_distribution.png)

- Income Distribution  
![Income Distribution](images/income_distribution.png)

**Default Rate by Loan Intent**  
![Loan Intent vs Default Rate](images/loan_intent_default.png)

**Correlation Heatmap**  
![Correlation Heatmap](images/correlation_heatmap.png)

**Loan Amount vs Interest Rate**  
![Loan Amount vs Interest Rate](images/loan_vs_rate.png)

**Default Rate by Loan Grade**  
![Loan Grade vs Default Rate](images/loan_grade_risk.png)

---

## ⚙️ Data Preprocessing

1. **Missing Value Handling** – Dropped missing entries
2. **Encoding** – Converted categorical variables to numeric (Label Encoding / One-Hot Encoding)
3. **Balancing Classes** – Applied **SMOTE** to address target imbalance
4. **Feature Scaling** – StandardScaler for numeric features

---

## 🤖 Model Building

- **Model Used:** Random Forest Classifier
- **Train/Test Split:** 80/20
- **Hyperparameters:** Default (can be tuned for optimization)

**Model Evaluation:**
- **ROC-AUC Score:** ~0.90+
- **Classification Report:**
  - Precision, Recall, F1-Score for both classes

---

## 📊 Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.90+** |
| Precision (Default=1) | High |
| Recall (Default=1) | High |
| Balanced Accuracy | High |

✅ **Random Forest** achieved strong predictive performance with balanced precision and recall.

---

## 💡 Key Insights

- Borrowers with **higher loan interest rates** and **lower credit grades** show significantly higher default risk.
- Certain **loan purposes** (like small business loans) have higher average default rates.
- **Income-to-loan ratio** is a strong predictor of repayment ability.

---

## 📦 Installation & Usage

**Requirements:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
