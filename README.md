# diabetes-prediction
End-to-end ML project predicting diabetes diagnosis using Logistic Regression and Random Forest with 91% recall on 96,146 patient records
# Diabetes Patient Prediction with Machine Learning

## 👤 Author
Ifeoma Mary-Ann James 

## 📌 Project Overview
An end-to-end machine learning project to predict the probability 
that a patient will be diagnosed with diabetes using real clinical 
data of 96,146 patients.

**The core question:**
> "Can we predict a diabetes diagnosis early enough to save lives?"

**Primary Metric: Recall**
In healthcare, missing a diabetic diagnosis is far more costly 
than a false alarm. Every percentage point of recall represents 
real patients correctly identified before complications set in.

---

## 🗂️ Project Structure

| File | Description |
|---|---|
| `project_1.ipynb` | Main analysis notebook |
| `diabetes_prediction_dataset.csv` | Dataset — 96,146 patients |
| `README.md` | Project documentation |

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Diabetes Prediction Dataset |
| Total Records | 96,146 patients |
| Features | age, bmi, blood_glucose_level, HbA1c_level, hypertension, heart_disease, smoking_history, gender |
| Target | diabetes (0 = No Diabetes, 1 = Diabetic) |
| Class Distribution | Non-Diabetic: 91.2% \| Diabetic: 8.8% |

---

## 🔍 Key EDA Findings

- **Blood glucose** is the strongest predictor (correlation = 0.42)
- **Diabetic patients** have mean glucose of 194 mg/dL vs 133 mg/dL
- **Mean HbA1c** of 6.93% in diabetics — crosses the 6.5% diagnostic boundary
- **Median age** of diabetic patients is 62 vs 40 in non-diabetic patients
- **32.3%** of heart disease patients are diabetic
- **28%** of hypertension patients are diabetic
- **Risk profile: Older Age + Higher BMI + High Blood Glucose + High HbA1c**

---

## 🤖 Models Built

### Model 1 — Logistic Regression (Baseline)
- class_weight='balanced' to handle imbalance
- Evaluated on training data as baseline

### Model 2 — Random Forest (Champion)
- n_estimators=100, max_depth=10, min_samples_leaf=5
- class_weight='balanced'
- 5-Fold Stratified K-Fold Cross-Validation

---

## 📈 Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Train Accuracy | High (overfitted) | 90% |
| Test Accuracy | — | 89% |
| Train Recall (Diabetic) | 99% (overfitted) | 93% |
| CV Mean Recall | — | 91% |
| Test Recall (Diabetic) | — | **91%** ✅ |

**Champion Model: Random Forest**
- 91% Recall on completely unseen test data
- Consistent with 91% CV mean — confirms no overfitting
- Blood glucose and HbA1c confirmed as top predictors

---

## 💡 Recommendations

- **SMOTE** — synthetic oversampling for better class balance
- **XGBoost / LightGBM** — expected to push recall beyond 93%
- **SHAP Values** — for clinical explainability and patient trust
- **Threshold Tuning** — optimise precision-recall balance
- **Concept Drift Monitoring** — retrain as patient patterns evolve

---

## 🛠️ Tools & Libraries

```python
pandas | numpy | scikit-learn | plotnine | 
matplotlib | seaborn | janitor
```

## 🚀 How To Run

1. Clone this repository
```bash
   git clone https://github.com/Ifeoma28/diabetes-prediction
```
2. Install requirements
```bash
   pip install pandas numpy scikit-learn plotnine 
               matplotlib seaborn pyjanitor
```
3. Add dataset to the same folder as the notebook
4. Open `project_1.ipynb` and run all cells

---

## 🔗 Connect
- LinkedIn: https://www.linkedin.com/in/ifeoma-james-4458321ba
- GitHub: https://github.com/Ifeoma28
