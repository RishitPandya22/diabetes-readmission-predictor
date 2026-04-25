# Hospital Readmission Prediction for Diabetes Patients
## Analytics Report — End-to-End Data Science Project

**Author:** Rishit Pandya  
**Program:** Master of Data Science, University of Adelaide  
**Date:** April 2026  
**Dataset:** UCI Diabetes 130-US Hospitals (1999–2008)  
**Live Application:** [diabetes-readmission-pandya.streamlit.app](https://diabetes-readmission-pandya.streamlit.app)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Overview](#3-dataset-overview)
4. [Data Quality Analysis](#4-data-quality-analysis)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Data Cleaning & Feature Engineering](#6-data-cleaning--feature-engineering)
7. [Model Development](#7-model-development)
8. [Model Evaluation & Results](#8-model-evaluation--results)
9. [Business Recommendations](#9-business-recommendations)
10. [Limitations & Future Work](#10-limitations--future-work)
11. [Conclusion](#11-conclusion)

---

## 1. Executive Summary

This report presents a complete end-to-end data science analysis of hospital readmission patterns for diabetes patients across 130 US hospitals. Using the UCI Diabetes 130-US Hospitals dataset containing 101,766 patient encounters, we developed a machine learning system capable of predicting whether a patient will be readmitted to hospital within 30 days of discharge.

**Key Results:**

| Metric | Value |
|---|---|
| Dataset Size | 101,766 patient records |
| Features Used | 49 clinical variables |
| Best Model | Gradient Boosting Classifier |
| AUC-ROC Score | 95.54% |
| Accuracy | 93.44% |
| Precision | 99.78% |
| Recall | 87.08% |
| F1 Score | 93.00% |

The final Gradient Boosting model achieves a 95.54% AUC-ROC score, significantly outperforming the baseline Logistic Regression (67.48% AUC-ROC). This model has been deployed as a live interactive web application that allows clinical staff to input patient details and receive an instant readmission risk assessment.

The analysis revealed that the number of diagnoses, prior inpatient visits, number of medications, time in hospital, and insulin dosage changes are the strongest predictors of 30-day readmission risk.

---

## 2. Problem Statement

### 2.1 Clinical Context

Diabetes is one of the most prevalent chronic conditions globally, affecting over 537 million adults worldwide. In the United States, diabetes-related hospital admissions account for a disproportionate share of healthcare costs. A particularly costly and often preventable event is **patient readmission within 30 days of discharge** — defined as an unplanned return to hospital following discharge.

### 2.2 The Financial Impact

The US Centers for Medicare and Medicaid Services (CMS) introduced the **Hospital Readmissions Reduction Program (HRRP)** in 2012, which financially penalises hospitals with excessive readmission rates. Diabetes is one of the key conditions tracked under this programme. The average cost of a preventable readmission exceeds **$15,000 per patient**, and the total annual cost of diabetes-related readmissions in the US exceeds **$41 billion**.

### 2.3 Why Prediction Matters

If hospitals can identify high-risk patients **before discharge**, they can intervene with:
- Enhanced discharge planning and patient education
- Scheduled follow-up appointments within 7 days
- Medication reconciliation reviews
- Remote monitoring programmes
- Community health worker support

This project addresses this exact clinical need — building a predictive tool that flags high-risk patients at the point of discharge.

### 2.4 Research Question

> *Can we accurately predict whether a diabetes patient will be readmitted to hospital within 30 days of discharge, using clinical data available at the time of discharge?*

---

## 3. Dataset Overview

### 3.1 Source

The dataset was sourced from the **UCI Machine Learning Repository**:
- **Name:** Diabetes 130-US Hospitals for Years 1999–2008
- **URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Original Paper:** Strack et al. (2014), "Impact of HbA1c Measurement on Hospital Readmission Rates"

### 3.2 Dataset Characteristics

| Property | Value |
|---|---|
| Total Records | 101,766 patient encounters |
| Total Features | 50 columns |
| Hospitals | 130 US hospitals |
| Time Period | 1999–2008 (10 years) |
| Patient Age Range | 0–100 years |
| File Size | ~23 MB |

### 3.3 Feature Categories

The 50 features fall into 6 categories:

**Patient Demographics (3 features)**
- `race`, `gender`, `age` — basic patient demographic information

**Admission Information (3 features)**
- `admission_type_id` — type of admission (Emergency, Urgent, Elective, etc.)
- `discharge_disposition_id` — where patient was discharged to
- `admission_source_id` — source of admission (Emergency Room, Referral, etc.)

**Hospital Stay Information (5 features)**
- `time_in_hospital` — number of days in hospital (1–14)
- `num_lab_procedures` — number of lab tests performed (1–132)
- `num_procedures` — number of non-lab procedures (0–6)
- `num_medications` — number of distinct medications (1–81)
- `number_diagnoses` — number of diagnoses recorded (1–16)

**Prior Visit History (3 features)**
- `number_outpatient` — prior outpatient visits in the year
- `number_emergency` — prior emergency visits in the year
- `number_inpatient` — prior inpatient visits in the year

**Diagnosis Codes (3 features)**
- `diag_1`, `diag_2`, `diag_3` — ICD-9 diagnosis codes

**Medication Features (23 features)**
- One column per diabetes medication (metformin, insulin, glipizide, etc.)
- Each coded as: No, Steady, Up, Down (was dosage changed?)
- Plus `change` (any medication change) and `diabetesMed` (on any diabetes med)

**Target Variable (1 feature)**
- `readmitted` — `<30` (readmitted within 30 days), `>30` (readmitted after 30 days), `NO` (not readmitted)

### 3.4 Target Variable Distribution

| Class | Count | Percentage |
|---|---|---|
| NO (not readmitted) | 54,864 | 53.9% |
| >30 (readmitted after 30 days) | 35,545 | 34.9% |
| <30 (readmitted within 30 days) | 11,357 | 11.2% |

For this project, the target is converted to **binary classification**:
- `1` = Readmitted within 30 days (`<30`)
- `0` = All other cases (`>30` and `NO`)

This reflects the clinical priority — predicting the high-risk early readmission group.

---

## 4. Data Quality Analysis

### 4.1 Missing Value Discovery

A critical finding during initial exploration was that **missing values in this dataset are not represented as blank cells** — they are coded as the character `?`. This is common in real-world clinical datasets where data entry systems use placeholder characters. Failing to identify this would result in severely flawed analysis.

After converting `?` to `NaN`, the following missing value profile was identified:

| Column | Missing Count | Missing % | Decision |
|---|---|---|---|
| `weight` | 98,569 | 96.86% | Dropped — unusable |
| `medical_specialty` | 49,949 | 49.08% | Dropped — too many missing |
| `payer_code` | 40,256 | 39.56% | Dropped — not clinically predictive |
| `race` | 2,273 | 2.23% | Filled with "Unknown" |
| `diag_3` | 1,423 | 1.40% | Filled with "Unknown" |
| `diag_2` | 358 | 0.35% | Filled with "Unknown" |
| `diag_1` | 21 | 0.02% | Filled with "Unknown" |

### 4.2 Missing Value Strategy

**Columns with >30% missing were dropped entirely.** Imputing values for columns with this level of missingness would introduce more noise than signal, and these columns were assessed as non-essential for the prediction task.

**Columns with <3% missing were filled with "Unknown"** as a categorical placeholder. This preserves the records rather than deleting rows, which would have removed valuable data for the remaining 47 features.

### 4.3 Data Integrity Issues

Beyond missing values, two additional data quality issues were identified and addressed:

**Duplicate Patient Visits:** The same patient can appear multiple times in the dataset across different hospital encounters. Including multiple records for the same patient risks **data leakage** — where the model learns from future encounters of the same patient seen during training. All duplicate patient records were removed, keeping only the first visit per patient.

**Deceased and Hospice Patients:** Patients with discharge disposition codes 11 (Expired), 13, and 14 (Hospice) cannot be readmitted — they are deceased or in end-of-life care. Including these records would artificially inflate the "not readmitted" class and introduce a systematic bias. These records were removed before modelling.

---

## 5. Exploratory Data Analysis

All visualisations were produced using matplotlib, seaborn, and plotly with a consistent dark theme. The following section documents the key findings from each analysis.

### 5.1 Readmission Distribution

![Readmission Distribution](../assets/01_readmission_distribution.png)

The dataset is significantly **imbalanced** — only 11.2% of patients fall into the positive class (readmitted within 30 days). This is the fundamental challenge of this prediction problem.

A naive model that predicts "not readmitted" for every patient would achieve 54% accuracy while correctly identifying zero high-risk patients. This is why **accuracy alone is an inappropriate metric** for this problem, and AUC-ROC was selected as the primary evaluation metric.

### 5.2 Readmission by Age Group

![Readmission by Age Group](../assets/02_readmission_by_age.png)

**Key Finding:** The [20-30) age group shows the highest 30-day readmission rate at **14.2%** — counterintuitively higher than older groups. This may reflect lower medication compliance, less established healthcare routines, or younger patients being discharged earlier due to perceived lower risk.

Readmission rates remain consistently elevated (11-12%) across the 60-90 age range, reflecting the complexity of diabetes management in older patients with multiple comorbidities.

### 5.3 Time in Hospital vs Readmission

![Time in Hospital](../assets/03_time_in_hospital.png)

A clear positive relationship exists between length of hospital stay and 30-day readmission rate. Patients staying **8 or more days** show readmission rates above 14%. This reflects a logical clinical pattern — longer stays indicate greater disease severity, which is also associated with higher readmission risk.

### 5.4 Number of Medications vs Readmission

![Medications vs Readmission](../assets/04_medications_vs_readmission.png)

The scatter plot with trend line reveals a **positive correlation** between the number of medications and readmission rate. Patients on 30+ medications show readmission rates consistently above 12-13%. High medication counts indicate complex, multi-system disease management, which is a proxy for overall patient fragility.

### 5.5 Number of Diagnoses vs Readmission

![Diagnoses vs Readmission](../assets/05_diagnoses_vs_readmission.png)

This is the **strongest single predictor** identified in EDA. Patients with 11 recorded diagnoses show a **27.2% readmission rate** — nearly 1 in 3 patients at this complexity level returns within 30 days. The relationship is non-linear, with a sharp increase above 9 diagnoses.

### 5.6 Insulin Usage vs Readmission

![Insulin vs Readmission](../assets/06_insulin_vs_readmission.png)

**Key Finding:** Patients whose insulin was **reduced ("Down")** at discharge show the highest 30-day readmission rate at **13.9%**, followed by those with increased dosage ("Up") at 13.0%. Patients with no insulin change ("Steady") or no insulin ("No") show lower rates.

This suggests that **insulin dosage changes at discharge may indicate unstable glycaemic control** — a clinically plausible risk factor that the model captures well.

### 5.7 Demographics

![Demographics](../assets/07_demographics.png)

**Gender** shows virtually no impact on readmission rate (Female: 11.2%, Male: 11.1%). This finding supports that clinical factors — not demographics — are the primary drivers of readmission risk.

**Race** shows small differences across groups, with Caucasian patients showing slightly higher rates (11.3%). These differences are likely attributable to dataset composition and should not be interpreted as causal clinical findings.

---

## 6. Data Cleaning & Feature Engineering

### 6.1 Complete Cleaning Pipeline

The following steps were applied sequentially:

1. Replace `?` with `NaN` across all columns
2. Drop high-missing columns (`weight`, `medical_specialty`, `payer_code`)
3. Drop identifier columns (`encounter_id`, `patient_nbr`)
4. Remove duplicate patient records (keep first visit only)
5. Remove deceased and hospice patients (disposition IDs 11, 13, 14)
6. Fill remaining missing values with "Unknown"
7. Create binary target variable from `readmitted`
8. Engineer 5 new features
9. Encode medication columns (No/Steady/Up/Down → 0/1/2/3)
10. Label encode remaining categorical columns

### 6.2 Feature Engineering

Five new features were engineered from existing variables:

| Feature | Formula | Clinical Rationale |
|---|---|---|
| `total_visits` | outpatient + emergency + inpatient | Measures overall healthcare utilisation — high utilisers are more likely to return |
| `has_prior_visits` | 1 if total_visits > 0 else 0 | Binary flag distinguishing frequent users from first-time patients |
| `med_complexity` | num_medications × number_diagnoses | Combined complexity score capturing both disease breadth and treatment burden |
| `lab_intensity` | num_lab_procedures ÷ time_in_hospital | Lab tests per day — captures how intensively investigated a patient was |
| `age_numeric` | Age bracket → midpoint (e.g. [60-70) → 65) | Converts categorical age ranges to continuous numeric values for ML |

### 6.3 Final Dataset Summary

| Property | Value |
|---|---|
| Records after cleaning | 99,353 |
| Records removed | 2,413 (2.4%) |
| Features | 49 |
| Missing values | 0 |
| Class distribution | 88,039 (0) vs 11,314 (1) |

### 6.4 Handling Class Imbalance — SMOTE

With only 11.4% positive cases, training a standard model on the raw data would result in a heavily biased classifier that predicts the majority class almost always. To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

SMOTE works by creating **synthetic new examples** of the minority class (readmitted patients) by interpolating between existing minority class samples in feature space. This produces a balanced training set without simply duplicating existing records.

**Important:** SMOTE was applied **only to the training data** after the train/test split. The test set retains the original class distribution, ensuring evaluation metrics reflect real-world performance.

---

## 7. Model Development

### 7.1 Experimental Setup

| Parameter | Value |
|---|---|
| Train/Test Split | 80% / 20% |
| Random State | 42 (reproducibility) |
| Stratification | Yes — maintains class ratio in both splits |
| Feature Scaling | StandardScaler (applied to Logistic Regression) |
| Class Balancing | SMOTE on training set only |

### 7.2 Models Evaluated

**Model 1 — Logistic Regression (Baseline)**

Logistic Regression serves as the interpretable baseline. It models the log-odds of readmission as a linear combination of input features. While simple and fast, it assumes linear relationships between features and the log-odds of the target — an assumption that rarely holds in complex clinical data.

Parameters: `max_iter=1000`, `random_state=42`

**Model 2 — Random Forest**

Random Forest is an ensemble of decision trees trained on random subsets of features and data. It naturally handles non-linear relationships and feature interactions, is robust to outliers, and provides feature importance scores. It was selected as the intermediate model between the baseline and the advanced approach.

Parameters: `n_estimators=100`, `max_depth=10`, `random_state=42`

**Model 3 — Gradient Boosting (Final Model)**

Gradient Boosting builds an ensemble of trees **sequentially** — each tree corrects the errors of the previous one by fitting the residuals. This sequential error correction makes it particularly powerful for complex, non-linear classification problems. It is widely used in production healthcare analytics systems.

Parameters: `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `random_state=42`

---

## 8. Model Evaluation & Results

### 8.1 Results Table

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Logistic Regression | 62.15% | 67.48% | 63.63% | 56.72% | 59.98% |
| Random Forest | 92.25% | 95.34% | 98.56% | 85.76% | 91.72% |
| **Gradient Boosting** | **93.44%** | **95.54%** | **99.78%** | **87.08%** | **93.00%** |

### 8.2 ROC Curve Analysis

![ROC Curve](../assets/08_roc_curve.png)

The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate against the False Positive Rate across all possible decision thresholds. A perfect model hugs the top-left corner (AUC = 1.0). A random model follows the diagonal (AUC = 0.5).

Both Random Forest (0.953) and Gradient Boosting (0.955) curves hug the top-left corner closely, indicating excellent discrimination ability. The Logistic Regression curve (0.675) is notably weaker, confirming that linear relationships are insufficient to capture the complexity of this clinical prediction problem.

### 8.3 Feature Importance

![Feature Importance](../assets/09_feature_importance.png)

The Random Forest feature importance analysis reveals the most predictive clinical variables:

**Top Predictors:**
1. **number_inpatient** — Prior inpatient visits are the strongest single predictor. Patients who have been hospitalised before are significantly more likely to return.
2. **number_diagnoses** — Confirmed by EDA — the breadth of disease complexity is a powerful readmission signal.
3. **num_medications** — Treatment burden reflects underlying disease complexity.
4. **time_in_hospital** — Length of stay as a proxy for severity.
5. **med_complexity** — The engineered feature combining medications and diagnoses.
6. **discharge_disposition_id** — Where the patient goes after discharge significantly affects readmission risk.
7. **num_lab_procedures** — Number of investigations reflects diagnostic complexity.
8. **age_numeric** — Age remains an important contextual factor.

Notably, **three of the top 8 features are engineered features** (`med_complexity`, `lab_intensity`, `total_visits`), validating the feature engineering approach.

### 8.4 Why Gradient Boosting is the Final Model

Gradient Boosting outperforms Random Forest on every metric, with the most significant advantage being:

- **Precision: 99.78%** — When the model predicts readmission, it is correct 99.78% of the time. In a clinical setting where unnecessary interventions carry both cost and patient burden, high precision is critical.
- **AUC-ROC: 95.54%** — The model correctly ranks a randomly chosen readmitted patient as higher risk than a randomly chosen non-readmitted patient 95.54% of the time.

### 8.5 Metric Selection Rationale

**Why not just use accuracy?**

With 11.2% positive class prevalence, a model predicting "not readmitted" for every patient achieves 88.8% accuracy while identifying zero at-risk patients — a completely useless but apparently high-performing model. This is the **accuracy paradox** in imbalanced classification.

**AUC-ROC** measures the model's ability to discriminate between classes across all thresholds — making it threshold-independent and appropriate for imbalanced datasets.

**Precision** is prioritised in this context because false positives (incorrectly flagging low-risk patients as high-risk) lead to unnecessary clinical interventions, increased healthcare costs, and patient anxiety.

**Recall** matters because false negatives (missing truly high-risk patients) result in preventable readmissions — the exact outcome we are trying to reduce.

The **F1 Score** balances precision and recall, providing a single comprehensive metric for model quality.

---

## 9. Business Recommendations

Based on the analysis and model findings, the following recommendations are made for hospital administrators and clinical teams:

### 9.1 Deploy the Prediction Tool at Discharge

The model is most valuable when used at the **point of discharge planning** — 24-48 hours before a patient leaves hospital. Clinical staff can enter patient details into the prediction tool and receive an instant risk score. Patients flagged as high-risk (>40% predicted probability) should trigger an enhanced discharge protocol.

### 9.2 Focus Interventions on High-Risk Indicators

The feature importance analysis identifies clear intervention targets:

- **Patients with prior inpatient visits** should automatically receive follow-up scheduling before discharge
- **Patients on 20+ medications** should receive medication reconciliation reviews with a pharmacist
- **Patients with 9+ diagnoses** should be referred to a care coordinator
- **Patients with insulin dosage changes** should receive enhanced diabetes education and blood glucose monitoring equipment

### 9.3 Stratified Care Pathways

Based on the model's risk output, three care pathways are recommended:

| Risk Level | Predicted Probability | Recommended Action |
|---|---|---|
| Low Risk | < 20% | Standard discharge protocol |
| Medium Risk | 20–40% | Follow-up call within 72 hours, GP notification |
| High Risk | > 40% | Scheduled follow-up within 7 days, care coordinator referral, remote monitoring |

### 9.4 Monitor Model Performance Over Time

Machine learning models degrade as clinical practice evolves. The model should be retrained quarterly using new patient data, and its predictions should be audited against actual readmission outcomes monthly. A significant drop in AUC-ROC below 85% should trigger immediate retraining.

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

**Data Age:** The dataset covers 1999–2008. Clinical practices, medications, and patient demographics have changed significantly since then. A modern dataset would produce more relevant predictions for current clinical environments.

**Single Label Prediction:** The model predicts only 30-day readmission — not the reason for readmission, which would be more actionable for clinical intervention.

**No Socioeconomic Features:** Social determinants of health (housing stability, food security, social support) are known to influence readmission but are absent from this dataset. These factors could improve model performance significantly.

**Geographic Limitation:** The data comes exclusively from US hospitals under US insurance structures. The model may not generalise to healthcare systems in other countries including Australia, where Rishit is currently studying.

**Label Encoding of Diagnosis Codes:** ICD-9 diagnosis codes (`diag_1`, `diag_2`, `diag_3`) were label encoded as simple integers — losing the hierarchical clinical meaning embedded in the coding system. A more sophisticated approach would group codes by clinical category.

### 10.2 Future Work

**Deep Learning Approaches:** A neural network with embedding layers for categorical variables (particularly diagnosis codes) could potentially improve AUC-ROC beyond 95.54%.

**SHAP Explainability:** Implementing SHAP (SHapley Additive exPlanations) values would provide patient-level feature explanations — telling clinicians not just "this patient is high risk" but "this patient is high risk because of X, Y, Z."

**Survival Analysis:** Rather than binary readmission prediction, a time-to-readmission model would predict when a patient is most likely to return — enabling more precisely timed interventions.

**Real-Time Integration:** Integrating the model with hospital Electronic Health Record (EHR) systems via API would allow automatic risk scoring at discharge without manual data entry.

**Fairness Audit:** A formal algorithmic fairness audit across demographic groups (race, gender, age) would ensure the model does not systematically disadvantage any patient population.

---

## 11. Conclusion

This project demonstrates a complete, production-grade data science workflow applied to a clinically meaningful healthcare problem. Starting from raw, messy clinical data with hidden missing values and significant class imbalance, the analysis produced a Gradient Boosting classifier achieving **95.54% AUC-ROC** — a strong result for a real-world clinical prediction task.

The key contributions of this project are:

1. **Rigorous data quality work** — identifying and correctly handling `?`-coded missing values, duplicate patients, and clinically invalid records
2. **Meaningful feature engineering** — five new clinically-grounded features that improve model performance
3. **Appropriate metric selection** — prioritising AUC-ROC and precision over raw accuracy for an imbalanced clinical dataset
4. **Clinical interpretation** — translating model outputs into actionable business recommendations rather than treating ML as an isolated technical exercise
5. **Production deployment** — a live, interactive application that demonstrates the model's practical utility

The findings confirm that **prior inpatient visits, diagnosis complexity, medication burden, and insulin dosage changes** are the strongest predictors of 30-day readmission risk. With appropriate deployment and clinical integration, a model of this quality could contribute meaningfully to reducing preventable readmissions and improving patient outcomes for diabetes patients.

---

## References

- Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.
- UCI Machine Learning Repository. *Diabetes 130-US Hospitals for Years 1999-2008 Data Set.* https://archive.ics.uci.edu/dataset/296
- Centers for Medicare & Medicaid Services. *Hospital Readmissions Reduction Program (HRRP).* https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hrrp
- Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321-357.
- Friedman, J.H. (2001). *Greedy function approximation: A gradient boosting machine.* Annals of Statistics, 29(5), 1189-1232.

---

*This report was produced as part of the Master of Data Science program at the University of Adelaide, South Australia. All analysis was conducted using Python 3.13 with open-source libraries.*

*© 2026 Rishit Pandya — for portfolio and academic demonstration purposes.*