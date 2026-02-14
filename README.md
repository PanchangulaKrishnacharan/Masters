
# Machine Learning Assignment 2  
## Bank Marketing Classification

---

## a. Problem Statement

The objective of this project is to predict whether a customer will subscribe to a term deposit based on various demographic and campaign-related features.

This is a binary classification problem where:
- 0 represents "No Subscription"
- 1 represents "Subscription"

The goal is to implement multiple machine learning classification models, compare their performance using evaluation metrics, and identify the best-performing model.

---

## b. Dataset Description

Dataset Name: Bank Marketing Dataset  
Source: UCI Machine Learning Repository  
File Used: bank-full.csv  

Dataset Details:
- Total Instances: 41,188  
- Number of Features: 20 (before encoding)  
- Target Variable: y  
- Type: Binary Classification  

Feature Types:
- Demographic features (age, job, marital status, education)
- Financial features (housing loan, personal loan)
- Campaign-related features (contact type, month, duration, campaign count)

Preprocessing Steps:
1. Converted target variable (yes/no) into (1/0).
2. Applied one-hot encoding to categorical features.
3. Split dataset into 80% training and 20% testing.
4. Standardized features using StandardScaler.

The dataset is slightly imbalanced, as most customers did not subscribe.

---

## c. Models Used

The following six machine learning models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## Comparison Table of Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|------|------------|--------|------|------|
| Logistic Regression | 0.901 | 0.905 | 0.644 | 0.349 | 0.452 | 0.426 |
| Decision Tree | 0.878 | 0.713 | 0.478 | 0.499 | 0.488 | 0.419 |
| kNN | 0.894 | 0.808 | 0.586 | 0.309 | 0.405 | 0.374 |
| Naive Bayes | 0.864 | 0.809 | 0.428 | 0.488 | 0.456 | 0.380 |
| Random Forest (Ensemble) | 0.906 | 0.929 | 0.665 | 0.393 | 0.494 | 0.465 |
| XGBoost (Ensemble) | 0.908 | 0.929 | 0.635 | 0.503 | 0.561 | 0.515 |

---

## Observations on Model Performance

| ML Model Name | Observation |
|---------------|------------|
| Logistic Regression | Achieved high accuracy and good AUC, but recall is relatively low. Struggles with class imbalance. |
| Decision Tree | Moderate performance with lower AUC compared to ensemble models. May slightly overfit. |
| kNN | Good accuracy but low recall. Distance-based learning struggles with imbalance. |
| Naive Bayes | Lowest accuracy among models. Independence assumption limits performance. |
| Random Forest (Ensemble) | Strong performance with high AUC. Handles non-linear relationships well. |
| XGBoost (Ensemble) | Best overall model with highest F1 and MCC. Provides good balance between precision and recall. |

---

## Final Conclusion

Among all models, XGBoost achieved the best overall performance based on F1-score, MCC, and AUC. Although Random Forest also performed well, XGBoost provided a better balance between precision and recall, making it the most suitable model for this dataset.

Since the dataset is imbalanced, F1-score and MCC are more reliable indicators than accuracy alone.
