# Semiconductor Manufacturing Process Classification

## 📌 Project Overview

This project uses machine learning to predict whether a semiconductor manufacturing process will **Pass or Fail** based on sensor measurements.

In semiconductor manufacturing, detecting process failures early is important for maintaining product quality, reducing defects, and improving manufacturing efficiency.

The project focuses on data preprocessing, handling class imbalance, feature scaling, model comparison, and hyperparameter tuning.

---

## 🎯 Objectives

- Explore and understand the semiconductor manufacturing dataset
- Handle missing values and clean the data
- Perform exploratory data analysis (EDA)
- Handle class imbalance using SMOTE
- Scale features using StandardScaler
- Train and compare multiple machine learning models
- Tune the best-performing model using GridSearchCV
- Evaluate model performance
- Save the optimized model for future predictions

---

## 📊 Dataset

The dataset contains semiconductor manufacturing sensor measurements.

- **Samples:** 1,567
- **Features:** 591 sensor/process features
- **Target:** Pass/Fail
- **Pass:** `-1`
- **Fail:** `1`

The target variable is highly imbalanced, with significantly fewer failure samples than pass samples.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

---

## 🔄 Machine Learning Workflow

The project follows these main steps:

1. Data Loading
2. Data Exploration
3. Data Cleaning
4. Missing Value Analysis
5. Exploratory Data Analysis
6. Train-Test Split
7. Class Imbalance Handling using SMOTE
8. Feature Scaling using StandardScaler
9. Model Training
10. Model Evaluation
11. Cross-Validation
12. Hyperparameter Tuning using GridSearchCV
13. Final Model Selection

---

## ⚖️ Handling Class Imbalance

The dataset contains a significant imbalance between Pass and Fail samples.

SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the training data.

Before SMOTE:

- Pass: 1,170
- Fail: 83

After SMOTE:

- Pass: 1,170
- Fail: 1,170

This helps the models learn from the minority Fail class more effectively.

---

## 🤖 Machine Learning Models

Three supervised learning algorithms were trained and evaluated:

### 1. Random Forest

A tree-based ensemble learning algorithm that combines multiple decision trees to make predictions.

### 2. Gaussian Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem.

### 3. Support Vector Machine (SVM)

A classification algorithm that finds an optimal decision boundary between different classes.

The SVM model was further optimized using **GridSearchCV**.

---

## 🔧 Feature Scaling

`StandardScaler` was used to standardize the features before training the models.

This is particularly important for algorithms such as SVM because features with different scales can negatively affect model performance.

---

## 🎯 Hyperparameter Tuning

GridSearchCV was used to optimize the SVM model.

The parameters explored included:

- `C`
- `gamma`
- `kernel`

The tuning process used **5-fold cross-validation** with accuracy as the scoring metric.

---

## 📈 Model Performance

| Model | Test Accuracy | Cross-Validation Accuracy |
|---|---:|---:|
| Random Forest | 92.99% | 99.36% |
| Naive Bayes | 25.48% | 60.30% |
| SVM | 93.31% | 99.57% |
| **Optimized SVM** | **93.63%** | **99.62%** |

### 🏆 Best Model

The **Optimized SVM** achieved the highest test accuracy of **93.63%** and a cross-validation accuracy of **99.62%**.

---

## 📊 Results

The model comparison shows that SVM performed better than Random Forest and Gaussian Naive Bayes on this dataset.

After hyperparameter tuning, the Optimized SVM achieved a small improvement over the original SVM:

- SVM Test Accuracy: **93.31%**
- Optimized SVM Test Accuracy: **93.63%**

The optimized SVM was therefore selected as the final model.

---

## ⚠️ Important Note

Although the overall accuracy of the models is high, the dataset is highly imbalanced.

Therefore, accuracy alone should not be considered sufficient for evaluating failure detection. Precision, recall, F1-score, and the confusion matrix are also important when evaluating the model's ability to identify actual manufacturing failures.

---

## 📁 Project Structure

```text
Semiconductor-Manufacturing-Classification/
│
├── Signal_Data_Major_Project.ipynb
├── README.md
├── data/
│   └── signal-data.csv
│
└── models/
    └── optimized_svm.pkl
