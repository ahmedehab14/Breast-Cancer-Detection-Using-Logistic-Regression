# Breast Cancer Detection Using Logistic Regression

This project aims to build a machine learning model for detecting breast cancer using **Logistic Regression**, a commonly used classification algorithm. The goal is to classify tumors as either **benign** or **malignant** based on medical features extracted from breast cancer cell samples. The dataset used for this project is the well-known **Wisconsin Breast Cancer Dataset** (WBCD) from Kaggle, which contains numerous features describing the characteristics of cell nuclei present in breast cancer biopsies.

By training a Logistic Regression model, the system predicts whether a tumor is benign (non-cancerous) or malignant (cancerous), helping medical professionals make more informed decisions in the diagnosis of breast cancer.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Model Evaluation](#model-evaluation)
4. [Conclusion](#conclusion)

## Project Overview

Breast cancer detection is one of the most important and sensitive tasks in medical diagnostics. Early detection of cancerous tumors can significantly improve the prognosis and survival rates of patients. In this project, we use **Logistic Regression**, a simple yet effective machine learning classification algorithm, to predict the malignancy of a tumor based on a set of features.

### Objectives:
- Preprocess the **Wisconsin Breast Cancer Dataset** to extract relevant features.
- Train a Logistic Regression model on the dataset.
- Evaluate the model's performance using various metrics like accuracy, precision, recall, and F1-score.
- Generate a confusion matrix to visually assess the model's performance.
- Provide a detailed analysis and conclusions based on the results.

## Dataset

The dataset used in this project is the **Wisconsin Breast Cancer Dataset (WBCD)**, which includes the following attributes:
- **Number of Instances**: 569
- **Number of Features**: 30
- **Target Variable**: 
  - **0** for benign tumors
  - **1** for malignant tumors
- **Features**: Each instance contains 30 features describing the cell nuclei characteristics, such as:
  - Radius (mean, standard error, and worst)
  - Texture (mean, standard error, and worst)
  - Smoothness, Compactness, Concavity, and others
  
This dataset is widely used for evaluating machine learning models for cancer classification and is publicly available on [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

## Technologies Used

- **Python**: The programming language used to implement the solution.
- **Scikit-learn**: A Python library used for implementing machine learning algorithms, including Logistic Regression and performance metrics.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations and handling arrays.
- **Matplotlib and Seaborn**: For data visualization and model evaluation.

## Model Evaluation

The performance of the Logistic Regression model was evaluated using the following metrics:

### Accuracy:
- **Accuracy**: 98%  
  This metric indicates that the model correctly classified 98% of the instances.

### Precision, Recall, and F1-Score:
- **Precision for class 0 (benign)**: 0.99  
  This means that 99% of the tumors classified as benign were indeed benign.
- **Recall for class 0 (benign)**: 0.98  
  This indicates that 98% of benign tumors were correctly identified.
- **F1-Score for class 0 (benign)**: 0.99  
  The F1-Score combines precision and recall, showing a balanced performance for benign tumors.

- **Precision for class 1 (malignant)**: 0.97  
  This means that 97% of tumors classified as malignant were indeed malignant.
- **Recall for class 1 (malignant)**: 0.98  
  This indicates that 98% of malignant tumors were correctly identified.
- **F1-Score for class 1 (malignant)**: 0.98  
  The F1-Score shows a good balance for malignant tumors.

### Confusion Matrix:

```lua
[[106   2]   # 106 true negatives, 2 false positives
 [  1  62]]   # 1 false negative, 62 true positives

## Conclusion:
Overall, the model exhibits excellent performance in distinguishing between benign and malignant cases. It achieves high accuracy, precision, recall, and F1-score, making it suitable for practical use in classifying medical data such as breast cancer diagnosis.


