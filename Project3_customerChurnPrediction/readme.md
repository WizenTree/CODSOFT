# Bank Customer Churn Prediction

This project aims to predict customer churn for a bank using various machine learning models. By identifying customers likely to churn, the bank can implement targeted retention strategies.

## Project Overview

Customer churn is a significant concern for banks, leading to lost revenue and increased acquisition costs. This project develops predictive models to identify customers who are at risk of leaving the bank. The goal is to build accurate models that can be used to inform customer retention efforts.

## Dataset

The dataset used in this project is the [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) dataset from Kaggle. It contains information about bank customers, including their demographics, account details, and whether they have churned or not.

The dataset includes the following features:

- `RowNumber`: Row number
- `CustomerId`: Unique customer ID
- `Surname`: Customer's surname
- `CreditScore`: Credit score of the customer
- `Geography`: Country of the customer (France, Spain, Germany)
- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Tenure`: Number of years the customer has been with the bank
- `Balance`: Account balance of the customer
- `NumOfProducts`: Number of bank products the customer uses
- `HasCrCard`: Whether the customer has a credit card (1=Yes, 0=No)
- `IsActiveMember`: Whether the customer is an active member (1=Yes, 0=No)
- `EstimatedSalary`: Estimated salary of the customer
- `Exited`: Whether the customer has churned (1=Yes, 0=No) - **Target Variable**

## Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading and Exploration:** The dataset is loaded and initial data exploration is performed to understand the data structure, types, and distributions.
2.  **Data Preprocessing:**
    *   Irrelevant columns (`RowNumber`, `CustomerId`, `Surname`) are dropped.
    *   Categorical features (`Gender`, `Geography`) are one-hot encoded to convert them into a numerical format suitable for modeling.
3.  **Data Splitting:** The dataset is split into training and testing sets to evaluate the performance of the models on unseen data.
4.  **Model Training:** Three different classification models are trained on the preprocessed training data.
5.  **Model Evaluation:** The trained models are evaluated using various metrics relevant to classification problems, including Accuracy, Precision, Recall, F1-Score, and ROC AUC Score. A classification report and confusion matrix are also generated for each model.

## Models Used

The following machine learning models were used for churn prediction:

1.  **Logistic Regression:** A linear model for binary classification.
2.  **Random Forest:** An ensemble learning method that builds multiple decision trees and combines their predictions.
3.  **Gradient Boosting:** Another ensemble technique that builds trees sequentially, where each new tree corrects the errors of the previous ones.

## Results

The models were evaluated based on their performance metrics. The results are summarized below:

| Model                | Accuracy | Precision | Recall | F1 Score | ROC AUC Score |
| :------------------- | :------- | :-------- | :----- | :------- | :------------ |
| Logistic Regression  | 0.8015   | 0.5582    | 0.2080 | 0.3039   | 0.7790        |
| Random Forest        | 0.8700   | 0.7747    | 0.4842 | 0.5964   | 0.9173        |
| Gradient Boosting    | 0.8675   | 0.7441    | 0.5059 | 0.6022   | 0.9179        |

Based on the evaluation metrics, Random Forest and Gradient Boosting models demonstrate superior performance compared to Logistic Regression for this churn prediction task.

## Getting Started

To run this project locally, you will need:

*   Python 3.6 or higher
*   Required libraries: `pandas`, `scikit-learn`, `kagglehub`, `IPython`

You can install the required libraries using pip:
```
pip install pandas scikit-learn kagglehub ipython
```
The code is available in a Colab notebook format. You can open the notebook in Google Colab or download it and run it in a Jupyter Notebook environment.
