# Credit Card Fraud Detection

This project implements and evaluates several classification models to detect fraudulent credit card transactions. The goal is to identify potentially fraudulent transactions within a dataset to minimize financial losses.

## Project Overview

The project involves the following steps:

1.  **Data Acquisition:** Downloading the transaction data from Kaggle.
2.  **Data Preprocessing and Feature Engineering:** Cleaning the data, transforming date and time information into useful features (hour, day of week, weekend status, month, year), calculating age from date of birth, and performing one-hot encoding on categorical variables.
3.  **Model Selection:** Utilizing three different classification algorithms: Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.
4.  **Model Training:** Training each selected model on the prepared training data.
5.  **Model Evaluation:** Assessing the performance of each model using standard classification metrics such as Accuracy, Precision, Recall, F1-score, ROC AUC, and Confusion Matrix.
6.  **Analysis of Results:** Discussing the performance of the models, particularly in the context of imbalanced datasets, and highlighting the challenges encountered (e.g., Accuracy Paradox).

## Dataset

The dataset used in this project is the "Fraud Detection" dataset from Kaggle, available at: `kagglehub.dataset_download('kartik2112/fraud-detection')`. It contains information about credit card transactions, including transaction details, customer information, and a label indicating whether the transaction is fraudulent or not.

## Requirements

To run this project, you will need to install the following libraries:

-   `kagglehub`
-   `pandas`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`

You can install these using pip:
```bash
pip install kagglehub pandas scikit-learn matplotlib seaborn
```
## Usage

1.  Clone this repository.
2.  Ensure you have the required libraries installed.
3.  Run the Python script (or Jupyter Notebook) containing the project code.
4.  The script will download the data, preprocess it, train the models, and print the evaluation metrics for each model.

## Code Structure

The core logic of the project is contained within a single script or notebook, following these logical sections:

-   Importing necessary libraries and the dataset.
-   Data cleaning and feature engineering steps.
-   Defining and training the classification models.
-   Evaluating the models and printing performance metrics.
-   Visualizing data distribution (e.g., fraud vs. non-fraud).

## Results and Discussion

The evaluation metrics show that while the overall accuracy scores are high (above 90%), this is misleading due to the highly imbalanced nature of the dataset (a significantly larger number of non-fraudulent transactions compared to fraudulent ones). This phenomenon is known as the Accuracy Paradox. The project includes a visualization demonstrating this class imbalance. Further steps would involve addressing this imbalance using techniques like oversampling, undersampling, or using different evaluation metrics more suitable for imbalanced datasets.

## Limitations

-   The current implementation does not address the data imbalance issue, leading to potentially misleading accuracy scores.
-   Further model tuning and exploration of other algorithms could improve performance.
