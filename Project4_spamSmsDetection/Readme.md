# SMS Spam Classification Project

## Project Goal

This project addresses the task of building an AI model to classify SMS messages as either spam or legitimate (ham). The objective is to explore and implement various techniques for text representation and classification to effectively identify spam messages.

## Approach

The project leverages established natural language processing (NLP) and machine learning techniques to achieve the classification goal. The methodology includes:

1.  **Data Preparation:** The SMS Spam Collection Dataset is used. Initial steps involve loading the data, handling missing values (if any), and cleaning the text messages. Text cleaning includes removing irrelevant characters, converting text to lowercase, and removing common English stopwords.
2.  **Feature Extraction:** Two primary techniques for converting text data into numerical features are considered and implemented:
    *   **TF-IDF (Term Frequency-Inverse Document Frequency):** This method quantifies the importance of words in a document relative to a collection of documents.
    *   *(While word embeddings were mentioned, this project primarily focuses on TF-IDF as demonstrated in the provided code. If word embeddings were implemented, they would be listed here as well.)*
3.  **Model Selection and Training:** Several classification algorithms suitable for text data are employed and trained on the extracted features:
    *   **Multinomial Naive Bayes:** A probabilistic classifier often effective for text classification tasks due to its simplicity and efficiency.
    *   **Logistic Regression:** A linear model that estimates the probability of a message being spam.
    *   **Support Vector Machine (SVM):** A powerful model that finds an optimal hyperplane to separate the classes.
4.  **Model Evaluation:** The trained models are evaluated using standard metrics such as accuracy, precision, recall, and F1-score to assess their performance in identifying spam messages.

## Dataset

The project utilizes the publicly available SMS Spam Collection Dataset, accessed via Kaggle Hub.

## Implementation Details

The project is implemented in a Google Colab environment using Python. Key libraries include `pandas` for data manipulation, `scikit-learn` for model building and evaluation, and `nltk` for text preprocessing tasks like stopword removal and lemmatization.

## Results

The performance of each model (Multinomial Naive Bayes, Logistic Regression, and SVM) is evaluated and compared. The results indicate the effectiveness of these methods in classifying SMS spam, with the SVM model demonstrating strong performance in this specific implementation.

## Technical Requirements

*   Python 3.6+
*   Google Colab or Jupyter Notebook
*   Required Python libraries: `kagglehub`, `pandas`, `scikit-learn`, `nltk`, `re`
*   NLTK data: `stopwords`, `wordnet`

## Usage

To run this project:

1.  Open the provided Google Colab notebook.
2.  Execute all code cells sequentially.
3.  The output will display the evaluation metrics for each trained model.
