# Movie Genre Classification

This project aims to classify movie genres using machine learning models based on movie descriptions. The project uses a dataset from Kaggle and explores Logistic Regression, Multinomial Naive Bayes, and Linear SVC models.

## Data

The dataset is sourced from Kaggle: `hijest/genre-classification-dataset-imdb`.
It consists of three files:
- `train_data.txt`: Training data with Title, Genre, and Description.
- `test_data.txt`: Test data with Id, Title, and Description.
- `test_data_solution.txt`: Solution data with Title, Genre, and Description for the test set.

The data is loaded directly in the Colab environment using `kagglehub`.

## Preprocessing

The movie descriptions undergo the following preprocessing steps:
1.  **Character Cleaning:** Removing non-alphabetic characters and converting to lowercase.
2.  **Stop Word Removal:** Removing common English stop words using `nltk.corpus.stopwords`.
3.  **Lemmatization:** Reducing words to their base form using `nltk.stem.WordNetLemmatizer`.

## Feature Extraction

The preprocessed descriptions are converted into numerical features using `TfidfVectorizer`. This creates a matrix where each row represents a document (movie description) and each column represents a term, with values indicating the TF-IDF score.

## Models and Evaluation

Three classification models were trained and evaluated:

1.  **Logistic Regression:**
    - Accuracy: ~49%
    - Evaluation based on `accuracy_score` and `classification_report`.

2.  **Multinomial Naive Bayes:**
    - Accuracy: ~44%
    - Evaluation based on `accuracy_score` and `classification_report` with `zero_division=0`.

3.  **Linear Support Vector Machine (SVM):**
    - Accuracy: ~58%
    - Evaluation based on `accuracy_score` and `classification_report`.

The SVM model achieved the highest accuracy among the three, although the overall performance is limited.

## Analysis of Results

The relatively low accuracy across all models is likely due to the **imbalance in the dataset's genre distribution**. The plot of genre counts shows that some genres, such as Drama and Documentary, appear much more frequently than others in the training data. This imbalance can cause models to perform poorly on underrepresented genres.

## Visualizations

The notebook includes visualizations to show the genre distribution:

- **Distribution of Genres in Training Data:** A count plot showing the frequency of each genre in the training set.
- **Comparison of Actual and Predicted Genre Distributions (SVM Model):** A bar plot comparing the actual distribution of genres in the test solution data to the predicted distribution from the SVM model.

These plots highlight the imbalance in the dataset and how the predictions reflect this imbalance.

## Conclusion

The project demonstrates the challenges of classifying movie genres with an imbalanced dataset. While SVM performed best among the tested models, the accuracy indicates that further techniques to handle data imbalance or more advanced models may be necessary to improve performance.
