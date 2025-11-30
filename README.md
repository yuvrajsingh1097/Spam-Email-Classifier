Simple Spam Email Classifier

Project Overview

This project implements a basic machine learning model to classify text messages as either "Ham" (legitimate) or "Spam" (unsolicited/malicious). It serves as a fundamental example of natural language processing (NLP) and supervised learning using Python and the scikit-learn library.

The core objective is to demonstrate the process of:

Loading and preparing text data.

Vectorizing text features using TF-IDF.

Training a powerful, yet simple, Naive Bayes classifier.

Evaluating the model's performance.

Key Technologies

Python: The primary programming language.

Pandas: Used for data handling and manipulation (creating the dataset DataFrame).

Scikit-learn (sklearn): The essential machine learning library, providing:

TfidfVectorizer for feature extraction.

MultinomialNB for model training.

Various metrics (accuracy_score, confusion_matrix) for evaluation.

TF-IDF (Term Frequency-Inverse Document Frequency): The technique used to convert raw text into a numerical matrix that the classifier can process.

How It Works

The classifier follows a standard text classification pipeline:

Data Preparation: A small, labeled dataset of example messages is created and assigned numerical labels (0 for Ham, 1 for Spam).

Train/Test Split: The dataset is split to train the model on one subset and test its performance on an unseen subset.

Vectorization (TF-IDF): The TfidfVectorizer calculates a numerical weight for each word, reflecting its importance in the document relative to the entire dataset. This creates the feature matrix used for training.

Model Training: A Multinomial Naive Bayes model is trained on the TF-IDF vectors and their corresponding labels. Naive Bayes is chosen for its efficiency and effectiveness in text classification tasks.

Prediction and Evaluation: The model's predictions are compared against the true labels of the test set, and metrics like Accuracy and the Classification Report are printed to assess performance.

