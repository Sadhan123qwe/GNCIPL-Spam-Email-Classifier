# GNCIPL-Spam-Email-Classifier

Spam Email Classifier 📧

A machine learning-based application to effectively classify emails as "Spam" or "Ham" (Not Spam) using Natural Language Processing (NLP) techniques.

Table of Contents

    Overview

    Features

    Tech Stack

    Project Structure

    Installation & Setup

    Usage

    Model Performance

    Contributing

    License

    Contact

Overview

The proliferation of unsolicited emails (spam) is a persistent issue, cluttering inboxes and posing security risks. This project implements a robust spam classifier using a supervised machine learning approach. The model is trained on a labeled dataset of emails to learn the patterns and features that distinguish spam from legitimate emails (ham). It utilizes NLP techniques for text preprocessing and feature extraction to build an accurate and efficient classification model.

The primary goal is to achieve high accuracy and a low false-positive rate, ensuring that important emails are not mistakenly flagged as spam.

Features

    Text Preprocessing: Includes lowercasing, tokenization, stop-word removal, and stemming/lemmatization to clean and prepare the email text.

    Feature Extraction: Converts text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).

    Multiple Models: Implemented and compared several classification algorithms, including:

        Multinomial Naive Bayes

        Support Vector Machine (SVM)

        Logistic Regression

    Performance Evaluation: Assessed models using key metrics like Accuracy, Precision, Recall, and F1-Score.

    Prediction Script: A simple command-line interface to classify a new email on the fly.

Tech Stack

    Language: Python 3.8+

    Libraries:

        Scikit-learn: For machine learning models and metrics.

        Pandas: For data manipulation and loading.

        NLTK (Natural Language Toolkit): For text preprocessing tasks.

        Matplotlib & Seaborn: For data visualization and results plotting.

        Joblib/Pickle: For saving and loading the trained model.

Project Structure

Spam-Email-Classifier/
│
├── data/
│   └── spam.csv                # The dataset used for training
│
├── models/
│   ├── spam_classifier.pkl     # Saved trained model
│   └── vectorizer.pkl          # Saved TF-IDF vectorizer
│
├── notebooks/
│   └── spam_classification_eda.ipynb  # Jupyter Notebook for EDA and model building
│
├── src/
│   ├── preprocess.py           # Script for text preprocessing functions
│   ├── train.py                # Script to train and save the model
│   └── predict.py              # Script to make predictions on new emails
│
├── requirements.txt            # Project dependencies
├── .gitignore                  # Files to be ignored by Git
└── README.md                   # This file

Installation & Setup

Follow these steps to get the project up and running on your local machine.

    Clone the repository:
    Bash

git clone https://github.com/your-username/Spam-Email-Classifier.git
cd Spam-Email-Classifier

Create and activate a virtual environment (recommended):
Bash

# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate

Install the required dependencies:
Bash

pip install -r requirements.txt

Download NLTK data:
Run the following command in your Python interpreter to download the necessary NLTK packages.
Python

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

Usage

1. Training the Model

To train the classifier from scratch, run the train.py script. This will process the dataset in the data/ folder and save the trained model and vectorizer to the models/ directory.
Bash

python src/train.py

2. Classifying a New Email

Use the predict.py script to classify a single email. You can pass the email text directly as a command-line argument.
Bash

python src/predict.py --text "Congratulations! You've won a $1,000,000 prize. Click here to claim."

Expected Output:

Prediction: Spam

Bash

python src/predict.py --text "Hey, just checking in. Are we still on for the meeting tomorrow at 10 AM?"

Expected Output:

Prediction: Ham

Model Performance

The models were evaluated on a held-out test set. The performance metrics are as follows:
Model	Accuracy	Precision (Spam)	Recall (Spam)	F1-Score (Spam)
Multinomial Naive Bayes	98.3%	0.98	0.89	0.93
Logistic Regression	97.9%	0.97	0.86	0.91
Support Vector Machine	98.6%	0.99	0.90	0.94

Based on the results, the Support Vector Machine (SVM) was selected as the final model due to its superior F1-score and precision.

Contributing

Contributions are welcome! If you have any suggestions for improvements or want to add new features, please follow these steps:

    Fork the repository.

    Create a new branch (git checkout -b feature/your-feature-name).

    Commit your changes (git commit -m 'Add some amazing feature').

    Push to the branch (git push origin feature/your-feature-name).

    Open a Pull Request.
