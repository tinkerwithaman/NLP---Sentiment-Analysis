# Sentiment Analysis of Text

## Description
This project implements a classic sentiment analysis model to classify text as either positive or negative. It uses a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer to convert text into numerical features and a Logistic Regression classifier for the prediction task. The model is trained on a small, sample dataset included directly in the script.

## Features
- TF-IDF for text vectorization.
- Logistic Regression for classification.
- Simple, self-contained, and easy to run.
- Includes a function to predict the sentiment of new, unseen sentences.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Sentiment_Analysis/
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the script:**
    ```bash
    python src/main.py
    ```

## Example Output
```
Model trained successfully!
Accuracy on test set: 100.00%
---
Testing with new sentences:
Sentence: 'This movie was absolutely fantastic!'
Predicted Sentiment: Positive

Sentence: 'I did not like the plot, it was boring.'
Predicted Sentiment: Negative
```
