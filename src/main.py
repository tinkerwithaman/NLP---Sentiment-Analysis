import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_sentiment_model(data, labels):
    """
    Trains a sentiment analysis model.

    Args:
        data (list): A list of text samples.
        labels (list): A list of corresponding labels (0 for negative, 1 for positive).

    Returns:
        tuple: A tuple containing the trained vectorizer and model.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Create and train the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully!")
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    return vectorizer, model

def predict_sentiment(text, vectorizer, model):
    """
    Predicts the sentiment of a given text.

    Args:
        text (str): The input text.
        vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        model (LogisticRegression): The trained classifier.

    Returns:
        str: The predicted sentiment ('Positive' or 'Negative').
    """
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

if __name__ == "__main__":
    # Sample dataset (in a real project, this would be in the data/ folder)
    # 1 for positive, 0 for negative
    sample_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience I have ever had.",
        "The customer service was excellent and very helpful.",
        "A complete waste of money and time.",
        "I am very satisfied with my purchase.",
        "The movie was boring and too long.",
        "What a fantastic performance by the actors!",
        "I will never buy from this company again.",
        "The book was a masterpiece of storytelling.",
        "The food was cold and tasteless.",
        "A truly inspiring and heartwarming story.",
        "The software is buggy and crashes constantly.",
        "I highly recommend this to everyone.",
        "It failed to meet my expectations.",
        "An incredible journey from start to finish.",
        "The directions were confusing and unclear."
    ]
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    # Train the model
    vectorizer, model = train_sentiment_model(sample_texts, sample_labels)

    # Test with new sentences
    print("\n---\nTesting with new sentences:")
    test_sentence_1 = "This movie was absolutely fantastic!"
    sentiment_1 = predict_sentiment(test_sentence_1, vectorizer, model)
    print(f"Sentence: '{test_sentence_1}'")
    print(f"Predicted Sentiment: {sentiment_1}\n")

    test_sentence_2 = "I did not like the plot, it was boring."
    sentiment_2 = predict_sentiment(test_sentence_2, vectorizer, model)
    print(f"Sentence: '{test_sentence_2}'")
    print(f"Predicted Sentiment: {sentiment_2}")
