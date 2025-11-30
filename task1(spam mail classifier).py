import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

data = {
    'label': ['ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
    'message': [
        'Hey, how are you doing?',
        'you got intership letter from kodbud. Congratulations!',
        'WINNER! Claim your FREE $1000 prize now.',
        'The meeting for kodbud task briefing is scheduled for tomorrow at 10 AM.',
        'Urgent! You have won a lottery. Click this link to redeem.',
        'Please confirm receipt of this email.',
        'FREE membership! Limited time offer. Subscribe today.',
        'Can we reschedule the dinner?, As i have to compplete my kodbud task.',
        'Cash reward waiting for you! Text CLAIM to 4444.',
        'Let us know about the kodbud project status update.',
    ]
}
df = pd.DataFrame(data)

df['label_code'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label_code']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))

def predict_spam(message, model, vectorizer):
    message_tfidf = vectorizer.transform([message])

    prediction_code = model.predict(message_tfidf)[0]

    return 'Spam' if prediction_code == 1 else 'Ham'

new_message_1 = input("Enter a message to classify (spam/ham): ")
new_message_2 = "Hey John, did you get my email about the plan?"

print("\n--- New Message Prediction ---")
print(f"Message 1: '{new_message_1}'")
print(f"Prediction: {predict_spam(new_message_1, model, tfidf_vectorizer)}")

print(f"\nMessage 2: '{new_message_2}'")
print(f"Prediction: {predict_spam(new_message_2, model, tfidf_vectorizer)}")
