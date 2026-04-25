#Email Spam Detection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    'text': [
        'Win money now!!!',
        'Hello, how are you?',
        'Claim your prize',
        'Let us meet tomorrow',
        'Congratulations! You won a lottery',
        'Are you coming to class?',
        'Free entry in contest',
        'Project submission deadline'
    ],
    'label': [1,0,1,0,1,0,1,0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
msg = input("Enter message: ")
msg_vec = vectorizer.transform([msg])
prediction = model.predict(msg_vec)

if prediction == 1:
    print("📛 Spam Message")
else:
    print("✅ Not Spam")

# Accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
