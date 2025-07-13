
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Load datasets
ratings = pd.read_csv(r"C:\Users\HP\Downloads\ml-latest-small\ml-latest-small\ratings.csv")
movies = pd.read_csv(r"C:\Users\HP\Downloads\ml-latest-small\ml-latest-small\movies.csv")

movies = movies.dropna(subset=['title', 'genres'])

ratings['liked'] = (ratings['rating'] >= 4).astype(int)

data = pd.merge(ratings, movies, on='movieId', how='inner')

data['primary_genre'] = data['genres'].apply(lambda x: x.split('|')[0])

data['text'] = data['title'] + ' ' + data['primary_genre']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['liked']  # Target is whether user liked the movie

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
recall = recall_score(y_test, y_pred, average='binary', zero_division=0)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
