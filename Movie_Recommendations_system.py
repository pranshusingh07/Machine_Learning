#Movie Recommendation System
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Dataset
data = {
    'movie': [
        'Avengers', 'Iron Man', 'Captain America',
        'Titanic', 'Notebook', 'Inception'
    ],
    'genre': [
        'action superhero', 'action tech', 'action patriot',
        'romance drama', 'romance love', 'sci-fi thriller'
    ]
}

df = pd.DataFrame(data)

# Convert text to vectors
cv = CountVectorizer()
matrix = cv.fit_transform(df['genre'])

# Similarity Matrix
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie_name):
    if movie_name not in df['movie'].values:
        print("Movie not found")
        return
    
    index = df[df['movie'] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")
    for i in scores[1:4]:
        print(df.iloc[i[0]]['movie'])

# User Input
movie = input("Enter movie name: ")
recommend(movie)
