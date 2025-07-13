import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

# --- Load Data ---
ratings = pd.read_csv(r"C:\Users\HP\Downloads\ml-100k\ml-100k\u.data", sep="\t",
                      names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv(r"C:\Users\HP\Downloads\ml-100k\ml-100k\u.item", sep="|", encoding="ISO-8859-1", header=None,
                     names=["movieId", "title", "release_date", "video_release_date", "IMDb_URL",
                            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                            "Romance", "Sci-Fi", "Thriller", "War", "Western"])

# --- Process Genre Info ---
genre_cols = movies.columns[5:]
movies['genres'] = movies[genre_cols].apply(lambda x: [g for g, v in zip(genre_cols, x) if v == 1], axis=1)
genre_matrix = movies[['movieId', 'genres']]
ratings_genre = ratings.merge(genre_matrix, on='movieId')

# --- Encode Genre Info ---
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(ratings_genre['genres']), columns=mlb.classes_)
ratings_genre = pd.concat([ratings_genre, genre_encoded], axis=1)

for genre in mlb.classes_:
    ratings_genre[genre] = ratings_genre[genre] * ratings_genre['rating']

# --- User Genre Profile Matrix ---
user_genre = ratings_genre.groupby('userId')[mlb.classes_].mean().fillna(0)

# --- K-Means Clustering based on Genre Info ---
k = 20  # 20 genres → 20 clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
user_genre['cluster'] = kmeans.fit_predict(user_genre)

# --- Pivot Ratings to User-Item Matrix ---
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# --- Calculate item average ratings for Popularity term in PIP ---
item_avg_ratings = ratings.groupby('movieId')['rating'].mean().to_dict()
R_max = 5
R_min = 1
R_m = (R_max + R_min) / 2

# --- PIP Similarity Function ---
def pip_similarity(u1, u2):
    common_items = user_item_matrix.columns[(user_item_matrix.loc[u1] > 0) & (user_item_matrix.loc[u2] > 0)]
    pip = 0
    for item in common_items:
        r1, r2 = user_item_matrix.loc[u1, item], user_item_matrix.loc[u2, item]
        alpha = item_avg_ratings.get(item, 3.0)
        
        # Agreement
        agree = not ((r1 > R_m and r2 < R_m) or (r1 < R_m and r2 > R_m))
        
        # Proximity
        d = abs(r1 - r2) if agree else 2 * abs(r1 - r2)
        proximity = (2 * (R_max - R_min) + 1 - d) / 2

        # Impact
        impact = ((abs(r1 - R_m) + 1) * (abs(r2 - R_m) + 1)) if agree else \
                 1 / ((abs(r1 - R_m) + 1) * (abs(r2 - R_m) + 1))

        # Popularity
        if (r1 > alpha and r2 > alpha) or (r1 < alpha and r2 < alpha):
            popularity = 1 + ((r1 + r2) / 2 - alpha) ** 2
        else:
            popularity = 1

        pip += proximity * impact * popularity

    return pip

# --- Recommendation Function ---
def recommend_for_user(target_user, top_n=10):
    target_cluster = user_genre.loc[target_user, 'cluster']
    print(f"Target user {target_user} is in cluster {target_cluster}\n")
    
    # Users from other clusters only
    other_users = user_genre[user_genre['cluster'] != target_cluster].index
    pip_scores = {u: pip_similarity(target_user, u) for u in other_users}
    pip_scores = {u: score for u, score in pip_scores.items() if score > 0}
    
    if not pip_scores:
        print("No similar users found from other clusters.")
        return

    # Sorted similar users
    neighbors = sorted(pip_scores.items(), key=lambda x: -x[1])
    neighbor_ids = [u for u, _ in neighbors]

    # Rated and unrated items by target user
    rated_items = user_item_matrix.loc[target_user][user_item_matrix.loc[target_user] > 0].index
    unrated_items = user_item_matrix.columns.difference(rated_items)

    predictions = {}
    for item in unrated_items:
        num, denom = 0, 0
        for neighbor in neighbor_ids:
            r = user_item_matrix.loc[neighbor, item]
            if r > 0:
                sim = pip_scores[neighbor]
                mean_neighbor = user_item_matrix.loc[neighbor][user_item_matrix.loc[neighbor] > 0].mean()
                num += sim * (r - mean_neighbor)
                denom += abs(sim)
        if denom > 0:
            user_mean = user_item_matrix.loc[target_user][user_item_matrix.loc[target_user] > 0].mean()
            predictions[item] = user_mean + num / denom

    # Top N Recommendations
    top_items = sorted(predictions.items(), key=lambda x: -x[1])[:top_n]
    print(f"Top {top_n} Recommended Movies for User {target_user}:\n")
    print(f"{'Title':<45} {'Genres':<35} {'Predicted Rating'}")
    print("=" * 95)
    for movie_id, rating in top_items:
        row = movies[movies['movieId'] == movie_id]
        if not row.empty:
            title = row['title'].values[0]
            genres = ", ".join(row['genres'].values[0])
            print(f"{title:<45} {genres:<35} {rating:.2f}")
        else:
            print(f"Movie ID {movie_id} not found")

# --- Run for Users 12 and 31 ---
recommend_for_user(10, top_n=5)
print("\n" + "-" * 100 + "\n")
recommend_for_user(22, top_n=5)
