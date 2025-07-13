
#same output
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# === 1. Load Data ===
ratings_df = pd.read_csv("C:/Users/HP/Downloads/ml-latest-small/ml-latest-small/ratings.csv")
movies_df = pd.read_csv("C:/Users/HP/Downloads/ml-latest-small/ml-latest-small/movies.csv")

user_ids = ratings_df["userId"].unique()
movie_ids = ratings_df["movieId"].unique()

user_map = {uid: idx for idx, uid in enumerate(user_ids)}
movie_map = {mid: idx for idx, mid in enumerate(movie_ids)}
index_to_movie = {v: k for k, v in movie_map.items()}
movie_id_to_title = dict(zip(movies_df.movieId, movies_df.title))
movie_id_to_genre = dict(zip(movies_df.movieId, movies_df.genres))

n_users = len(user_ids)
n_movies = len(movie_ids)

# === 2. Create Rating Matrix ===
rating_matrix = np.zeros((n_users, n_movies), dtype=np.float32)
for row in ratings_df.itertuples():
    rating_matrix[user_map[row.userId], movie_map[row.movieId]] = row.rating

# === 3. Hold-Out 1 Test Sample per User ===
train_matrix = rating_matrix.copy()
test_samples = []

np.random.seed(42)
torch.manual_seed(42)

for u in range(n_users):
    rated = np.where(rating_matrix[u] > 0)[0]
    if len(rated) > 0:
        m = np.random.choice(rated)
        test_samples.append((u, m, rating_matrix[u, m]))
        train_matrix[u, m] = 0.0

# === 4. Dataset for Training on Withheld Ratings Only ===
class WithheldRatingDataset(Dataset):
    def __init__(self, train_matrix, test_samples):
        self.train_matrix = torch.tensor(train_matrix, dtype=torch.float32)
        self.test_samples = test_samples
    def __len__(self):
        return len(self.test_samples)
    def __getitem__(self, idx):
        u, m, r = self.test_samples[idx]
        return self.train_matrix[u], m, r

train_loader = DataLoader(WithheldRatingDataset(train_matrix, test_samples), batch_size=64, shuffle=True)

# === 5. 7-layer Autoencoder with Bottleneck ===
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096), nn.ReLU(),
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 4096), nn.ReLU(),
            nn.Linear(4096, input_dim)
        )
    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction, bottleneck

# === 6. Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(n_movies).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
sparsity_lambda = 1e-5  # L1 regularization

train_rmse_per_epoch = []

for epoch in range(20):
    model.train()
    total_rmse = 0
    for user_profiles, movie_indices, true_ratings in train_loader:
        user_profiles = user_profiles.to(device)
        movie_indices = movie_indices.to(device)
        true_ratings = true_ratings.to(device)

        pred_matrix, encoded = model(user_profiles)
        batch_preds = pred_matrix[torch.arange(pred_matrix.size(0)), movie_indices]

        rmse_loss = torch.sqrt(torch.mean((batch_preds - true_ratings) ** 2))
        sparsity_loss = torch.norm(encoded, p=1)
        loss = rmse_loss + sparsity_lambda * sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_rmse += rmse_loss.item()

    avg_rmse = total_rmse / len(train_loader)
    train_rmse_per_epoch.append(avg_rmse)
    print(f"Epoch {epoch+1}: Train RMSE (on withheld) = {avg_rmse:.4f}")

# === 7. Final Test RMSE ===
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for u, m, true_rating in test_samples:
        input_vec = torch.tensor(train_matrix[u:u+1], dtype=torch.float32, device=device)
        pred_vec, _ = model(input_vec)
        test_preds.append(pred_vec[0, m].item())
        test_targets.append(true_rating)

test_rmse = np.sqrt(np.mean((np.array(test_preds) - np.array(test_targets)) ** 2))
print(f"\n Final Test RMSE on withheld ratings: {test_rmse:.4f}")

# === 8. Plot Loss Curve ===
plt.figure(figsize=(8, 4))
plt.plot(train_rmse_per_epoch, marker='o', color='tomato')
plt.title("Train RMSE vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("RMSE (on withheld ratings)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 9. Recommend Movies with Genre ===
def get_movie_title_by_index(index):
    movie_id = index_to_movie[index]
    return movie_id_to_title.get(movie_id, f"MovieID {movie_id}")

def get_movie_genre_by_index(index):
    movie_id = index_to_movie[index]
    return movie_id_to_genre.get(movie_id, "Unknown")

def recommend_movies_for_user(user_index, top_n=5):
    model.eval()
    with torch.no_grad():
        input_vec = torch.tensor(train_matrix[user_index:user_index+1], dtype=torch.float32, device=device)
        predicted_ratings, _ = model(input_vec)
        predicted_ratings = predicted_ratings.cpu().numpy().flatten()
    already_rated = rating_matrix[user_index] > 0
    predicted_ratings[already_rated] = -np.inf
    top_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    print(f"\n Top {top_n} movie recommendations for User {user_index+1}:")
    for idx in top_indices:
        title = get_movie_title_by_index(idx)
        genre = get_movie_genre_by_index(idx)
        print(f"{title} — Genre: {genre} — Predicted Rating: {predicted_ratings[idx]:.2f}")

# === 10. Interactive Test ===
while True:
    try:
        user_id = int(input(f"\nEnter a user ID (1 to {n_users}), or 0 to exit: "))
        if user_id == 0:
            print("Exiting recommendations.")
            break
        if 1 <= user_id <= n_users:
            recommend_movies_for_user(user_id - 1)
        else:
            print(" Invalid user ID.")
    except Exception as e:
        print(" Invalid input. Please enter a valid user ID or 0 to exit.")
