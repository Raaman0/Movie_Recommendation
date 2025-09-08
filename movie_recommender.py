# movie_recommender.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Load and preprocess data
# ---------------------------

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, left_on='id', right_on='movie_id')

# Combine features for content-based filtering
def combine_features(row):
    return str(row['genres']) + " " + str(row['keywords']) + " " + str(row['overview']) + " " + str(row['cast']) + " " + str(row['crew'])

movies['combined_features'] = movies.apply(combine_features, axis=1)
movies['title'] = movies['title_x']  # convenient column

# Indices for quick lookup
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ---------------------------
# Step 2: Content-based filtering
# ---------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommend(movie_title, top_n=5):
    if movie_title not in indices:
        return []
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].values

# ---------------------------
# Step 3: Collaborative filtering (simulated ratings)
# ---------------------------
num_users = 50
num_movies = 200
np.random.seed(42)
user_ids = np.arange(1, num_users+1)
movie_ids_sample = movies['id'].sample(num_movies, random_state=42).values

ratings_list = []
for user in user_ids:
    for movie in movie_ids_sample:
        ratings_list.append([user, movie, np.random.randint(1, 6)])  # ratings 1-5

ratings = pd.DataFrame(ratings_list, columns=['user_id', 'movie_id', 'rating'])

# User-item matrix
user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
collab_similarity = cosine_similarity(user_movie_matrix.T)
collab_similarity_df = pd.DataFrame(collab_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

def collaborative_recommend(movie_id, top_n=5):
    if movie_id not in collab_similarity_df.columns:
        return []
    sim_scores = collab_similarity_df[movie_id]
    sim_scores = sim_scores.sort_values(ascending=False)
    sim_scores = sim_scores.drop(movie_id)
    top_movie_ids = sim_scores.head(top_n).index
    return movies[movies['id'].isin(top_movie_ids)]['title'].values

# ---------------------------
# Step 4: Hybrid recommendation
# ---------------------------
def hybrid_recommend(movie_title, top_n=5):
    # User-friendly partial match
    title_lower = movie_title.lower()
    matches = movies[movies['title'].str.lower().str.contains(title_lower)]
    if matches.empty:
        print(f"Movie '{movie_title}' not found.")
        return []
    movie_title = matches['title'].iloc[0]  # first match

    # Content-based recommendations
    content_recs = content_based_recommend(movie_title, top_n*2)

    # Collaborative recommendations
    movie_id = movies[movies['title']==movie_title]['id'].values[0]
    collab_recs = collaborative_recommend(movie_id, top_n*2)

    # Combine and remove duplicates
    combined_recs = list(pd.Series(list(content_recs) + list(collab_recs)).drop_duplicates())
    return combined_recs[:top_n]

# ---------------------------
# Step 5: Optional visualization
# ---------------------------
def visualize_recommendations(movie_title, top_n=5):
    idx = indices[movie_title]
    top_idx = [idx] + [indices[m] for m in content_based_recommend(movie_title, top_n)]
    sns.heatmap(cosine_sim[np.ix_(top_idx, top_idx)], annot=True, cmap='coolwarm', xticklabels=[movies['title'].iloc[i] for i in top_idx], yticklabels=[movies['title'].iloc[i] for i in top_idx])
    plt.title(f"Content Similarity for '{movie_title}' and Top Recommendations")
    plt.show()

# ---------------------------
# Step 6: Run hybrid system
# ---------------------------
if __name__ == "__main__":
    movie_name = input("Enter a movie title: ")
    recommendations = hybrid_recommend(movie_name)

    if recommendations != []:
        print(f"\nHybrid recommendations for '{movie_name}':")
        for i, title in enumerate(recommendations, 1):
            print(f"{i}. {title}")

        # Optional: visualize top recommendations
        visualize = input("\nDo you want to visualize similarity? (y/n): ")
        if visualize.lower() == 'y':
            visualize_recommendations(movie_name)
