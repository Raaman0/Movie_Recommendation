import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommendation System")

# ---------------------------
# Step 1: Load and preprocess data
# ---------------------------
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, left_on='id', right_on='movie_id')

def combine_features(row):
    return str(row['genres']) + " " + str(row['keywords']) + " " + str(row['overview']) + " " + str(row['cast']) + " " + str(row['crew'])

movies['combined_features'] = movies.apply(combine_features, axis=1)
movies['title'] = movies['title_x']
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
# Step 3: Collaborative filtering (simulated)
# ---------------------------
num_users = 50
num_movies = 200
np.random.seed(42)
user_ids = np.arange(1, num_users+1)
movie_ids_sample = movies['id'].sample(num_movies, random_state=42).values

ratings_list = []
for user in user_ids:
    for movie in movie_ids_sample:
        ratings_list.append([user, movie, np.random.randint(1,6)])

ratings = pd.DataFrame(ratings_list, columns=['user_id', 'movie_id', 'rating'])
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
    title_lower = movie_title.lower()
    matches = movies[movies['title'].str.lower().str.contains(title_lower)]
    if matches.empty:
        return []
    movie_title = matches['title'].iloc[0]
    content_recs = content_based_recommend(movie_title, top_n*2)
    movie_id = movies[movies['title']==movie_title]['id'].values[0]
    collab_recs = collaborative_recommend(movie_id, top_n*2)
    combined_recs = list(pd.Series(list(content_recs) + list(collab_recs)).drop_duplicates())
    return combined_recs[:top_n]

# ---------------------------
# Step 5: Visualization function
# ---------------------------
def visualize_recommendations(movie_title, top_n=5):
    idx = indices[movie_title]
    top_idx = [idx] + [indices[m] for m in content_based_recommend(movie_title, top_n)]
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cosine_sim[np.ix_(top_idx, top_idx)], annot=True,
                cmap='coolwarm',
                xticklabels=[movies['title'].iloc[i] for i in top_idx],
                yticklabels=[movies['title'].iloc[i] for i in top_idx],
                ax=ax)
    st.pyplot(fig)

# ---------------------------
# Step 6: Streamlit interface
# ---------------------------
movie_name = st.text_input("Enter a movie title:")
top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie title!")
    else:
        recommendations = hybrid_recommend(movie_name, top_n)
        if recommendations:
            st.success(f"Top {top_n} recommendations for '{movie_name}':")
            for i, title in enumerate(recommendations, 1):
                st.write(f"{i}. {title}")
            
            # Checkbox for visualization
            if st.checkbox("Visualize content similarity of recommendations"):
                visualize_recommendations(recommendations[0], top_n)
        else:
            st.error(f"No recommendations found for '{movie_name}'")
