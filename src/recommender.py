import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")


class MovieRecommender:
    """Content-based and collaborative movie recommendation engine."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.movies: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.knn_model = None
        self.indices = None

    def load_and_preprocess(self):
        """Load data and build content-based similarity matrix."""
        from .data_loader import MovieDataLoader

        loader = MovieDataLoader(self.data_dir)
        self.movies, self.ratings = loader.load()

        # Build TF-IDF on genres
        tfidf = TfidfVectorizer(
            tokenizer=lambda x: x.split("|"),
            lowercase=False,
            max_features=500
        )
        tfidf_matrix = tfidf.fit_transform(self.movies["genres"])
        self.tfidf_matrix = tfidf_matrix

        # Cosine similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Create series mapping titles to indices
        self.indices = pd.Series(
            self.movies.index, index=self.movies["title"]
        ).drop_duplicates()

        # Build KNN model for collaborative filtering
        self._build_knn(loader)

        print(f"Recommender ready — {len(self.movies)} movies loaded")

    def _build_knn(self, loader):
        """Build KNN model for collaborative filtering using user-item matrix."""
        user_item = loader.get_user_item_matrix()
        # Train on movie features transposed for item-based CF
        movie_features = user_item.T.values
        if movie_features.shape[0] > 2:
            self.knn_model = NearestNeighbors(
                metric="cosine", algorithm="brute", n_neighbors=min(10, movie_features.shape[0])
            )
            self.knn_model.fit(movie_features)

    def content_based_recommendations(
        self, movie_title: str, n_recommendations: int = 10
    ) -> List[dict]:
        """Get content-based recommendations using TF-IDF cosine similarity."""
        if movie_title not in self.indices:
            # Fuzzy match
            matches = [t for t in self.indices.index if movie_title.lower() in t.lower()]
            if not matches:
                return [{"error": f"Movie '{movie_title}' not found. Try: {', '.join(self.movies['title'].head(5).values)}"}]
            movie_title = matches[0]

        idx = self.indices[movie_title]

        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip the first one (itself), take top N
        sim_scores = sim_scores[1 : n_recommendations + 1]
        movie_indices = [i[0] for i in sim_scores]
        scores = [round(i[1] * 100, 1) for i in sim_scores]

        results = self.movies.iloc[movie_indices].copy()
        results["similarity_score"] = scores

        return results[["title", "genres", "similarity_score"]].to_dict("records")

    def genre_based_recommendations(self, genre: str, n: int = 10) -> List[dict]:
        """Get top-rated movies in a specific genre."""
        if self.ratings is None:
            self.load_and_preprocess()

        genre_movies = self.movies[
            self.movies["genres"].str.contains(genre, case=False, na=False)
        ]

        if genre_movies.empty:
            return [{"error": f"No movies found for genre '{genre}'"}]

        # Merge with ratings
        merged = genre_movies.merge(
            self.ratings.groupby("movieId").agg(
                avg_rating=("rating", "mean"),
                num_ratings=("rating", "count")
            ).reset_index(),
            on="movieId"
        )

        # Filter movies with at least 3 ratings
        merged = merged[merged["num_ratings"] >= 3]
        merged = merged.sort_values("avg_rating", ascending=False).head(n)

        return merged[["title", "genres", "avg_rating", "num_ratings"]].to_dict("records")

    def hybrid_recommendations(
        self, movie_title: str = None, genre: str = None, n: int = 10
    ) -> List[dict]:
        """Hybrid: Combine content-based scores with popularity."""
        if movie_title:
            cb_results = self.content_based_recommendations(movie_title, n)
            if cb_results and "error" not in cb_results[0]:
                return cb_results

        if genre:
            return self.genre_based_recommendations(genre, n)

        # Fallback: top rated movies overall
        stats = self.ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count")
        ).reset_index()
        stats = stats[stats["num_ratings"] >= 3].sort_values("avg_rating", ascending=False)
        top = stats.head(n).merge(self.movies, on="movieId")
        return top[["title", "genres", "avg_rating", "num_ratings"]].to_dict("records")

    def search_movies(self, query: str) -> List[dict]:
        """Search movies by title."""
        mask = self.movies["title"].str.contains(query, case=False, na=False)
        results = self.movies[mask].head(10)
        return results[["movieId", "title", "genres"]].to_dict("records")
