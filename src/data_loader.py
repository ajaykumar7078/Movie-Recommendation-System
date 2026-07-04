import pandas as pd
import numpy as np
import os


class MovieDataLoader:
    """Load and preprocess movie and rating data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.movies: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load movies and ratings CSV files."""
        movies_path = os.path.join(self.data_dir, "movies.csv")
        ratings_path = os.path.join(self.data_dir, "ratings.csv")

        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

        # Parse genres into list
        self.movies["genres_list"] = self.movies["genres"].str.split("|")

        print(f"Loaded {len(self.movies)} movies and {len(self.ratings)} ratings")
        return self.movies, self.ratings

    def get_genre_matrix(self) -> pd.DataFrame:
        """Create a binary genre matrix (one-hot encoded)."""
        if self.movies is None:
            self.load()

        genres = set()
        for glist in self.movies["genres_list"]:
            genres.update(glist)
        genres = sorted(genres)

        genre_data = []
        for glist in self.movies["genres_list"]:
            genre_data.append([1 if g in glist else 0 for g in genres])

        genre_matrix = pd.DataFrame(genre_data, columns=genres, index=self.movies["movieId"])
        return genre_matrix

    def get_user_item_matrix(self) -> pd.DataFrame:
        """Create user-item rating matrix."""
        if self.ratings is None:
            self.load()
        return self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)

    def get_movie_stats(self) -> pd.DataFrame:
        """Compute rating statistics per movie."""
        if self.ratings is None:
            self.load()
        stats = self.ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count"),
            std_rating=("rating", "std")
        ).round(2)
        return stats
