import streamlit as st
import pandas as pd
import sys, os, requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommender import MovieRecommender

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🍿",
    layout="wide"
)

# Initialize
@st.cache_resource
def load_recommender():
    rec = MovieRecommender(data_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    rec.load_and_preprocess()
    return rec

recommender = load_recommender()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/popcorn.png", width=80)
st.sidebar.title("🍿 Movie Recommender")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Choose Mode",
    ["🎯 Content-Based", "🎭 Genre Explorer", "🔗 Hybrid Picks", "📊 Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by **Ajay**")
st.sidebar.caption("Using TF-IDF + Cosine Similarity + KNN")

# Stats bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎬 Movies", len(recommender.movies))
col2.metric("⭐ Ratings", len(recommender.ratings))
col3.metric("👥 Users", recommender.ratings["userId"].nunique())
all_genres = set(g for glist in recommender.movies["genres"].str.split("|") for g in glist)
col4.metric("🏷️ Genres", len(all_genres))

if mode == "🎯 Content-Based":
    st.header("🎯 Content-Based Recommendations")
    st.caption("Get movies similar to one you already love — powered by TF-IDF & Cosine Similarity")

    movie_list = sorted(recommender.movies["title"].tolist())
    selected = st.selectbox("Select a movie you like:", movie_list)
    n = st.slider("Number of recommendations", 3, 20, 8)

    if st.button("Get Recommendations 🚀", type="primary"):
        with st.spinner("Finding similar movies..."):
            recs = recommender.content_based_recommendations(selected, n)
            if recs and "error" not in recs[0]:
                df = pd.DataFrame(recs)
                df["similarity_score"] = df["similarity_score"].apply(lambda x: f"{x:.1f}%")
                st.success(f"Movies similar to **{selected}**")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.error(recs[0]["error"] if recs else "No results")

elif mode == "🎭 Genre Explorer":
    st.header("🎭 Genre Explorer")
    st.caption("Discover top-rated movies in your favorite genre")

    genres = sorted(set(g for glist in recommender.movies["genres"].str.split("|") for g in glist))
    selected_genre = st.selectbox("Pick a genre:", genres)
    n = st.slider("Number of movies", 3, 20, 10)

    with st.spinner(f"Finding top {selected_genre} movies..."):
        recs = recommender.genre_based_recommendations(selected_genre, n)
        if recs and "error" not in recs[0]:
            df = pd.DataFrame(recs)
            df["avg_rating"] = df["avg_rating"].apply(lambda x: f"⭐ {x:.1f}")
            st.success(f"Top {selected_genre} Movies")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("Not enough data for this genre. Try another!")

elif mode == "🔗 Hybrid Picks":
    st.header("🔗 Hybrid Recommendations")
    st.caption("Smart picks combining content similarity, user ratings & popularity")
    n = st.slider("Number of picks", 3, 20, 10)

    if st.button("🎲 Generate Picks", type="primary"):
        with st.spinner("Crunching data..."):
            recs = recommender.hybrid_recommendations(n=n)
            if recs:
                df = pd.DataFrame(recs)
                if "avg_rating" in df.columns:
                    df["avg_rating"] = df["avg_rating"].apply(lambda x: f"⭐ {x:.1f}")
                st.success("Your personalized picks!")
                st.dataframe(df, use_container_width=True, hide_index=True)

elif mode == "📊 Analytics":
    st.header("📊 Data Analytics Dashboard")
    st.caption("Explore the movie dataset")

    tab1, tab2, tab3 = st.tabs(["Genre Distribution", "Rating Distribution", "Top Movies"])

    with tab1:
        st.subheader("Genre Distribution")
        genre_counts = {}
        for glist in recommender.movies["genres"].str.split("|"):
            for g in glist:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_df = pd.DataFrame(
            sorted(genre_counts.items(), key=lambda x: x[1], reverse=True),
            columns=["Genre", "Count"]
        )
        st.bar_chart(genre_df.set_index("Genre"))

    with tab2:
        st.subheader("Rating Distribution")
        rating_hist = recommender.ratings["rating"].value_counts().sort_index()
        st.bar_chart(rating_hist)
        st.caption(f"Average rating: {recommender.ratings['rating'].mean():.2f} ⭐")

    with tab3:
        st.subheader("Highest Rated Movies")
        stats = recommender.ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count")
        ).reset_index()
        stats = stats[stats["num_ratings"] >= 3].sort_values("avg_rating", ascending=False).head(10)
        top = stats.merge(recommender.movies, on="movieId")
        top["avg_rating"] = top["avg_rating"].apply(lambda x: f"⭐ {x:.1f}")
        st.dataframe(top[["title", "genres", "avg_rating", "num_ratings"]], use_container_width=True, hide_index=True)
