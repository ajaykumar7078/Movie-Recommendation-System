"""Streamlit web app for the Movie Recommendation System."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.data_loader import load_movies, get_all_titles, get_movie_by_title
from src.recommender import ContentBasedRecommender

st.set_page_config(page_title="Movie Recommender", page_icon=":movie_camera:", layout="centered")

@st.cache_resource
def load_recommender():
    df = load_movies()
    recommender = ContentBasedRecommender()
    recommender.fit(df)
    return df, recommender

df, recommender = load_recommender()
all_titles = get_all_titles(df)

st.title(":movie_camera: Movie Recommendation System")
st.markdown("Find movies similar to your favorites using **content-based filtering** with TF-IDF and cosine similarity.")
st.divider()

st.subheader("Choose a movie you like")
col1, col2 = st.columns([3, 1])

with col1:
    default_idx = all_titles.index("Inception") if "Inception" in all_titles else 0
    selected_title = st.selectbox("Select a movie:", options=all_titles, index=default_idx, label_visibility="collapsed")

with col2:
    top_n = st.number_input("How many?", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("Finding similar movies..."):
        movie = get_movie_by_title(df, selected_title)
        if movie is not None:
            with st.container(border=True):
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.markdown("### :clapper:")
                with col_b:
                    st.markdown(f"### {movie['title']}")
                    st.markdown(f":label: **Genres:** {movie['genres']}")
                    overview = movie['overview']
                    st.caption(overview[:200] + "..." if len(overview) > 200 else overview)
            st.divider()
            st.subheader("You might also like...")
            recommendations = recommender.recommend(df, movie["title"], top_n=top_n)
            if not recommendations:
                st.warning("No recommendations found. Try another movie.")
            else:
                for i, rec in enumerate(recommendations, 1):
                    score = rec['similarity_score'] * 100
                    with st.container(border=True):
                        cols = st.columns([1, 4, 1])
                        with cols[0]:
                            st.markdown(f"### {i}")
                        with cols[1]:
                            st.markdown(f"**{rec['title']}**")
                            st.markdown(f":label: {rec['genres']}")
                            ov = rec['overview']
                            st.caption(ov[:150] + "..." if len(ov) > 150 else ov)
                        with cols[2]:
                            st.markdown(f"### {score:.0f}%")
                            st.caption("match")

with st.sidebar:
    st.markdown("## :mag: About")
    st.markdown("""
    This system uses **TF-IDF vectorization** to convert movie descriptions and genres into numerical vectors, then finds similar movies using **cosine similarity**.

    **Tech Stack:**
    - Python, Scikit-learn, Pandas, NumPy
    - Streamlit (UI), Flask (API)
    - Docker, Render (Deployment)
    """)
    st.divider()
    st.markdown(f":bar_chart: **Dataset:** {len(df)} movies")
