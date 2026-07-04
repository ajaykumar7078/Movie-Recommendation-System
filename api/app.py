"""Flask REST API for Movie Recommendation System."""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from src.recommender import MovieRecommender

app = Flask(__name__)
CORS(app)

# Initialize recommender
recommender = MovieRecommender(data_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
recommender.load_and_preprocess()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommendation System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, system-ui, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 2rem; }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #f7971e, #ffd200);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 { margin-bottom: 1rem; color: #f7971e; }
        .form-group { margin-bottom: 1rem; }
        label { display: block; margin-bottom: 0.5rem; color: #ccc; }
        input, select {
            width: 100%;
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid #555;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 1rem;
        }
        button {
            background: linear-gradient(90deg, #f7971e, #ffd200);
            color: #000;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.05); }
        .results { margin-top: 1rem; }
        .movie-item {
            padding: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .movie-item:last-child { border-bottom: none; }
        .movie-title { font-weight: 600; color: #fff; }
        .movie-genres { color: #888; font-size: 0.85rem; }
        .movie-score { color: #ffd200; font-size: 0.9rem; }
        .genre-tag {
            display: inline-block;
            background: rgba(247,151,30,0.2);
            color: #f7971e;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            margin: 0.25rem;
            cursor: pointer;
        }
        .genre-tag:hover { background: rgba(247,151,30,0.4); }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }
        .stat-value { font-size: 2rem; font-weight: 700; color: #ffd200; }
        .stat-label { color: #888; font-size: 0.85rem; margin-top: 0.25rem; }
        .search-section { display: flex; gap: 0.5rem; }
        .search-section input { flex: 1; }
        .error { color: #ff6b6b; padding: 1rem; }
        footer { text-align: center; color: #555; margin-top: 3rem; font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍿 Movie Recommendation System</h1>
        <p class="subtitle">Discover your next favorite movie</p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{{ movie_count }}</div>
                <div class="stat-label">Movies</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ rating_count }}</div>
                <div class="stat-label">Ratings</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ genre_count }}</div>
                <div class="stat-label">Genres</div>
            </div>
        </div>

        <div class="card">
            <h2>🎯 Content-Based Recommendations</h2>
            <p style="color:#888;margin-bottom:1rem;">Get recommendations similar to a movie you love</p>
            <div class="search-section">
                <input type="text" id="movieSearch" placeholder="Search or type a movie title..." list="movieList">
                <datalist id="movieList">
                    {% for movie in movies %}
                    <option value="{{ movie }}">
                    {% endfor %}
                </datalist>
                <button onclick="getContentRecs()">Get Recs</button>
            </div>
            <div id="contentResults" class="results"></div>
        </div>

        <div class="card">
            <h2>🎭 Genre-Based Recommendations</h2>
            <p style="color:#888;margin-bottom:1rem;">Explore top-rated movies by genre</p>
            <div style="margin-bottom:1rem;">
                {% for genre in genres %}
                <span class="genre-tag" onclick="getGenreRecs('{{ genre }}')">{{ genre }}</span>
                {% endfor %}
            </div>
            <div id="genreResults" class="results"></div>
        </div>

        <div class="card">
            <h2>🔗 Hybrid Recommendations</h2>
            <p style="color:#888;margin-bottom:1rem;">Smart recommendations combining content, ratings & popularity</p>
            <button onclick="getHybridRecs()">🎲 Surprise Me!</button>
            <div id="hybridResults" class="results"></div>
        </div>
    </div>

    <footer>Built with ❤️ by Ajay • Movie Recommendation System</footer>

    <script>
        function getContentRecs() {
            const movie = document.getElementById("movieSearch").value;
            if (!movie) return alert("Please enter a movie title");
            document.getElementById("contentResults").innerHTML = "Loading...";
            fetch("/api/recommend?movie=" + encodeURIComponent(movie) + "&n=8")
                .then(r => r.json()).then(renderResults("contentResults"))
                .catch(e => document.getElementById("contentResults").innerHTML = "<div class='error'>" + e.message + "</div>");
        }

        function getGenreRecs(genre) {
            document.getElementById("genreResults").innerHTML = "Loading...";
            fetch("/api/recommend?genre=" + encodeURIComponent(genre) + "&n=8")
                .then(r => r.json()).then(renderResults("genreResults"))
                .catch(e => document.getElementById("genreResults").innerHTML = "<div class='error'>" + e.message + "</div>");
        }

        function getHybridRecs() {
            document.getElementById("hybridResults").innerHTML = "Loading...";
            fetch("/api/recommend/hybrid?n=8")
                .then(r => r.json()).then(renderResults("hybridResults"))
                .catch(e => document.getElementById("hybridResults").innerHTML = "<div class='error'>" + e.message + "</div>");
        }

        function renderResults(containerId) {
            return function(data) {
                const container = document.getElementById(containerId);
                if (data.error || data.length === 0) {
                    container.innerHTML = "<div class='error'>" + (data.error || "No results found") + "</div>";
                    return;
                }
                let html = "";
                data.forEach(function(item) {
                    const score = item.avg_rating ? "⭐ " + item.avg_rating.toFixed(1) : (item.similarity_score ? item.similarity_score.toFixed(1) + "% match" : "");
                    html += "<div class='movie-item'>";
                    html += "<div class='movie-title'>" + item.title + "</div>";
                    html += "<div class='movie-genres'>" + (item.genres || "") + "</div>";
                    if (score) html += "<div class='movie-score'>" + score + "</div>";
                    html += "</div>";
                });
                container.innerHTML = html;
            };
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Render the main dashboard."""
    try:
        all_genres = sorted(set(
            g for glist in recommender.movies["genres"].str.split("|")
            for g in glist
        ))
        movie_list = sorted(recommender.movies["title"].tolist())
        return render_template_string(
            HTML_TEMPLATE,
            movie_count=len(recommender.movies),
            rating_count=len(recommender.ratings),
            genre_count=len(all_genres),
            movies=movie_list,
            genres=all_genres
        )
    except Exception as e:
        return f"<pre>Error loading dashboard: {e}</pre>"


@app.route("/api/recommend", methods=["GET"])
def recommend():
    """Get recommendations by movie title or genre."""
    movie_title = request.args.get("movie")
    genre = request.args.get("genre")
    n = int(request.args.get("n", 10))

    try:
        if movie_title:
            recs = recommender.content_based_recommendations(movie_title, n)
        elif genre:
            recs = recommender.genre_based_recommendations(genre, n)
        else:
            recs = recommender.hybrid_recommendations(n=n)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommend/hybrid", methods=["GET"])
def hybrid():
    """Get hybrid recommendations."""
    n = int(request.args.get("n", 10))
    try:
        recs = recommender.hybrid_recommendations(n=n)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def search():
    """Search movies by title."""
    q = request.args.get("q", "")
    try:
        results = recommender.search_movies(q)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """Get dataset statistics."""
    return jsonify({
        "movies": len(recommender.movies),
        "ratings": len(recommender.ratings),
        "genres": len(set(g for glist in recommender.movies["genres"].str.split("|") for g in glist)),
        "users": recommender.ratings["userId"].nunique()
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "Movie Recommendation System"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
