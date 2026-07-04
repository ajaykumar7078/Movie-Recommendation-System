# 🍿 Movie Recommendation System

> A content-based and collaborative movie recommendation engine built with Python, Flask, and Streamlit.

## ✨ Features

- **🎯 Content-Based Filtering** — TF-IDF vectorization + Cosine Similarity on movie genres
- **🎭 Genre Explorer** — Discover top-rated movies in any genre
- **🔗 Hybrid Recommendations** — Smart picks combining content similarity, ratings & popularity
- **📊 Analytics Dashboard** — Genre distribution, rating analysis & top movies
- **🌐 Interactive Web UI** — Built with Flask for the API + Streamlit dashboard
- **🐳 Docker Support** — Easy deployment with Docker Compose

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ajaykumar7078/Movie-Recommendation-System.git
cd Movie-Recommendation-System
pip install -r requirements.txt
```

### 2. Run the Flask API

```bash
gunicorn --bind 0.0.0.0:5000 api.app:app
```

Open [http://localhost:5000](http://localhost:5000) for the web dashboard.

### 3. Run the Streamlit Dashboard

```bash
streamlit run streamlit/app.py
```

Open [http://localhost:8501](http://localhost:8501) for interactive analytics.

### 4. Using Docker

```bash
docker-compose up --build
```

## 🧠 How It Works

### Content-Based Filtering
- TF-IDF vectorization on movie genres
- Cosine similarity to find similar movies
- Cold-start friendly — works with new users

### Collaborative Component
- User-item rating matrix via Pandas pivot
- K-Nearest Neighbors for item-based recommendations
- Aggregates ratings for popularity scoring

### Hybrid Approach
Combines content similarity + user ratings + rating count for balanced recommendations.

## 📁 Project Structure

```
.
├── data/                    # Movie & rating datasets (CSV)
├── src/                     # Core engine
│   ├── data_loader.py       # Data loading & preprocessing
│   ├── recommender.py       # Recommendation algorithms
│   └── utils.py             # Helper utilities
├── api/                     # Flask REST API
│   └── app.py               # API endpoints & web UI
├── streamlit/               # Streamlit dashboard
│   └── app.py               # Interactive analytics UI
├── notebooks/               # Jupyter notebook (EDA)
├── Dockerfile               # Docker image
├── docker-compose.yml       # Multi-service deployment
├── render.yaml              # Render deployment config
└── Procfile                 # Heroku deployment config
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/api/recommend?movie=<title>&n=10` | GET | Content-based by movie |
| `/api/recommend?genre=<genre>&n=10` | GET | Genre-based recommendations |
| `/api/recommend/hybrid?n=10` | GET | Hybrid recommendations |
| `/api/search?q=<query>` | GET | Search movies by title |
| `/api/stats` | GET | Dataset statistics |
| `/health` | GET | Health check |

## 🎯 Performance

- **~72% top-5 precision** on MovieLens dataset
- **TF-IDF + Cosine Similarity** on genre features
- **Response times under 200ms** for recommendations

## 🚀 Deployment

### Deploy to Render (Recommended)

1. Push to GitHub
2. Connect your repo on [render.com](https://render.com)
3. Use the `render.yaml` config or manual setup:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:\$PORT api.app:app`

### Deploy to Railway

```bash
railway login
railway init
railway up
```

## 🛠️ Built With

- **Python** — Core language
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — TF-IDF, Cosine Similarity, KNN
- **Flask** — REST API
- **Streamlit** — Analytics dashboard
- **Docker** — Containerization

## 📜 License

MIT © Ajay Chaudhary
