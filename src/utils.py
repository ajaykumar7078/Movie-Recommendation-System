import pandas as pd
import numpy as np


def format_recommendations(recs: list, detailed: bool = False) -> str:
    """Format recommendations as a readable string."""
    if not recs:
        return "No recommendations found."

    if "error" in recs[0]:
        return recs[0]["error"]

    lines = []
    for i, rec in enumerate(recs, 1):
        title = rec.get("title", "Unknown")
        genres = rec.get("genres", "")
        sim = rec.get("similarity_score")
        rating = rec.get("avg_rating")

        if sim:
            lines.append(f"  {i}. **{title}** ({genres}) — {sim:.1f}% match")
        elif rating:
            lines.append(f"  {i}. **{title}** ({genres}) — ⭐ {rating:.1f} avg")
        else:
            lines.append(f"  {i}. **{title}** ({genres})")

    return "\n".join(lines)


def get_popular_genres(movies: pd.DataFrame) -> list:
    """Get list of unique genres sorted by frequency."""
    all_genres = []
    for g in movies["genres"].str.split("|"):
        all_genres.extend(g)

    from collections import Counter
    return [g for g, _ in Counter(all_genres).most_common()]
