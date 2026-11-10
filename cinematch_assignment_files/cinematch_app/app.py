"""CineMatch: a TMDB-powered movie recommendation app for Assignment 3.

The app implements:
- TMDB API data retrieval
- Content-based recommendations using TF-IDF and cosine similarity
- Sentiment-based matching using VADER on plot descriptions
- Required temporal, quality, content, personnel, and genre filters
- Three enhanced features: natural-language requests, similar-movie discovery,
  and trending movies
"""

from datetime import date
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# -----------------------------------------------------------------------------
# Page setup and configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"
CURRENT_YEAR = date.today().year
MAX_DISCOVERY_RESULTS = 20  # One TMDB page, before detail enrichment.
MAX_SIMILAR_RESULTS = 10

GENRES = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37,
}

LANGUAGES = {
    "Any language": None,
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Chinese": "zh",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Arabic": "ar",
}

CERTIFICATIONS = ["G", "PG", "PG-13", "R", "NC-17", "NR"]


# -----------------------------------------------------------------------------
# API authentication and TMDB requests
# -----------------------------------------------------------------------------
def get_tmdb_token() -> str:
    """Read the TMDB API Read Access Token from Streamlit secrets or environment."""
    try:
        token = st.secrets.get("TMDB_API_TOKEN", "")
    except Exception:
        token = ""
    return token or os.getenv("TMDB_API_TOKEN", "")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_tmdb_get(
    path: str,
    params_items: Tuple[Tuple[str, Any], ...],
    token: str,
) -> Dict[str, Any]:
    """Make a cached GET request to TMDB. The token is passed in the header."""
    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
    }
    params = dict(params_items)
    response = requests.get(
        f"{TMDB_BASE_URL}{path}",
        headers=headers,
        params=params,
        timeout=25,
    )

    if response.status_code == 401:
        raise RuntimeError(
            "TMDB did not accept the API token. Check that you entered your "
            "TMDB API Read Access Token correctly."
        )
    if response.status_code == 429:
        raise RuntimeError("TMDB rate limit reached. Please wait a moment and try again.")
    response.raise_for_status()
    return response.json()


def tmdb_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate credentials and call the cached TMDB request function."""
    token = get_tmdb_token()
    if not token:
        raise RuntimeError(
            "TMDB API token is missing. Add TMDB_API_TOKEN to your local "
            ".streamlit/secrets.toml file or your hosting platform's Secrets settings."
        )
    clean_params = params or {}
    params_items = tuple(sorted(clean_params.items()))
    return cached_tmdb_get(path, params_items, token)


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_details(movie_id: int, token: str) -> Dict[str, Any]:
    """Retrieve a movie and its keywords, credits, and release-date certifications."""
    return cached_tmdb_get(
        f"/movie/{movie_id}",
        (("append_to_response", "keywords,credits,release_dates"),),
        token,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def search_tmdb_person(name: str, token: str) -> Optional[Dict[str, Any]]:
    """Return the best TMDB person-search match for a name."""
    if not name.strip():
        return None
    data = cached_tmdb_get(
        "/search/person",
        tuple(sorted({"query": name.strip(), "include_adult": "false"}.items())),
        token,
    )
    results = data.get("results", [])
    return results[0] if results else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_list(source: str, params_items: Tuple[Tuple[str, Any], ...], token: str) -> List[Dict[str, Any]]:
    """Retrieve one page of discover or trending movie results."""
    movies: List[Dict[str, Any]] = []
    endpoint = "/trending/movie/week" if source == "Trending this week" else "/discover/movie"

    for page in (1,):
        params = dict(params_items)
        params["page"] = page
        page_data = cached_tmdb_get(endpoint, tuple(sorted(params.items())), token)
        movies.extend(page_data.get("results", []))

    # Keep IDs unique and cap the amount of detail enrichment for predictable performance.
    unique_movies = {}
    for movie in movies:
        unique_movies[movie.get("id")] = movie
    return list(unique_movies.values())[:MAX_DISCOVERY_RESULTS]


@st.cache_data(ttl=3600, show_spinner=False)
def get_similar_movie_list(movie_id: int, token: str) -> List[Dict[str, Any]]:
    """Retrieve movies TMDB identifies as similar to the selected movie."""
    data = cached_tmdb_get(
        f"/movie/{movie_id}/similar",
        (("page", 1), ("language", "en-US")),
        token,
    )
    return data.get("results", [])[:MAX_SIMILAR_RESULTS]


# -----------------------------------------------------------------------------
# Movie data preparation
# -----------------------------------------------------------------------------
def get_certification(release_dates_data: Dict[str, Any]) -> str:
    """Use the first available US movie certification; use NR when unavailable."""
    countries = release_dates_data.get("results", [])
    for country in countries:
        if country.get("iso_3166_1") == "US":
            for release in country.get("release_dates", []):
                certification = (release.get("certification") or "").strip()
                if certification:
                    return certification.upper()
    return "NR"


def normalize_movie(details: Dict[str, Any]) -> Dict[str, Any]:
    """Convert TMDB's nested response into a consistent movie dictionary."""
    credits = details.get("credits", {})
    cast = credits.get("cast", [])
    crew = credits.get("crew", [])
    keyword_data = details.get("keywords", {})

    if isinstance(keyword_data, dict):
        keyword_list = keyword_data.get("keywords", [])
    else:
        keyword_list = []

    directors = [
        person.get("name", "")
        for person in crew
        if person.get("job") == "Director" and person.get("name")
    ]

    release_date = details.get("release_date") or ""
    year = None
    if len(release_date) >= 4 and release_date[:4].isdigit():
        year = int(release_date[:4])

    return {
        "id": details.get("id"),
        "title": details.get("title") or details.get("original_title") or "Untitled",
        "year": year,
        "release_date": release_date,
        "overview": details.get("overview") or "No plot summary is available.",
        "runtime": details.get("runtime") or 0,
        "certification": get_certification(details.get("release_dates", {})),
        "genres": [genre.get("name", "") for genre in details.get("genres", []) if genre.get("name")],
        "keywords": [item.get("name", "") for item in keyword_list if item.get("name")],
        "cast": [person.get("name", "") for person in cast if person.get("name")],
        "directors": directors,
        "original_language": details.get("original_language") or "",
        "vote_average": float(details.get("vote_average") or 0),
        "vote_count": int(details.get("vote_count") or 0),
        "popularity": float(details.get("popularity") or 0),
        "revenue": int(details.get("revenue") or 0),
        "poster_path": details.get("poster_path"),
    }


def enrich_movies(raw_movies: List[Dict[str, Any]], token: str) -> List[Dict[str, Any]]:
    """Fetch full movie details and skip individual records that fail to load."""
    enriched = []
    for raw_movie in raw_movies:
        movie_id = raw_movie.get("id")
        if not movie_id:
            continue
        try:
            details = get_movie_details(int(movie_id), token)
            enriched.append(normalize_movie(details))
        except (requests.RequestException, RuntimeError, ValueError):
            # One unavailable movie should not stop the whole recommendation request.
            continue
    return enriched


# -----------------------------------------------------------------------------
# Filtering and recommendation algorithms
# -----------------------------------------------------------------------------
@st.cache_resource
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    """Create one reusable VADER analyzer for plot-description sentiment."""
    return SentimentIntensityAnalyzer()


def get_plot_sentiment(plot: str) -> Tuple[float, str]:
    """Analyze plot-description tone with VADER; this is not a movie-quality score."""
    if not plot or plot == "No plot summary is available.":
        return 0.0, "Neutral"
    compound = get_sentiment_analyzer().polarity_scores(plot)["compound"]
    if compound >= 0.15:
        label = "Positive"
    elif compound <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"
    return float(compound), label


def apply_filters(
    movies: List[Dict[str, Any]],
    filters: Dict[str, Any],
    actor_name: str = "",
    director_name: str = "",
) -> List[Dict[str, Any]]:
    """Apply the assignment's temporal, quality, content, personnel, and genre filters."""
    filtered = []
    year_min, year_max = filters["year_range"]

    if filters["decade"] != "Any decade":
        decade_start = int(filters["decade"][:4])
        decade_end = min(decade_start + 9, CURRENT_YEAR)
        year_min = max(year_min, decade_start)
        year_max = min(year_max, decade_end)

    if year_min > year_max:
        return []

    actor_key = actor_name.casefold().strip()
    director_key = director_name.casefold().strip()

    for movie in movies:
        year = movie.get("year")
        if year is None or not (year_min <= year <= year_max):
            continue

        if movie["vote_average"] < filters["min_rating"]:
            continue
        if movie["vote_count"] < filters["min_votes"]:
            continue

        runtime = movie.get("runtime", 0)
        runtime_min, runtime_max = filters["runtime_range"]
        if runtime < runtime_min or runtime > runtime_max:
            continue

        if filters["certifications"] and movie["certification"] not in filters["certifications"]:
            continue

        selected_language = filters["language"]
        if selected_language and movie["original_language"] != selected_language:
            continue

        selected_genres = filters["genres"]
        if selected_genres:
            movie_genres = set(movie["genres"])
            if filters["genre_logic"] == "ALL" and not set(selected_genres).issubset(movie_genres):
                continue
            if filters["genre_logic"] == "ANY" and not set(selected_genres).intersection(movie_genres):
                continue

        # Check the requested people against the actual cast and director credits.
        cast_names = [name.casefold() for name in movie["cast"]]
        director_names = [name.casefold() for name in movie["directors"]]
        if actor_key and not any(actor_key in name for name in cast_names):
            continue
        if director_key and not any(director_key in name for name in director_names):
            continue

        filtered.append(movie)

    return filtered


def movie_document(movie: Dict[str, Any]) -> str:
    """Build the text representation used by content-based similarity."""
    parts = [
        movie.get("overview", ""),
        " ".join(movie.get("genres", [])),
        " ".join(movie.get("keywords", [])),
        " ".join(movie.get("cast", [])[:5]),
        " ".join(movie.get("directors", [])),
    ]
    return " ".join(parts).strip()


def rank_recommendations(
    movies: List[Dict[str, Any]],
    natural_query: str,
    sentiment_preference: str,
) -> List[Dict[str, Any]]:
    """Combine TF-IDF content similarity and VADER sentiment matching."""
    if not movies:
        return []

    documents = [movie_document(movie) or movie["title"] for movie in movies]
    if natural_query.strip():
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
            matrix = vectorizer.fit_transform(documents + [natural_query.strip()])
            content_scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        except ValueError:
            content_scores = [0.0] * len(movies)
    else:
        # With no text request, popularity provides a sensible discovery order.
        import math

        popularity_values = [math.log1p(max(0.0, movie["popularity"])) for movie in movies]
        max_popularity = max(popularity_values) if popularity_values else 0.0
        content_scores = [value / max_popularity if max_popularity else 0.0 for value in popularity_values]

    scored_movies = []
    for movie, content_score in zip(movies, content_scores):
        compound, sentiment_label = get_plot_sentiment(movie["overview"])
        movie_copy = dict(movie)
        movie_copy["sentiment_score"] = compound
        movie_copy["sentiment_label"] = sentiment_label

        if sentiment_preference == "Any":
            final_score = float(content_score)
            movie_copy["sentiment_match"] = "Not used"
        else:
            sentiment_match = 1.0 if sentiment_label == sentiment_preference else 0.0
            final_score = (0.80 * float(content_score)) + (0.20 * sentiment_match)
            movie_copy["sentiment_match"] = "Match" if sentiment_match else "Different tone"

        movie_copy["content_score"] = float(content_score)
        movie_copy["recommendation_score"] = final_score
        scored_movies.append(movie_copy)

    return sorted(scored_movies, key=lambda item: item["recommendation_score"], reverse=True)


def compare_similar_movies(seed_movie: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank TMDB similar titles with TF-IDF and produce feature-overlap explanations."""
    if not candidates:
        return []

    documents = [movie_document(seed_movie) or seed_movie["title"]]
    documents.extend(movie_document(movie) or movie["title"] for movie in candidates)

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        matrix = vectorizer.fit_transform(documents)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError:
        similarities = [0.0] * len(candidates)

    seed_genres = set(seed_movie.get("genres", []))
    seed_keywords = {word.casefold() for word in seed_movie.get("keywords", [])}
    seed_people = {name.casefold() for name in seed_movie.get("cast", [])}
    seed_people.update(name.casefold() for name in seed_movie.get("directors", []))

    output = []
    for movie, score in zip(candidates, similarities):
        shared_genres = sorted(seed_genres.intersection(movie.get("genres", [])))
        shared_keywords = sorted(
            seed_keywords.intersection(word.casefold() for word in movie.get("keywords", []))
        )
        candidate_people = {name.casefold() for name in movie.get("cast", [])}
        candidate_people.update(name.casefold() for name in movie.get("directors", []))
        shared_people = sorted(seed_people.intersection(candidate_people))

        reasons = []
        if shared_genres:
            reasons.append("Shared genres: " + ", ".join(shared_genres[:4]))
        if shared_keywords:
            reasons.append("Shared keywords: " + ", ".join(shared_keywords[:4]))
        if shared_people:
            reasons.append("Shared cast/crew: " + ", ".join(shared_people[:3]))
        if not reasons:
            reasons.append("The plot and movie attributes have similar TF-IDF text patterns.")

        movie_copy = dict(movie)
        movie_copy["similarity_score"] = float(score)
        movie_copy["similarity_reasons"] = reasons
        output.append(movie_copy)

    return sorted(output, key=lambda item: item["similarity_score"], reverse=True)


# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------
def format_money(amount: int) -> str:
    return f"${amount:,.0f}" if amount else "Not available"


def display_movie_card(movie: Dict[str, Any], show_recommendation: bool = True) -> None:
    """Display key movie information in a consistent card-like layout."""
    poster_column, detail_column = st.columns([1, 3])

    with poster_column:
        if movie.get("poster_path"):
            st.image(f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}")
        else:
            st.markdown("**Poster unavailable**")

    with detail_column:
        year_text = str(movie["year"]) if movie.get("year") else "Year unavailable"
        st.subheader(f"{movie['title']} ({year_text})")

        if show_recommendation:
            st.caption(f"Recommendation match: {movie.get('recommendation_score', 0) * 100:.1f}%")
        elif "similarity_score" in movie:
            st.caption(f"Content similarity: {movie['similarity_score'] * 100:.1f}%")

        genres = ", ".join(movie.get("genres", [])) or "Not listed"
        st.write(f"**Genres:** {genres}")
        st.write(
            f"**Rating:** {movie['vote_average']:.1f}/10  |  "
            f"**Votes:** {movie['vote_count']:,}  |  "
            f"**Runtime:** {movie['runtime']} min  |  "
            f"**Certification:** {movie['certification']}"
        )
        st.write(movie.get("overview", "No plot summary is available."))

        with st.expander("More movie information"):
            st.write(f"**Release date:** {movie.get('release_date') or 'Not available'}")
            st.write(f"**Original language:** {movie.get('original_language', '').upper() or 'Not available'}")
            st.write(f"**Director(s):** {', '.join(movie.get('directors', [])) or 'Not listed'}")
            st.write(f"**Cast:** {', '.join(movie.get('cast', [])[:8]) or 'Not listed'}")
            st.write(f"**Keywords:** {', '.join(movie.get('keywords', [])[:10]) or 'Not listed'}")
            st.write(f"**Box office revenue:** {format_money(movie.get('revenue', 0))}")
            if "sentiment_label" in movie:
                st.write(
                    f"**Plot sentiment:** {movie['sentiment_label']} "
                    f"(VADER compound score: {movie['sentiment_score']:.2f})"
                )
                if movie.get("sentiment_match") != "Not used":
                    st.write(f"**Sentiment preference:** {movie['sentiment_match']}")
            if movie.get("similarity_reasons"):
                st.write("**Why it is similar:**")
                for reason in movie["similarity_reasons"]:
                    st.write(f"- {reason}")

    st.divider()


# -----------------------------------------------------------------------------
# Streamlit interface
# -----------------------------------------------------------------------------
st.title("🎬 CineMatch")
st.write("Find movies that match your interests, mood, and viewing preferences.")
st.caption("Movie metadata supplied by The Movie Database (TMDB). Plot sentiment is an automated text estimate, not a rating of the movie.")

if not get_tmdb_token():
    st.error(
        "TMDB API token not found. Before using the app, add your TMDB API Read Access Token "
        "as TMDB_API_TOKEN in .streamlit/secrets.toml (local) or your hosting platform's Secrets settings."
    )
    st.stop()

with st.sidebar:
    st.header("Movie preferences")
    with st.form("movie_preferences"):
        source = st.radio(
            "Movie source",
            ["Discover based on preferences", "Trending this week"],
        )
        natural_query = st.text_area(
            "Describe the movie you want",
            placeholder="Example: a clever mystery with a suspenseful plot and strong characters",
            help="Use everyday language. The app compares your request with plot descriptions, genres, keywords, cast, and directors.",
        )
        sentiment_preference = st.selectbox(
            "Preferred plot sentiment",
            ["Any", "Positive", "Neutral", "Negative"],
            help="VADER estimates the tone of the plot description. It does not measure movie quality.",
        )

        st.subheader("Release period")
        year_range = st.slider(
            "Release year range",
            min_value=1900,
            max_value=CURRENT_YEAR,
            value=(1900, CURRENT_YEAR),
        )
        decade_options = ["Any decade"] + [f"{decade}s" for decade in range((CURRENT_YEAR // 10) * 10, 1900 - 1, -10)]
        decade = st.selectbox("Decade", decade_options)

        st.subheader("Quality")
        min_rating = st.slider("Minimum TMDB rating", 0.0, 10.0, 5.0, 0.5)
        min_votes = st.number_input("Minimum vote count", min_value=0, max_value=100000, value=100, step=50)

        st.subheader("Content")
        runtime_range = st.slider("Runtime (minutes)", 0, 300, (60, 240), 5)
        certifications = st.multiselect(
            "Certification (US rating) — leave blank for any",
            CERTIFICATIONS,
        )
        language_label = st.selectbox("Original language", list(LANGUAGES.keys()))
        selected_genres = st.multiselect("Genres", list(GENRES.keys()))
        genre_logic = st.radio(
            "Genre matching",
            ["ANY", "ALL"],
            horizontal=True,
            help="ANY: match at least one selected genre. ALL: match every selected genre.",
        )

        st.subheader("People")
        actor_name = st.text_input("Actor (optional)", placeholder="e.g., Zendaya")
        director_name = st.text_input("Director (optional)", placeholder="e.g., Denis Villeneuve")

        submitted = st.form_submit_button("Find movies", type="primary")


if submitted:
    # Intersect the selected year range with the optional decade.
    effective_year_min, effective_year_max = year_range
    if decade != "Any decade":
        decade_start = int(decade[:4])
        decade_end = min(decade_start + 9, CURRENT_YEAR)
        effective_year_min = max(effective_year_min, decade_start)
        effective_year_max = min(effective_year_max, decade_end)

    if effective_year_min > effective_year_max:
        st.session_state.pop("recommendations", None)
        st.error("The selected year range and decade do not overlap. Adjust one of them and try again.")
    else:
        token = get_tmdb_token()
        actor_person = None
        director_person = None
        person_error = None

        try:
            with st.spinner("Searching TMDB and preparing recommendations..."):
                if actor_name.strip():
                    actor_person = search_tmdb_person(actor_name, token)
                    if not actor_person:
                        person_error = f"TMDB could not find an actor matching '{actor_name}'. Check the spelling or clear the actor filter."

                if not person_error and director_name.strip():
                    director_person = search_tmdb_person(director_name, token)
                    if not director_person:
                        person_error = f"TMDB could not find a director matching '{director_name}'. Check the spelling or clear the director filter."

                if person_error:
                    st.session_state.pop("recommendations", None)
                    st.error(person_error)
                else:
                    api_params: Dict[str, Any] = {
                        "language": "en-US",
                        "include_adult": "false",
                        "sort_by": "popularity.desc",
                        "primary_release_date.gte": f"{effective_year_min}-01-01",
                        "primary_release_date.lte": f"{effective_year_max}-12-31",
                        "vote_average.gte": min_rating,
                        "vote_count.gte": int(min_votes),
                    }

                    if selected_genres:
                        separator = "," if genre_logic == "ALL" else "|"
                        api_params["with_genres"] = separator.join(str(GENRES[name]) for name in selected_genres)
                    if language_label != "Any language":
                        api_params["with_original_language"] = LANGUAGES[language_label]
                    if actor_person:
                        api_params["with_cast"] = actor_person["id"]
                    if director_person:
                        # TMDB's with_crew narrows results; credits are checked below to confirm the person is a director.
                        api_params["with_crew"] = director_person["id"]

                    raw_movies = get_movie_list(
                        "Trending this week" if source == "Trending this week" else "Discover based on preferences",
                        tuple(sorted(api_params.items())) if source != "Trending this week" else tuple(),
                        token,
                    )
                    detailed_movies = enrich_movies(raw_movies, token)

                    filter_settings = {
                        "year_range": year_range,
                        "decade": decade,
                        "min_rating": min_rating,
                        "min_votes": int(min_votes),
                        "runtime_range": runtime_range,
                        "certifications": certifications,
                        "language": LANGUAGES[language_label],
                        "genres": selected_genres,
                        "genre_logic": genre_logic,
                    }
                    filtered_movies = apply_filters(
                        detailed_movies,
                        filter_settings,
                        actor_name if actor_person else "",
                        director_name if director_person else "",
                    )
                    recommendations = rank_recommendations(
                        filtered_movies,
                        natural_query,
                        sentiment_preference,
                    )
                    st.session_state["recommendations"] = recommendations
                    st.session_state["natural_query"] = natural_query
                    st.session_state["source"] = source
                    st.session_state.pop("similar_results", None)
                    st.session_state.pop("similar_seed_id", None)

        except (requests.RequestException, RuntimeError, ValueError) as error:
            st.error(f"There was a problem retrieving movie data: {error}")


# -----------------------------------------------------------------------------
# Recommendation results
# -----------------------------------------------------------------------------
recommendations = st.session_state.get("recommendations", [])
if recommendations:
    st.header("Your recommendations")
    st.write(
        f"Showing {len(recommendations)} matching movies from "
        f"**{st.session_state.get('source', 'TMDB')}**."
    )
    if st.session_state.get("natural_query", "").strip():
        st.caption("Ranking combines TF-IDF similarity to your natural-language request with the selected plot-sentiment preference.")
    else:
        st.caption("With no text request, movies are ordered using TMDB popularity. A selected plot-sentiment preference is also included in the ranking.")

    for movie in recommendations:
        display_movie_card(movie, show_recommendation=True)

    st.header("Find similar movies")
    st.write("Choose one of your recommendations to discover other titles TMDB considers similar. CineMatch ranks those titles and explains shared attributes.")
    movie_options = {
        f"{movie['title']} ({movie['year'] or 'Year unknown'})": movie
        for movie in recommendations
    }
    selected_seed_label = st.selectbox("Choose a movie", list(movie_options.keys()))
    if st.button("Find similar movies", type="secondary"):
        seed_movie = movie_options[selected_seed_label]
        try:
            with st.spinner(f"Finding movies similar to {seed_movie['title']}..."):
                token = get_tmdb_token()
                similar_raw = get_similar_movie_list(seed_movie["id"], token)
                similar_details = enrich_movies(similar_raw, token)
                st.session_state["similar_results"] = compare_similar_movies(seed_movie, similar_details)
                st.session_state["similar_seed_id"] = seed_movie["id"]
                st.session_state["similar_seed_title"] = seed_movie["title"]
        except (requests.RequestException, RuntimeError, ValueError) as error:
            st.error(f"Could not retrieve similar movies: {error}")

    similar_results = st.session_state.get("similar_results", [])
    if similar_results:
        st.subheader(f"Movies similar to {st.session_state.get('similar_seed_title', 'your selection')}")
        for movie in similar_results:
            display_movie_card(movie, show_recommendation=False)

elif "recommendations" in st.session_state:
    st.info("No movies matched all of those filters. Try widening the release years or runtime range, lowering the minimum rating or vote count, or removing a filter.")
else:
    st.info("Choose your preferences in the left panel, then select **Find movies** to get started.")


# -----------------------------------------------------------------------------
# Attribution
# -----------------------------------------------------------------------------
st.divider()
st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB. Movie data and images are provided by The Movie Database (TMDB).")
