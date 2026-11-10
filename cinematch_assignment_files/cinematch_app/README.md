# CineMatch

A Streamlit movie recommendation application for Assignment 3. It uses the TMDB API, TF-IDF content similarity, and VADER sentiment analysis.

## Files

- `app.py`: application source code
- `requirements.txt`: Python dependencies
- `.streamlit/secrets.toml.example`: example local API-token configuration
- `.gitignore`: prevents the local secret file from being committed

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install packages with `pip install -r requirements.txt`.
5. Create `.streamlit/secrets.toml` using `.streamlit/secrets.toml.example` as a template.
6. Put your TMDB API Read Access Token in `secrets.toml`.
7. Run `streamlit run app.py`.

Do not upload `secrets.toml` to GitHub or include it in a public submission.
