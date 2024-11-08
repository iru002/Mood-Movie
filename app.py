from flask import Flask, render_template, request
import pandas as pd
import pickle
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

app = Flask(__name__)

# Load the model components
with open('recommendation_model.pkl', 'rb') as file:
    df, tfidf, cosine_sim = pickle.load(file)

TMDB_API_KEY = 'b4b950e128eb60dca6ca043e772d1165'
TMDB_READ_ACCESS_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNGI5NTBlMTI4ZWI2MGRjYTZjYTA0M2U3NzJkMTE2NSIsIm5iZiI6MTcyMTk4MTI5Ni45NzAyOTUsInN1YiI6IjY2OTRkOTc5ZjEyOTNmOTg2ODZiNGJmOCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.9aCofOcJNY2cyvArMC5zTD41rISUwIV5yY62TMyqQIU'
BASE_URL = 'https://api.themoviedb.org/3'
IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'

def requests_retry_session(retries=5, backoff_factor=1.0, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_poster_url(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    headers = {
        'Authorization': f'Bearer {TMDB_READ_ACCESS_TOKEN}',
        'Content-Type': 'application/json;charset=utf-8'
    }
    params = {
        'api_key': TMDB_API_KEY
    }
    try:
        response = requests_retry_session().get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"{IMAGE_BASE_URL}{poster_path}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for movie_id {movie_id}: {e}")
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        movie_name = request.form.get('movie_name')
        genre = request.form.get('genre')
        language = request.form.get('language')
        mood = request.form.get('mood')

        recommendations = df

        if movie_name:
            recommendations = recommendations[recommendations['title'].str.contains(movie_name, case=False, na=False)]
        if genre:
            recommendations = recommendations[recommendations['genre'].str.contains(genre, case=False, na=False)]
        if language:
            recommendations = recommendations[recommendations['original_language'].str.contains(language, case=False, na=False)]
        if mood:
            recommendations = recommendations[recommendations['mood'] == mood]

        # Limit recommendations to 5
        recommendations = recommendations.head(5)

        # Add poster URLs to the recommendations
        recommendations['poster_url'] = recommendations['id'].apply(get_poster_url)
        
        return render_template('recommend.html', movies=recommendations.to_dict(orient='records'))

    return render_template('recommend.html')

if __name__ == '__main__':
    app.run(debug=True)
