from compute_scores import compare_users
import json
import numpy as np
import requests

"""
Dominik Hinc (s22436) & Sylwia Juda (s25373)

Movie recommendation system

The system recommends movies to a user based on the similarity and anti-similarity metric.

The system uses two types of similarity metrics:
1. Euclidean distance
2. Pearson correlation

The system uses a dataset containing the ratings of users for movies.

The system follows these steps to recommend movies:
1. Calculate the similarity score between the user and all other users
2. Sort the users based on the similarity score
3. For each user, recommend movies that they rated highly (>=7) and the user has not seen yet, until the number of recommendations is reached
4. For each user, recommend movies that they rated poorly (<5) and the user has not seen yet, until the number of anti-recommendations is reached
5. Return the list of recommended and anti-recommended movies
6. Fetch the movie details from the OMDB API
7. Print the movie details

The dataset is stored in a JSON file, where each user has rated a set of movies.

Setup:

Install dependencies:

pip install -r requirements.txt


Run the script:

python main.py

"""



'''
    Load the data from a JSON file

    :param ratings_file: The file containing the ratings data

    :return: The normalized data
'''
def load_data(ratings_file):
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())
    normalized_data = {}
    for user, movies in data.items():
        normalized_movies = {movie.lower().strip(): rating for movie, rating in movies.items()}
        normalized_data[user] = normalized_movies
    return normalized_data


'''
    Recommend movies to a user based on the similarity and anti-similarity metric, by following these steps:
    1. Calculate the similarity score between the user and all other users
    2. Sort the users based on the similarity score
    3. For each user, recommend movies that they rated highly (>=7) and the user has not seen yet, until the number of recommendations is reached
    5. For each user, recommend movies that they rated poorly (<5) and the user has not seen yet, until the number of anti-recommendations is reached
    6. Return the list of recommended and anti-recommended movies
    
    :param user: The user to recommend movies to
    :param data: The dataset
    :param metric: The similarity metric to use
    :param n_recommendations: The number of recommendations to make

    :return: The list of recommended movies
'''
def recommend_movies(user, data, metric, n_recommendations=5):
    scores = {}
    # Calculate the similarity score between the user and all other users
    for other_user in data:
        if other_user != user:
            score = compare_users(user, other_user, metric, data)
            scores[other_user] = score

    # Sort the users based on the similarity score
    sorted_users = sorted(scores, key=scores.get, reverse=True)
    recommendations = []
    anti_recommendations = []

    user_movies = set(data[user].keys())
    
    # For each user, recommend movies that they rated highly (>=7) and the user has not seen yet, until the number of recommendations is reached
    for other_user in sorted_users:
        other_user_movies = set(data[other_user].keys())
        other_user_movies_different_from_user_movies = other_user_movies - user_movies
        # Sort the movies based on the rating
        other_user_movies_different_from_user_movies = sorted(
            other_user_movies_different_from_user_movies, 
            key=lambda movie: data[other_user][movie], 
            reverse=True
        )

        for movie in other_user_movies_different_from_user_movies:
            if len(recommendations) < n_recommendations and data[other_user][movie] >= 7:
                recommendations.append(movie)

        if len(recommendations) >= n_recommendations:
            break

    # For each user, recommend movies that they rated poorly (<5) and the user has not seen yet, until the number of anti-recommendations is reached
    for other_user in sorted_users:
        other_user_movies = set(data[other_user].keys())
        other_user_movies_different_from_user_movies = other_user_movies - user_movies
        # Sort the movies based on the rating
        other_user_movies_different_from_user_movies = sorted(
            other_user_movies_different_from_user_movies, 
            key=lambda movie: data[other_user][movie], 
            reverse=False
        )

        for movie in other_user_movies_different_from_user_movies:
            if len(anti_recommendations) < n_recommendations and data[other_user][movie] < 5:
                anti_recommendations.append(movie)

        if len(anti_recommendations) >= n_recommendations:
            break

    return recommendations, anti_recommendations

'''
    Fetch the movie details from the OMDB API

    :param movie_title: The title of the movie to fetch details for

    :return: The movie details
'''
def fetch_movie_details(movie_title):
    api_key = '17dd6806'
    url = f'http://www.omdbapi.com/?t={movie_title}&apikey={api_key}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            movie_data = response.json()
            if movie_data['Response'] == 'True':
                return {
                    "title": movie_data.get('Title', 'Not found in database'),
                    "year": movie_data.get('Year', 'Not found in database'),
                    "genre": movie_data.get('Genre', 'Not found in database'),
                    "rating": movie_data.get('imdbRating', 'Not found in database')
                }
            else:
                print(f"Error fetching movie details for {movie_title}")
                raise Exception(response)
        else:
            print(f"Error fetching movie details for {movie_title}")
            raise Exception(response)
    except Exception as e:
        return {
            "title": movie_title,
            "year": "Not found in database",
            "genre": "Not found in database",
            "rating": "Not found in database"
        }

'''
    Print the movie details

    :param movies: The list of movies to print details for
'''
def print_movie_details(movies):
    for movie in movies:
        details = fetch_movie_details(movie)
        print(f"Title: {details['title']}, Year: {details['year']}, Genre: {details['genre']}, Rating: {details['rating']}")

'''
    Main function
'''
def main():
    try:
        data = load_data('ratings.json')
        print('Data loaded successfully')
    except Exception as e:
        print('Error loading data')
        return

    user = None
    while user not in data:
        print("Input a user to get recommendations for: ")
        user = input()
        if(user not in data):
            print('User not found in the dataset. Please try again.')
    recommendations, anti_recommendations = recommend_movies(user, data, 'Euclidean')
    print('Recommended movies (Euclidean):')
    print_movie_details(recommendations)
    print('\nMovies to avoid (Euclidean):')
    print_movie_details(anti_recommendations)

    recommendations, anti_recommendations = recommend_movies(user, data, 'Pearson')
    print('\nRecommended movies (Pearson):')
    print_movie_details(recommendations)
    print('\nMovies to avoid (Pearson):')
    print_movie_details(anti_recommendations)

main()
