import numpy as np

''' 
    Compute the Euclidean distance score between user1 and user2 

    :param dataset: The dataset containing the ratings
    :param user1: The first user
    :param user2: The second user

    :return: The Euclidean distance score between user1 and user2
'''
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {} 

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users, 
    # then the score is 0 
    if len(common_movies) == 0:
        return 0

    squared_diff = [] 

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))
        
    return 1 / (1 + np.sqrt(np.sum(squared_diff))) 

''' 
    Compute the Pearson correlation score between user1 and user2 

    :param dataset: The dataset containing the ratings
    :param user1: The first user
    :param user2: The second user

    :return: The Pearson correlation score between user1 and user2
'''
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies) 

    # If there are no common movies between user1 and user2, then the score is 0 
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies 
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies 
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings
    
    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)

'''
    Compare two users based on the similarity metric

    :param user1: The first user
    :param user2: The second user
    :param score_type: The similarity metric to use
    :param data: The dataset

    :return: The similarity score between the two users
'''
def compare_users(user1, user2, score_type, data):
    if score_type == 'Euclidean':
        return euclidean_score(data, user1, user2)
    else: 
        return pearson_score(data, user1, user2)

