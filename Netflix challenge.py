#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import os
from scipy import sparse
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# -- SECTION: Loading the dataset --

# creating temporal files
fileTempMovie = open('moviesTemporal.txt', 'w')
fileTempRating = open('ratingsTemporal.txt', 'w')
fileTempUser = open('usersTemporal.txt', 'w')

# function read lines
def readLines(name_file):
    movieId = '*'
    file = open(name_file, 'r')
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line[-1:] == ':':
            movieId = line[:-1] + "\n"
            continue
        # writing to each file: user data, rating data, movie data
        arrayLine = line.strip().split(',')
        fileTempMovie.writelines(movieId)
        fileTempRating.writelines(arrayLine[1] + "\n")
        fileTempUser.writelines(arrayLine[0] + "\n")
    file.close()
    return

# Reading 1st file: combined_data_1.txt
readLines('combined_data_1.txt')
# Reading 2nd file: combined_data_2.txt
readLines('combined_data_2.txt')
# Reading 3rd file: combined_data_3.txt
readLines('combined_data_3.txt')
# Reading 4rd file: combined_data_4.txt
readLines('combined_data_4.txt')

# Closing files
fileTempMovie.close()    
fileTempRating.close()    
fileTempUser.close() 

# loading data
dfMovies = np.loadtxt('moviesTemporal.txt', dtype= int)
dfRatings = np.loadtxt('ratingsTemporal.txt', dtype= int)
dfUsers = np.loadtxt('usersTemporal.txt', dtype= int)

# Creating and saving sparse matrix into a file
sparse_matrix = sparse.csr_matrix((dfRatings, (dfMovies, dfUsers)), shape=(np.max(dfMovies) + 1, np.max(dfUsers) + 1))
sparse.save_npz('sparse_matrix_movies_row.npz', sparse_matrix)

sparse_matrix2 = sparse.csr_matrix((dfRatings, (dfUsers, dfMovies)), shape=(np.max(dfUsers) + 1, np.max(dfMovies) + 1))
sparse.save_npz('sparse_matrix_users_row.npz', sparse_matrix2)

# Removing temporal files
os.remove("moviesTemporal.txt")
os.remove("ratingsTemporal.txt")
os.remove("usersTemporal.txt")

# -- SECTION: (Stochastic) Gradient Descent with Latent Factors --

# for the iteration all known ratings of a matrix
def knownRatings(matrix):
    return zip(matrix.nonzero()[0], matrix.nonzero()[1])

# RMSE
def rmse(values, predictedValues):
    # Root Mean Squared Error
    mse = np.square(np.subtract(values,predictedValues)).mean()
    return np.sqrt(mse)

# BGD
def batch_gradient_descent(epochs, lrate, matrix):  
    errorRmse = []
    
    # Randomly initialize P and Q
    Q = np.random.normal(size=(np.max(matrix.nonzero()[0]) + 1, 2), scale = 1/10) # Movies factors
    P = np.random.normal(size=(np.max(matrix.nonzero()[1]) + 1, 2), scale = 1/10) # Users factors
    
    # for a given number of times
    for epoch in range(epochs):
        values = []
        predictedValues = []
        # for all known ratings
        for u, i in knownRatings(matrix):
            # compute error: Rui - Rui(Predicted)
            error = matrix[u,i] - np.dot(Q[u, :], P[i,:].T)
            userValue = np.copy(Q[u, :])
            # update values
            Q[u, :]  += lrate * (error * P[i,:] - K * Q[u, :]) # movieValue += lrate * (err * userValue[user] - K * movieValue[movie])
            P[i,:] += lrate * (error * userValue - K * P[i,:]) # userValue += lrate * (err * movieValue[movie] - K * userValue[user])
            values.append(matrix[u,i])
            predictedValues.append(np.dot(Q[u, :], P[i,:].T))
        errorRmse.append(rmse(values, predictedValues))
    return errorRmse

# SGD
def stochastic_gradient_descent(epochs, lrate, matrix, indexes, lenIndexes, size):  
    sample_size = int(matrix.size * size)
    errorRmse = []
    print(sample_size)
    # initialize P and Q
    Q = np.random.normal(size=(np.max(matrix.nonzero()[0]) + 1, 2), scale = 1/10) # Movies factors
    P = np.random.normal(size=(np.max(matrix.nonzero()[1]) + 1, 2), scale = 1/10) # Users factors
    # for a given number of times
    for epoch in range(epochs):
        values = []
        predictedValues = []
        np.random.shuffle(indexes)
        # for all known ratings based on the sample
        for u, i in indexes[:sample_size]:
            # compute error: Rui - Rui(Predicted)
            error = matrix[u,i] - np.dot(Q[u, :], P[i,:].T)
            userValue = Q[u, :]
            # update values
            Q[u, :]  += lrate * (error * P[i,:] - K * Q[u, :]) # userValue += lrate * (err * movieValue[movie] - K * userValue[user])
            P[i,:] += lrate * (error * userValue - K * P[i,:]) # movieValue += lrate * (err * userValue[user] - K * movieValue[movie])
            values.append(matrix[u,i])
            predictedValues.append(np.dot(Q[u, :], P[i,:].T))
        errorRmse.append(rmse(values, predictedValues))
    return errorRmse, Q, P

#load sparse matrix Movies(rows), user(columns)
M = sparse.load_npz('sparse_matrix_movies_row.npz')
# M = sparse.load_npz('sparse_matrix_movies_row_50000.npz') *Matrix used for experiments
M = M.asfptype()
# variables
epochs = 100 #number of iterations
K = 0.02 #regularization

# Shuffle indexes for sampling in SGD
indexes_matrix = knownRatings(matrix):
indexes_matrix = list(tuple(indexes_matrix))
np.random.shuffle(indexes_matrix)

rmseEpochBGD = batch_gradient_descent(epochs = epochs, lrate = 0.1, matrix = M)
rmseEpochSGD, Q_predicted, P_predicted = stochastic_gradient_descent(epochs = 200, lrate = 0.01, matrix = M, indexes = indexes_matrix, lenIndexes = len(indexes_matrix), size = 0.1)

# plotting
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.title('RMSE per Epochs')
plt.plot(rmseEpochBGD)
plt.plot(rmseEpochSGD)
plt.legend(["BGD","SGD"])

# -- SECTION: Accuracy --

# Buil prediction
def prediction(Q,P):
    return np.dot(Q,P.T)

# using probe.txt to remove testing data on training set

testingIndices = np.loadtxt('Temporal3.txt',delimiter=',', dtype = int)

# using probe.txt to remove testing data on training set
def clean_train_test(matrix, testingIndices):
    newTest = []
    for u, i in testingIndices:
        try:
            matrix[u,i] = 0
            newTest.append([u,i])
        except IndexError:
            continue
    return matrix, newTest

# removing data from the test set in the training set
M, newTest = clean_train_test(M, testingIndices)  # M: contains the training set, newTest: contains the testing set indices
sparse.save_npz('sparse_matrix_training_row.npz', M) # saved for more experiments

# copying the matrix for training set and testing set
fullMatrix = M

# --TRAINING ERROR --

testRmse = []
predicted_values = []
matrix_values = []
for u, i in newTest:
    valueTemp = prediction(Q_predicted[u, :], P_predicted[i, :])
    testRmse.append(rmse(fullMatrix[u,i], valueTemp))
    predicted_values.append(valueTemp)
    matrix_values.append(fullMatrix[u,1])
    
# predictions for the training data(calculating P and Q, including test of ratings) vs a test of ratings
print("RMSE: ", rmse(matrix_values, predicted_values))
# first 20 values
j = 0
for u, i in newTest:
    j = j + 1
    print('# ', j ,' value: ' , fullMatrix[u,i] , ', predicted: ' , prediction(Q_predicted[u, :], P_predicted[i, :]) , 'RMSE: ' , rmse(fullMatrix[u,i], prediction(Q_predicted[u, :], P_predicted[i, :])))
    if j == 20:
        break

# plotting
plt.boxplot(testRmse)
plt.xlabel('RMSE errors')
plt.title('Predicted ratings evaluation')

# -- TESTING ERROR --

# Compute P, Q using the training set with SGD
rmseEpochSGD2, Q_predicted2, P_predicted2 = stochastic_gradient_descent(epochs = 200, lrate = 0.01, matrix = M, indexes = indexes_matrix, lenIndexes = len(indexes_matrix), size = 0.1)

# predictions: Use of the testing set (newTest) to predict how good the ratings are
testRmse = []
predicted_values = []
matrix_values = []
for u, i in newTest:
    valueTemp = prediction(Q_predicted2[u, :], P_predicted2[i, :])
    testRmse.append(rmse(fullMatrix[u,i], valueTemp))
    predicted_values.append(valueTemp)
    matrix_values.append(fullMatrix[u,1])
    
# predictions for the test set vs the training data
print("RMSE: ", rmse(matrix_values, predicted_values))
# first 20 values
j = 0
for u, i in newTest:
    j = j + 1
    print('# ', j ,' value: ' , fullMatrix[u,i] , ', predicted: ' , prediction(Q_predicted2[u, :], P_predicted2[i, :]) , 'RMSE: ' , rmse(fullMatrix[u,i], prediction(Q_predicted2[u, :], P_predicted2[i, :])))
    if j == 20:
        break

# plotting
plt.boxplot(testRmse)
plt.xlabel('RMSE errors')
plt.title('Predicted ratings evaluation')

