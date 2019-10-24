# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats
 

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """  
    dists = -2 * np.dot(Y, X.transpose()) + np.sum(X**2,axis=1) + np.sum(Y**2,axis=1)[:, np.newaxis]
    return dists
    #return euclidean_distances(Y,X)
 

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    #minInRows = np.argmin(dists, axis=0)
    minInRows = np.argpartition(dists, k, axis=1)
    #result = np.where(dists == minInRows)
    #index = np.array([])
    #print(minInRows[:k])

    # for r in range(len(minInRows)):
    #  print(labels[minInRows[r][:k]])

    # for r in range(len(minInRows)):
    #  m = stats.mode(labels[minInRows[r][:k]])
    #  print(m[0])

    dim=np.shape(dists)
    #print(dim[0])

    pred_labels = np.array([])

    for r in range(len(minInRows)):
        m = stats.mode(labels[minInRows[r][:k]])
        pred_labels = np.append(pred_labels, m[0])   
    #pred_labels = np.append(pred_labels, labels[minInRows])

    #print(np.shape(pred_labels))

    return pred_labels

     