# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np
import matplotlib.pyplot as plt

def predict(X,W,b):
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """

    return sigmoid(np.matmul(X,W)+b)


def sigmoid(a):
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """

    return 1/(1 + np.exp(-a))


def l2loss(X,y,W,b):
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """

    """
    The size of variables
    # X: [N,D]
    # Y: [N,1]
    # W: [D,1]
    # b: [1]
    
    np.matmul(X,W) = XW = [N x D][D x 1] = [N x 1] 
    sigmoid(XW + b) = [N x 1]
    """

    # function which calculates and returns the L2_Loss
    def loss_cal(X,y,W,b):
        # print(np.shape(np.square(y - predict(X,W,b))))
        # print(np.square(y - predict(X,W,b)))

        return np.sum(np.square(y - predict(X,W,b)))


    def get_part_dev_num_dif(X,y,W,b):
        """
        retrieve the calculated partial derivatives of given loss function by the numerical differentiation method with respect to parameters W and b

        :param X: input training data
        :param y: target value, label
        :param W: weights, D X 1 array
        :param b: bias, scalar value

        :return: partial derivatives with respect to W and b
        """

        # calculating partial derivatives with respect to weight w and bias b
        epsilon = 1e-6
        grad_b = (loss_cal(X,y,W,b+epsilon) - loss_cal(X,y,W,b))/epsilon


        # in order to get the partial derivatives for each weight in W
        grad_w_list = [] # size of [N x 1] list to store the calculated gradients of weights w(0), w(2), ..., w(N-1)
        for i in range(np.shape(W)[0]): # np.shape(W)[0] = D : number of weights
            W_epsilon = [0]*np.shape(W)[0]
            W_epsilon[i] = epsilon # set i-th item in list as the value of epsilon to calculate the respective gradient of i-th weight
            grad_w = (loss_cal(X,y,W+ W_epsilon,b) - loss_cal(X,y,W,b))/epsilon # calculate the gradient of i-th weight
            grad_w_list.append(grad_w) # add the calculated gradient of i-th weight to the list where to save all respective gradients with respect to w(0 ~ N-1)


        return grad_w_list, grad_b

    def get_part_dev_chain_rule(X,y,W,b):
        """
        retrieve the calculated partial derivatives of given loss function by the chain rule with respect to parameters W and b

        :param X: input training data [N, D]
        :param y: target value, label
        :param W: weights, [D, 1] array
        :param b: bias, scalar value

        :return: partial derivatives with respect to W and b
        """


        grad_w_list = [] # store all partial derivatives with respect to weights w[0 ~ D-1]

        X_t = np.transpose(X) # transposed matrix of input training data X [D, N]
        chain_rule_intermediate = np.array((y-predict(X,W,b))*(predict(X, W, b))*(1-predict(X, W, b))) # store the intermediate result of chain rule and make it as numpy array
        chain_rule_intermediate= np.reshape(chain_rule_intermediate,(-1,1)) # reshape the variable for the convenience of matrix multiplication, shpae: [N, 1]

        # calculate partial derivatives for each weight w[0 ~ D-1] and store in grad_w_list
        for i in range(np.shape(W)[0]): # np.shape(W)[0] =  D : number of weights

            #array of values at the index i in every row of input data which need to be multiplied to the chain_rule_intermediate for calculating the partial derivative with respect to the i-th weight:w_i
            X_i = np.reshape(X_t[i], (1, -1)) # shape: [1, N]


            # calculate the partial derivative of w[i] by the chain rule, scalar value,
            grad_wi = np.asscalar(-2*np.matmul(X_i,chain_rule_intermediate)) #shape: X_i[1, N] * chain_rule_intermediate[N, 1] = scalar value[1, 1]
            grad_w_list.append(grad_wi) # list of scalar values

        grad_b = -2*np.sum((y-predict(X,W,b))*(predict(X, W, b))*(1-predict(X, W, b))) # calculate the partial derivative of the bias by the chain rule

        return grad_w_list, grad_b

    l2_loss = loss_cal(X,y,W,b)
    grad_w_list, grad_b = get_part_dev_chain_rule(X,y,W,b)

    return l2_loss, grad_w_list, grad_b



def train(X,y,W,b, num_iters=1000, eta=0.001):
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
     """
    loss_list=[]
    for i in range(1, num_iters+1):

        l2_loss, grad_w_list, grad_b = l2loss(X,y,W,b)

        # update weight and bias with respect to gradients(partial derivatives) of those
        W -= (np.multiply(eta,grad_w_list))
        b -= (np.multiply(eta,grad_b))

        if i%50 == 0:
            print('Loss in step {} = {}'.format(i, l2_loss))

        loss_list.append(l2_loss)

    # plot and save the figure
    plt.plot(loss_list)
    plt.xlabel("Iteration step")
    plt.ylabel("Loss")
    plt.savefig('linear_classifer_figure.png')
    plt.show()

    return W, b


