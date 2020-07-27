# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:55:49 2020

@author: rahul
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df=pd.read_csv('ex1data2.txt',header=None)
df.info()
print(df.head())


#df[0]=(df[0]-df[0].mean())/df[0].std()
#df[1]=(df[1]-df[1].mean())/df[1].std()
#df[2]=(df[2]-df[2].mean())/df[2].std()



def compute_cost(X,y,theta):
    pred=X.dot(theta)
    m=len(y)
    error=(pred-y)**2
    return (1/(2*m))*np.sum(error)


#Linear_regression with multiple variables
m=len(df[2])
data=df.values
X=data[:,[0,1]].reshape(m,2)
y=data[:,2].reshape(m,1)
ones=np.ones((m,1))
X=np.hstack((ones,X))
X=np.log(X)
print(y)
print(X)
theta=np.zeros((3,1))
print(compute_cost(X,y,theta))

def grad(X,y,theta,alpha,iters):
    Com_history=[]
    for i in range(iters):
        m=len(y)
        pred=X.dot(theta)
        err=np.dot(X.transpose(),(pred-y))
        error=(alpha*err)/m
        theta-=error
        Com_history.append(compute_cost(X,y,theta))
    return theta,Com_history
iters=400
alpha=0.01
theta_new,Com_history=grad(X,y,theta,alpha,iters)
#print(Com_history)
print(theta_new)


plt.plot(Com_history) 
plt.xlabel("Iterations")
plt.ylabel("Cost_function")
plt.title("Convergence") 



def pred(x,theta):
    prediction=np.dot(x,theta)
    return prediction
def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    
    X_norm = (X - mean)/std
    
    return X_norm
x=[1,1650,3]
print(pred(np.log(x),theta_new))

from numpy.linalg import inv
#Implementation of normal equation to find analytic solution to linear regression
def normEqtn(X,y):
    #restheta = np.zeros((X.shape[1],1))
    return np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)


print("Normal equation prediction for price of house with 1650 square feet and 3 bedrooms")
print("$%0.2f" % float(normEqtn(X,y),[1,1650.,3]))