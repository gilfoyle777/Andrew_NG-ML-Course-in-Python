# Import the libraries

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd


#Load the training data,here we use loadmat() func from scipy to convert mat files into array

data=loadmat('ex3data1.mat')
X=data['X']
y=data['y']

print(X.shape)
print(y.shape)

# Loading and Visualizing Data
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.


# Randomly select 100 data points to display
import matplotlib.image as mpimg
fig,axis=plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order='F'),cmap='magma')
        axis[i,j].axis('off')
        
        
m,n=X.shape


def Sigmoid(z):
    return 1/(1 + np.exp(-z))


def lrCostFunc(theta,X,y,lambda_r):
    m=len(y)
    h=Sigmoid(X @ theta)
    term = (-y * np.log(h)) - ((1 - y) * np.log(1-h))
    cost=(1/m)*sum(term)
    regcost=cost+(1/(2*m))*lambda_r*sum(theta[1:]**2)
    j_0= 1/m * (X.transpose() @ (h - y))[0]
    j_1 = 1/m * (X.transpose() @ (h - y))[1:] + (lambda_r/m)* theta[1:]
    grad= np.vstack((j_0[:,np.newaxis],j_1))
    return regcost[0],grad

#Test case to check the cost function
    
theta_t = np.array([-2,-1, 1, 2]).reshape(4,1)
X_t = np.array([np.linspace(0.1,1.5,15)]).reshape(3,5).T
X_t= np.hstack((np.ones((5,1)),X_t))
y_t = np.array([1,0,1,0,1]).reshape(5,1)
lambda_t = 3
J,grad= lrCostFunc(theta_t, X_t, y_t, lambda_t)

print('\nCost: %f\n', J)
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(' %f \n', grad)
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')




def gra_des(theta,X,y,lambda_r,alpha,iters):
    Com_history=[]
    for i in range(iters):
        J,grad=lrCostFunc(theta,X,y,lambda_r)
        theta= theta - (alpha*grad)
        Com_history.append(J)
    return theta,Com_history



def OnevsAll(X,y,num_labels,lambda_r):
    m,n=X.shape[0],X.shape[1]
    init_theta=np.zeros(((n+1),1))
    all_theta=[]
    all_cost=[]
    X=np.hstack((np.ones((m,1)),X))
    for i in range(1,num_labels+1):
        theta,his=gra_des(init_theta,X,np.where(y==i,1,0),lambda_r,1,300)
        all_theta.extend(theta)
        all_cost.extend(his)
    return np.array(all_theta).reshape(num_labels,n+1),all_cost

num_labels=10
lambda_r=0.1
all_theta,all_cost=OnevsAll(X,y,num_labels,lambda_r)

plt.figure()
plt.plot(all_cost[0:300])
plt.xlabel("Num_iter")
plt.ylabel("Cost Func")
plt.title("Cost Function for 300 iterations")
plt.show()

def predOnevsAll(all_theta,X):
    m=X.shape[0]
    X=np.hstack((np.ones((m,1)),X))
    pred=X @ all_theta.T
    return np.argmax(pred,axis=1)+1



pred=predOnevsAll(all_theta,X)
print('\nTraining Set Accuracy with Logistic regression using One vs All: \n',sum(pred[:,np.newaxis]==y)[0]/50,'%')

weights=loadmat('ex3weights.mat')
Theta_1=weights['Theta1']
Theta_2=weights['Theta2']

    

def predict(theta1,theta2,X):
    m=X.shape[0]
    X=np.hstack((np.ones((m,1)),X))
    a1=Sigmoid(X @ theta1.T)
    a1=np.hstack((np.ones((m,1)),a1))
    a2=Sigmoid(a1 @ theta2.T)
    
    return np.argmax(a2,axis=1)+1

pred=predict(Theta_1,Theta_2,X)
print('\nTraining Set Accuracy with Forward Propogarion is : \n',sum(pred[:,np.newaxis]==y)[0]/50,'%')

    