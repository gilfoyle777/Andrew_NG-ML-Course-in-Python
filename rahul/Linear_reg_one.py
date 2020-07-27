# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:40:48 2020

@author: rahul
"""
#Exercise-1 Andrew Ng Coursera


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('ex1data1.txt',delimiter=',')


df=pd.read_csv('ex1data1.txt',header=None)
df=df.rename(columns={0:'Population',1:'Profits'})
pop=df['Population']
pro=df['Profits']

 
print(df.describe())
print(df.info())



#Linear Regression with one Variable
def compute_cost(X,y,theta):
    pred=X.dot(theta)
    m=len(y)
    error=(pred-y)**2
    return (1/(2*m))*np.sum(error)
df.insert(2,"No",1)
df=df[["Profits",'No','Population']]
print(df.head())
m=len(pop)
y=df.values[:,0].reshape(m,1)
X=df.values[:,[1,2]].reshape(m,2)
theta=np.zeros((2,1))


print(compute_cost(X,y,theta))



#Gradient Descent
def grad(X,y,theta,alpha,iter):
    Com_history=[]
    for i in range(iter):
        m=len(pop)
        pred=X.dot(theta)
        err=np.dot(X.transpose(),(pred-y))
        error=(alpha*err)/m
        theta-=error
        Com_history.append(compute_cost(X,y,theta))
    return theta,Com_history

iter=1500
alpha=0.01

theta,Com_history=grad(X,y,theta,alpha,iter)
print(Com_history)   
print(theta[1])
print("The Hypothesis Equation is h(x)="+str(theta[0])+"+"+str(theta[1])+"*x1")
plt.plot(Com_history) 
plt.xlabel("Iterations")
plt.ylabel("Cost_function")
plt.title("Convergence")  

plt.figure(figsize=(10,6))
plt.scatter(pop,pro,c='#2ca02c',marker='1')
x_v=[a for a in range(25)]
y_v=[y*theta[1]+theta[0] for y in x_v]
plt.plot(x_v,y_v,c='red')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profits in 10,000$")
plt.title("Profits Vs Population")
plt.show()
    
#Predict
def pred(x,theta):
    prediction=np.dot(x,theta)
    return prediction

x1=[1 ,3.5]
x2=[1 ,7]
y1=pred(x1,theta)
y2=pred(x2,theta)

print(y1*10000)


theta0_val = np.linspace(-10,10,100)
theta1_val = np.linspace(-1,4,100)


J_val=np.zeros((theta0_val.shape[0],theta1_val.shape[0]))

for i,theta0 in enumerate(theta0_val):
    for j,theta1 in enumerate(theta1_val):
        J_val[i, j] = compute_cost(X, y, [theta0, theta1])
        
        
        
J_val=J_val.T

import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_val, theta1_val, J_val, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')
#ax=fig.gca(projection='3d')
#surf=ax.plot_surface(theta0_val,theta1_val,J_val,cmap='viridis')


ax = plt.subplot(122)
plt.contour(theta0_val, theta1_val, J_val, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
plt.title('Contour, showing minimum')
pass



