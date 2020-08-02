
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as opt




df=pd.read_csv('ex2data2.txt',header=None)
df=df.rename(columns={0:'Test_1',1:'Test_2',2:'Status'})
acc=df[df['Status']==1]
rej=df[df['Status']==0]
df.info()
print(df.head())
X = df.iloc[:,:-1]
y = df.iloc[:,2]
plt.figure()
plt.scatter(acc['Test_1'],acc['Test_2'],c='green',marker='+',label='Accepted')
plt.scatter(rej['Test_1'],rej['Test_2'],c='red',marker='_',label='Rejected')
plt.xlabel("Test_1")
plt.ylabel('Test_2')
plt.title("Microship Data")
plt.legend()
plt.show()

#one=np.ones((118,1))
#X=np.hstack((one,X))


def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j),                                     np.power(X2, j))[:,np.newaxis]))
    return out


X=mapFeature(X.iloc[:,0],X.iloc[:,1])



def Sigmoid(z):
    return 1/(1 + np.exp(-z));

def Gradient(theta, X, y,lambda_r):
    grad=((1/m) * X.T @ (Sigmoid(X @ theta) - y))
    grad[1:]=grad[1:]-(lambda_r)*theta[1:]/m
    return grad

def CostFunc(theta,X,y,lambda_r):
    h=Sigmoid(X.dot(theta))
    term = y * np.log(h) + (1 - y) * np.log(1-h) 
    re=(lambda_r*(np.sum(theta.T @ theta)))/2*m
    J = -(np.sum(term))/m+re
    return J

def normalisation(X):
    mean=np.mean(X)
    stdev=np.std(X)
    X_norm=(X-mean)/stdev
    return X_norm,mean,stdev

m , n = X.shape
y=y[:,np.newaxis]
theta = np.zeros((n,1))
lambda_r=0.17
cost=CostFunc(theta,X,y,lambda_r)
grad=Gradient(theta,X,y,lambda_r)
print(cost)
print(grad[0])
temp = opt.fmin_tnc(func = CostFunc, 
                    x0 = theta.flatten(),fprime = Gradient, 
                    args = (X, y.flatten(),lambda_r))
optimal_theta=temp[0]
print(optimal_theta)


pred = [Sigmoid(np.dot(X,optimal_theta)) >= 0.5]
np.mean(pred == y.flatten()) * 100

new_J=CostFunc(optimal_theta,X,y,lambda_r)
print(new_J)



a=np.linspace(-1,2.5,45)
b=np.linspace(-1,2.5,45)
c=np.zeros((len(a),len(b)))

def mapFeaturePlot(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
    return out



for i in range(len(a)):
    for j in range(len(b)):
        c[i,j] = np.dot(mapFeaturePlot(a[i], b[j]),optimal_theta)



plt.figure()
plt.scatter(acc['Test_1'],acc['Test_2'],c='green',marker='+',label='Accepted')
plt.scatter(rej['Test_1'],rej['Test_2'],c='red',marker='_',label='Rejected')
plt.xlabel("Test_1")
plt.ylabel('Test_2')
plt.contour(a,b,c,0)
plt.title("Decision Boundary for the dataset")
plt.legend()
plt.show()

        




