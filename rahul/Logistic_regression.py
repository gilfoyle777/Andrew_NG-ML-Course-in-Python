
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd




df=pd.read_csv('eX2data1.tXt',header=None)
df=df.rename(columns={0:'Score_1',1:'Score_2',2:'Status'})
adm=df[df['Status']==1]
rej=df[df['Status']==0]
#df.info()
print(df.head())
data=df.values
X=data[:,:2]
y=data[:,-1]


one=np.ones((100,1))
X=np.hstack((one,X))
y=y.reshape(100,1)
def Sigmoid(z):
    return 1/(1 + np.exp(-z));

def Gradient(theta, X, y):
    return ((1/m) * X.T @ (Sigmoid(X @ theta) - y))

def CostFunc(theta,X,y):
    h=Sigmoid(X.dot(theta))
    term = y * np.log(h) + (1 - y) * np.log(1-h) 
    J = -((np.sum(term))/m)
    return J

def normalisation(X):
    mean=np.mean(X)
    stdev=np.std(X)
    X_norm=(X-mean)/stdev
    return X_norm,mean,stdev

m , n = X.shape;
theta = np.zeros((n,1));
temp = opt.fmin_tnc(func = CostFunc, 
                    x0 = theta.flatten(),fprime = Gradient, 
                    args = (X, y.flatten()))
optimal_theta=temp[0]
print(optimal_theta)
J=CostFunc(theta,X,y)
grad=Gradient(theta,X,y)

print(J)
print(grad)



x=[1,45,85]    
a=Sigmoid(np.dot(optimal_theta,x))
print(a)


new_J=CostFunc(optimal_theta,X,y)
print(new_J)
x_val=[np.min(X[:,1]-2), np.max(X[:,2]+2)]
y_val= -1/optimal_theta[2]*(optimal_theta[0] 
          + np.dot(optimal_theta[1],x_val)) 
plt.figure()
plt.scatter(adm['Score_1'],adm['Score_2'],c='green',marker='+',label='Admitted')
plt.scatter(rej['Score_1'],rej['Score_2'],c='red',marker='_',label='Rejected')
plt.plot(x_val,y_val)
l2=np.array((50,60))
trans_angle = plt.gca().transData.transform_angles(np.array((315,)),
                                                   l2.reshape((1, 2)))[0]
plt.text(50,60,"Decision Boundary",fontsize=16,
               rotation=trans_angle, rotation_mode='anchor')
plt.xlabel("Score_1")
plt.ylabel('Score_2')
plt.title("Decision Boundary for the dataset")
plt.legend()
plt.show()






