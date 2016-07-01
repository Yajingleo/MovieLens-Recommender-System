#This is a script of collaborative filtering algorithm for 
#movie recommendations of Andrew Ng. 

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import optimize


#Define the cost function.

def CostF(x, *arg):
    #The input x=[Flatten[X], Flatten[Theta]]
    #arg=Y, R, lam, n_m, n_u, n_f
    #n_m: number of movies
    #n_u: number of users
    #n_f: number of features
    
    Y, R, lam, n_m, n_u, n_f=arg

    X=x[0:n_m*n_f].reshape(n_m, n_f)
    Theta=x[n_m*n_f:].reshape(n_u, n_f)
    
    
    M=(X.dot(Theta.T)-Y)*R
    J=np.trace(M.dot(M.T))/2+lam*(np.trace(Theta.dot(Theta.T))+np.trace(X.dot(X.T)))/2
    return J

def GradF(x, *arg):
    Y, R, lam, n_m, n_u, n_f=arg

    X=x[0:n_m*n_f].reshape(n_m, n_f)

    Theta=x[n_m*n_f:].reshape(n_u, n_f)
    

    M=(X.dot(Theta.T)-Y)*R
    
    X_grad=M.dot(Theta)+lam*X
    Theta_grad=M.T.dot(X)+lam*Theta

    A=X_grad.reshape(X_grad.size)
    B=Theta_grad.reshape(Theta_grad.size)   
    return np.concatenate((A,B))

#Load data.
print "Loading the data."
data=io.loadmat('/Users/yajingleo/Downloads/Andrew_Ng_course/machine-learning-ex8/ex8/ex8_movies.mat')
print "Finished loading."

#The data Y is of shape n_movies * n_users.

Y=data['Y']
R=data['R']
lam=10
n_m, n_u=Y.shape
n_f=10

Y=Y*R
Y=Y.astype(np.float32)

Ymean=np.zeros(n_m)
for i in range(Y.shape[0]):
    Ymean[i]=np.mean(Y[i])
    Y[i]=Y[i]-Ymean[i]
    

x=np.random.rand(n_m*n_f+n_u*n_f)
args= (Y, R, lam, n_m, n_u, n_f)

#Learning the ratings.
x=optimize.fmin_cg(CostF, x, fprime=GradF, args=args, maxiter=100)


X=x[0:n_m*n_f].reshape(n_m, n_f)
Theta=x[n_m*n_f:].reshape(n_u, n_f)

