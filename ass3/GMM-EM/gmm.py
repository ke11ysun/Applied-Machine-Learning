# -*- coding: utf-8 -*-
"""
Created on Sat May  3 10:21:21 2014

@author: umb
"""

import numpy as np
    
class GMM:
    
    def __init__(self, k = 3, eps = 0.0001):
        self.k = k ## number of clusters
        self.eps = eps ## threshold to stop `epsilon`

    


    def __append_mu(self, mus, mu, d):
        for i in range(self.k):
            for j in range(d):
                mus[i][j].append(mu[i][j])



    def fit_EM(self, X, max_iters = 1000, mu=None, Sigma=None, init_w_kmeans=False):
        
        # n = number of data-points, d = dimension of data points        
        n, d = X.shape

        mus = [[[] for _ in range(d)] for __ in range(self.k)]

        if not init_w_kmeans:     
	        mu = X[np.random.choice(n, self.k, False), :]
        	Sigma= [np.eye(d)] * self.k

        print "mu.shape", mu.shape
        self.__append_mu(mus, mu, d)


        
        # initialize the probabilities/weights for each gaussians
        w = [1./self.k] * self.k
        
        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))
        
        ### log_likelihoods
        log_likelihoods = []
        
        P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(s) , (X - mu).T).T ) ) 
                        
        # Iterate till max_iters iterations        
        while len(log_likelihoods) < max_iters:
            
            # E - Step
            
            ## Vectorized implementation of e-step equation to calculate the 
            ## membership for each of k -gaussians
            for k in range(self.k):
                R[:, k] = w[k] * P(mu[k], Sigma[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            
            log_likelihoods.append(log_likelihood)
            
            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis = 1)).T
            
            ## The number of datapoints belonging to each gaussian            
            N_ks = np.sum(R, axis = 0)
            
            
            # M Step
            ## calculate the new mean and covariance for each gaussian by 
            ## utilizing the new responsibilities
            for k in range(self.k):
                
                ## means
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
                x_mu = np.matrix(X - mu[k])		# x_mu: yi-mu
                
                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                
                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]

            self.__append_mu(mus, mu, d)



            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break
        

        return mus, len(log_likelihoods)
    

        
        
# def demo_2d():
    
#     # Load data
#     #X = np.genfromtxt('data1.csv', delimiter=',')
#     ### generate the random data     
#     np.random.seed(3)
#     m1, cov1 = [9, 8], [[.5, 1], [.25, 1]] ## first gaussian
#     data1 = np.random.multivariate_normal(m1, cov1, 90)
    
#     m2, cov2 = [6, 13], [[.5, -.5], [-.5, .1]] ## second gaussian
#     data2 = np.random.multivariate_normal(m2, cov2, 45)
    
#     m3, cov3 = [4, 7], [[0.25, 0.5], [-0.1, 0.5]] ## third gaussian
#     data3 = np.random.multivariate_normal(m3, cov3, 65)
#     X = np.vstack((data1,np.vstack((data2,data3))))
#     np.random.shuffle(X)
# #    np.savetxt('sample.csv', X, fmt = "%.4f",  delimiter = ",")
#     ####
#     gmm = GMM(3, 0.000001)
#     # params = gmm.fit_EM(X, max_iters= 100)
#     # print params.log_likelihoods
#     mus, nums_iter = gmm.fit_EM(X, max_iters= 100)
#     print mus, nums_iter


       

# if __name__ == "__main__":

#     # demo_2d() 







import pandas as pd
import numpy as np

with open('/Users/sjx/Desktop/CS5785/ass3/faithful.csv', 'r') as f:
    lines = f.readlines()
    
columns = lines[0].split("\n")[0].split(" ")[-2:]
print columns

samples = []
for line in lines:
    samples.append(np.asarray([data for data in line.split("\n")[0].split(" ") if data != ""])[1:])
samples = np.asarray(samples[1:])

train_df = pd.DataFrame(samples, columns=columns).astype('float')
# print train_df

X = train_df.values
print X.shape







####
gmm = GMM(2, 0.000001)
# params = gmm.fit_EM(X, max_iters= 100)
# print params.log_likelihoods
mus, nums_iter = gmm.fit_EM(X, max_iters= 500)
print mus, nums_iter   

