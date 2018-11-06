# -*- coding: utf-8 -*-
"""
Created on Sat May  3 10:21:21 2014

@author: umb
"""

class GMM:
    
    def __init__(self, k = 3, eps = 0.0001):
        self.k = k 
        self.eps = eps 



    def __append_mu(self, mus, mu, d):
        for i in range(self.k):
            for j in range(d):
                mus[i][j].append(mu[i][j])



    def fit_EM(self, X, max_iters = 1000, mu=None, Sigma=None, init_w_kmeans=False):       
        n, d = X.shape
        mus = [[[] for _ in range(d)] for __ in range(self.k)]

        if not init_w_kmeans:     
	        mu = X[np.random.choice(n, self.k, False), :]
        	Sigma= [np.eye(d)] * self.k

        # print "mu.shape", mu.shape
        self.__append_mu(mus, mu, d)

        
        # probabilities/weights for each gaussians
        # responsibility for each of n points for eack of k gaussians
        w = [1./self.k] * self.k
        R = np.zeros((n, self.k))
        log_likelihoods = []
        
        P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(s) , (X - mu).T).T ) ) 
                               
        while len(log_likelihoods) < max_iters:
            
            # E - Step
            for k in range(self.k):
                R[:, k] = w[k] * P(mu[k], Sigma[k])

            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))           
            log_likelihoods.append(log_likelihood)
            
            R = (R.T / np.sum(R, axis = 1)).T
            
            ## The number of datapoints belonging to each gaussian            
            N_ks = np.sum(R, axis = 0)
            
            
            # M Step
            for k in range(self.k):
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
                x_mu = np.matrix(X - mu[k])		# x_mu: yi-mu
                
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                
                w[k] = 1. / n * N_ks[k]

            self.__append_mu(mus, mu, d)
            
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break
        

        return mus, len(log_likelihoods)
    

        
        

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
# gmm = GMM(2, 0.000001)
# mus, nums_iter = gmm.fit_EM(X, max_iters= 100)
# print mus, nums_iter   

# conv_iters = []
# for _ in range(50):
#     mus, iters = gmm.fit_EM(X, max_iters= 100)
#     conv_iters.append(iters)
# print(conv_iters)
# plt.hist(conv_iters, bins=range(max(conv_iters) + 2), align='left', rwidth=1)
# plt.show()

def run_kmeans(k, r, data):
    kmeans = KMeans(n_clusters=k, random_state=r).fit(data)
    pred = kmeans.predict(data)

    c1 = train_df[pred==0].values
    c2 = train_df[pred==1].values

    kmeans_mu = kmeans.cluster_centers_
    kmeans_sigma = [np.cov(c1, rowvar=False), np.cov(c2, rowvar=False)]
    return kmeans_mu, kmeans_sigma

kmeans_mu, kmeans_sigma = run_kmeans(2, 33, X)
gmm_w_kmeans = GMM(2, 0.000001)
mus, nums_iter = gmm_w_kmeans.fit_EM(X, max_iters= 100, mu=kmeans_mu, Sigma=kmeans_sigma, init_w_kmeans=True)
print mus, nums_iter



conv_iters = []
for i in range(50):
    kmeans_mu, kmeans_sigma = run_kmeans(2, i, X)
    mus, iters = gmm_w_kmeans.fit_EM(X, max_iters= 100, mu=kmeans_mu, Sigma=kmeans_sigma, init_w_kmeans=True)
    conv_iters.append(iters)
print(conv_iters)
plt.hist(conv_iters, bins=range(max(conv_iters) + 2), align='left', rwidth=1)
plt.show()
