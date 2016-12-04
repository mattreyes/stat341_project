import math
import sys
import numpy as np
from matplotlib import pyplot as plt


def phi(x,mean,var):
	k = math.sqrt(2*math.pi*var)
	p = math.exp((x-mean)*(x-mean)/(-2*var))
	return p/k
def mix_dist(x,pi,mean1,mean2,var1,var2):
	comp1 = pi*phi(x,mean1,var1)
	comp2 = (1-pi)*phi(x,mean2,var2)
	return comp1+comp2
def gaussian_likelihood(x,pi,mean1,mean2,var1,var2):
	# x is a vector of data
	n = len(x)
	p = 0
	for i in range(0,n):
		p = p + math.log(mix_dist(x[i],pi,mean1,mean2,var1,var2)) 
	return p


def em(data,truSd,max_iter=20000):
	#mu1 = 0
	#mu2 = 0
	sd1 = 1#truSd#1
	sd2 = 1#truSd#1
	sprd = max(data) - min(data)
	mu1 = min(data)+0.25*sprd
	mu2 = min(data)+0.75*sprd
	pi1 = 0.5
	pi2 = 1 - pi1
	n = len(data)
	
	def prob(x,mu,sd):
		nator = np.vectorize(math.exp)(((x-mu)*(x-mu))/(-2*sd*sd))
		dator = math.sqrt(2*math.pi)*sd
		return (nator/dator)
	
	logL = gaussian_likelihood(data,pi1,mu1,mu2,sd1*sd1,sd2*sd2)
	
	for i in range(0,max_iter):
		p1 = prob(data,mu1,sd1)
		p2 = prob(data,mu2,sd2)
		w = (pi1*p1) / ((pi1*p1)+(pi2*p2))
		n1 = np.sum(w)
		n2 = n - n1
		pi1 = n1 / n
		pi2 = 1 - pi1

		mu1 = (1/n1)*np.sum(w*data)
		mu2 = (1/n2)*np.sum((1-w)*data)
		
		newLogL = gaussian_likelihood(data,pi1,mu1,mu2,sd1*sd1,sd2*sd2)
		
		if abs(newLogL-logL) < 0.0001:
			break
		
		logL = newLogL

		sd1 = math.sqrt(np.sum(w*(data-mu1)*(data-mu1)) / n1   )
		sd2 = math.sqrt((1/n2)*np.sum((1-w)*(data-mu2)*(data-mu2)))
		
	print(str(mu1))    
	print(str(pi1))
	print(str(sd1))
	print(str(mu2)) # 
	print(str(pi2)) # did not change
	print(str(sd2)) # correct



# Get parameters from call:
sed = int(sys.argv[1])
mu1 = float(sys.argv[2])
mu2 = float(sys.argv[3])
sd1 = float(sys.argv[4])
sd2 = float(sys.argv[5])
pi = float(sys.argv[6])
n1 = int(200*pi) 
n2 = 200-n1
	
np.random.seed(seed=sed)
x1 = np.random.normal(mu1,sd1,n1)
x2 = np.random.normal(mu2,sd2,n2)
x = np.append(x1,x2)

em(x,sd1)





















