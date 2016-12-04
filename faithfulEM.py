import math
import sys
import csv
import numpy as np
from matplotlib import pyplot as plt

def em(data,truSd1,truSd2,num_iter=10000):
	sprd = max(data) - min(data)
	mu1 = min(data)+0.25*sprd
	mu2 = min(data)+0.75*sprd
	sd1 = 1#truSd1#1
	sd2 = 1#truSd2#1
	pi1 = 0.5
	pi2 = 1 - pi1
	n = len(data)
    
	def prob(x,mu,sd):
		nator = np.vectorize(math.exp)(((x-mu)*(x-mu))/(-2*sd*sd))
		dator = math.sqrt(2*math.pi)*sd
		return (nator/dator)
	def p1():
		return prob(data,mu1,sd1)
	def p2():
		return prob(data,mu2,sd2)

	for i in range(0,num_iter):
		w = (pi1*p1()) / (pi1*p1()+pi2*p2())
		n1 = np.sum(w)
		n2 = n - n1
		pi1 = n1 / n
		pi2 = 1 - pi1

		mu1 = (1/n1)*np.sum(w*data)
		mu2 = (1/n2)*np.sum((1-w)*data)
	
		sd1 = math.sqrt(np.sum(w*(data-mu1)*(data-mu1)) / n1   )
		sd2 = math.sqrt(np.sum((1-w)*(data-mu2)*(data-mu2)) / n2)
        
	print(str(mu1))    
	print(str(pi1))
	print(str(sd1))
	print(str(mu2))  
	print(str(pi2))
	print(str(sd2))



# Get parameters from call:
sed = int(sys.argv[1])
mu1 = float(sys.argv[2])
mu2 = float(sys.argv[3])
sd1 = float(sys.argv[4])
sd2 = float(sys.argv[5])
pi = float(sys.argv[6])
n1 = int(1000*pi) 
n2 = 1000-n1
	
#np.random.seed(seed=sed)
#x1 = np.random.normal(mu1,sd1,n1)
#x2 = np.random.normal(mu2,sd2,n2)
#x = np.append(x1,x2)

f = open('waittime.csv')
c = csv.reader(f)
i = 0
x = []
for r in c:
	if r != ['x']:
		x.append(float(r[0]))
x = np.array(x)	
em(x,sd1,sd2)





















