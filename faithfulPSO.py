import pandas as pd
import numpy as np
import sys
import math
import csv
from operator import attrgetter
from matplotlib import pyplot as plt

class Particle:
    def __init__(self,position,velocity):
        self.position = position
        self.velocity = velocity
        self.pbest = position
        self.current_fitness = 0
        self.best_fitness = 0
        self.num_parameters = len(self.position)
    def update_position(self,gbest,
                        use_boundary=False,
                        lower_bound=None,
                        upper_bound=None):              
        c1 = 2
        c2 = 2
        r1 = np.random.rand(1)
        r2 = np.random.rand(1)
        delta = 2*r1*(self.pbest - self.position) + 2*r2*(gbest - self.position)
        
        upper_bound=np.zeros(self.num_parameters),
        lower_bound=np.zeros(self.num_parameters)
        
        cond1 = self.velocity + delta <= upper_bound
        cond2 = self.velocity + delta >= lower_bound
        
        if (use_boundary and cond1.all() and cond2.all()) or use_boundary==False:
            self.velocity = self.velocity + delta
            self.position = self.position + self.velocity
    def _phi(self,x,mean,var):
        k = math.sqrt(2*math.pi*var)
        p = math.exp((x-mean)*(x-mean)/(-2*var))
        return p/k
    def _mix_dist(self,x,pi,mean1,mean2,var1,var2):
        comp1 = pi*self._phi(x,mean1,var1)
        comp2 = (1-pi)*self._phi(x,mean2,var2)
        return comp1+comp2
    def _gaussian_likelihood(self,x,pi,mean1,mean2,var1,var2):
        # x is a vector of data
        n = len(x)
        p = 0
        for i in range(0,n):
            p = p + math.log(self._mix_dist(x[i],pi,mean1,mean2,var1,var2)) 
        return p
    def calculate_fitness(self,x,pi,mean1,mean2,var1,var2):
        self.current_fitness = self._gaussian_likelihood(x,pi,mean1,mean2,var1,var2)
        if self.current_fitness > self.best_fitness or self.best_fitness == 0:
            self.pbest = self.position
            self.best_fitness = self.current_fitness

			
def pso(x,truSig1,truSig2,truPi,num_particles=50,num_iter=100):
	N = num_particles
	iterations = num_iter

	data_min = min(x)
	data_max = max(x)
	lower_boundary = np.array([0,data_min,data_min,0,0])
	upper_boundary = np.array([1,data_max,data_max,10,10])    
	particles = [0] * N
	tolerance = 0.001

	# random intialization of particles

	np.random.seed(69)

	# estimating a weight parameter and mu parameter for each of the two gaussian components
	# posn[0] = weight
	# posn[1] = mu1
	# posn[2] = mu2

	for p in range(N):    
		rand_pi = np.random.uniform(0,1)
		rand_mean1 = np.random.uniform(data_min,data_max)
		rand_mean2 = np.random.uniform(data_min,data_max)
		sigma1 = truSig1
		sigma2 = truSig2
		rand_posn = np.array([rand_pi,rand_mean1,rand_mean2])
		rand_velocity = np.array([np.random.uniform(0,1),
								 np.random.uniform(0,1),
								 np.random.uniform(0,1)])
		
		particles[p] = Particle(rand_posn,rand_velocity)
		particles[p].calculate_fitness(x,rand_pi,rand_mean1,rand_mean2,sigma1,sigma2)

		gbest = max(particles,key=attrgetter('best_fitness'))    

	# repeat until convergence
	for i in range(iterations):
		for p in particles:
			p.update_position(gbest.position,True,lower_boundary,upper_boundary)

			params = p.position
			pi = params[0]
			mean1 = params[1]
			mean2 = params[2]
			sigma1 = truSig1  #1
			sigma2 = truSig2 #1
			p.calculate_fitness(x,pi,mean1,mean2,sigma1*sigma1,sigma2*sigma2)
		gbest = max(particles,key=attrgetter('best_fitness'))
	
	# Output results:
	if abs(truPi-gbest.pbest[0]) < abs(truPi-(1-gbest.pbest[0])):  
		print(gbest.pbest[1]) # Est mu1
		print(gbest.pbest[0]) # Est pi1
		print("1")	
		print(gbest.pbest[2]) # Est mu2
		print(1-gbest.pbest[0]) # Est pi2
		print("1")
	else:
		print(gbest.pbest[2]) # Est mu1
		print(1-gbest.pbest[0]) # Est pi1
		print("1")	
		print(gbest.pbest[1]) # Est mu2
		print(gbest.pbest[0]) # Est pi1
		print("1")
	
	#print(gbest.best_fitness)
	return 0

	
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


pso(x,sd1,sd2,pi)













