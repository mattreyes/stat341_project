import math
import numpy as np



def em(data,num_iter=10000):
    mu1 = 0
    mu2 = 0
    sd1 = 1
    sd2 = 1
    pi1 = 0.65
    pi2 = 1 - pi1
    n = len(data)
    
    def prob(x,mu,sd):
        nator = np.vectorize(math.exp)(((x-mu)*(x-mu))/(-2*sd*sd))
        dator = math.sqrt(2*math.pi)*sd
        return (nator/dator)
    def p1():
        return prob(data,mu1,sd2)
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

        sd1 = math.sqrt(np.sum(w*(data-mu1)*(data-mu1))   )
        sd2 = math.sqrt((1/n2)*np.sum((1-w)*(data-mu2)*(data-mu2)))
        
    print(str(mu1))    
    print(str(pi1))
    print(str(sd1))
    print(str(mu2))
    print(str(pi2))
    print(str(sd2))
    


x1 = np.random.normal(0,1,600)
x2 = np.random.normal(6,3,400)
x = np.append(x1,x2)

em(x)



