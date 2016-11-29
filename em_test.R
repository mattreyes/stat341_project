# Code to estimate (via EM) the weight pi for a 
# mixture of two known Gaussian distributions

set.seed(69)
# Parameters for component 1
mu1 <- 5
sd1 <- 2
pi1 <- 0.33
# Parameters for component 2
mu2 <- 10
sd2 <- 40
pi2 <- 1-pi1

N <- 1000
n1 <- pi1*N
n2 <- pi2*N

x1 <- rnorm(n=n1,mean=mu1,sd=sd1)
x2 <- rnorm(n=n2,mean=mu2,sd=sd2)
x <- c(x1,x2)

prob <- function(x,mu,sd) {
	nator <- exp( -((x - mu)^2) / (2*sd*sd) )
	dator <- sqrt(2*pi) * sd
	return(nator / dator)
}

prob1 <- function(x) {
	return(prob(x,mu1,sd1))
}

prob2 <- function(x) {
	return(prob(x,mu2,sd2))
}

# EM estimation
pi1Est <- 0.5
pi2Est <- 1-pi1Est
for(i in 1:1000) {
	w1 <- (pi1Est * prob1(x)) / (pi1Est*prob1(x) + pi2Est*prob2(x))  
	n1Est <- sum(w1)
	pi1Est <- n1Est / N
	pi2Est <- 1-pi1Est
}

pi1Est
pi2Est