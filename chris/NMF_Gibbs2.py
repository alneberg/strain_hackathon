import sys, getopt
import pandas as p
import numpy as np
import scipy.stats as ss
import scipy as sp
import math
import argparse
from operator import mul, div, eq, ne, add, ge, le, itemgetter

def elop(X, Y, op):
    try:
        X[X == 0] = np.finfo(X.dtype).eps
        Y[Y == 0] = np.finfo(Y.dtype).eps
    except ValueError:
        return op(np.mat(X), np.mat(Y))
    return op(np.mat(X), np.mat(Y))

class NMF():

    def __init__(self,V,rank,n_run=None,max_iter=None,min_change=None):
        self.name = "NMF"

        self.V = V
        self.rank = rank
        
        if n_run is None:
            self.n_run = 1
        else:
             self.n_run = n_run

        if max_iter is None:
            self.max_iter = 5000
        else:
             self.max_iter = max_iter

        if min_change is None:
            self.min_change = 1.0e-3
        else:
            self.min_change = min_change

    def factorize(self):
    
        for run in xrange(self.n_run):
            self.random_initialize()
        
            divl = 0.0
            div = self.div_objective()
            iter=0
            while iter < self.max_iter and math.fabs(divl - div) > self.min_change:
                self.div_update()
                self._adjustment()
                divl = div
                div = self.div_objective()
 
                print str(iter) + "," + str(div)

                iter += 1

    def random_initialize(self):
        self.max = self.V.max()
        self.prng = np.random.RandomState()
            
        self.W = self.gen_dense(self.V.shape[0], self.rank)
        self.H = self.gen_dense(self.rank, self.V.shape[1])

        
    def gen_dense(self, dim1, dim2):
        return np.mat(self.prng.uniform(0, self.max, (dim1, dim2)))

    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = np.maximum(self.H, np.finfo(self.H.dtype).eps)
        self.W = np.maximum(self.W, np.finfo(self.W.dtype).eps)

    def euc_update(self):
        """Update basis and mixture matrix based on Euclidean distance multiplicative update rules."""
        self.H = np.multiply(
            self.H, elop(np.dot(self.W.T, self.V), np.dot(self.W.T, np.dot(self.W, self.H)), div))
        self.W = np.multiply(
            self.W, elop(np.dot(self.V, self.H.T), np.dot(self.W, np.dot(self.H, self.H.T)), div))
    
    def div_update(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        H1 = np.tile(np.asmatrix(self.W.sum(0).T), (1, self.V.shape[1]))
        self.H = np.multiply(
            self.H, elop(np.dot(self.W.T, elop(self.V, np.dot(self.W, self.H), div)), H1, div))
        W1 = np.tile(np.asmatrix(self.H.sum(1).T), (self.V.shape[0], 1))
        self.W = np.multiply(
            self.W, elop(np.dot(elop(self.V, np.dot(self.W, self.H), div), self.H.T), W1, div))  

 
    def div_updateHZ(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        H1 = np.tile(np.asmatrix(self.Z.sum(0).T), (1, self.V.shape[1]))
        self.HZ = np.multiply(
            self.HZ, elop(np.dot(self.Z.T, elop(self.V, np.dot(self.Z, self.HZ), div)), H1, div))
 
    def fro_objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.V - np.dot(self.W, self.H)
        return np.multiply(R, R).sum()
 
    def div_objective(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = np.dot(self.W, self.H)
        return (np.multiply(self.V, np.log(elop(self.V, Va, div))) - self.V + Va).sum()

    def div_objectiveHZ(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = np.dot(self.Z, self.HZ)
        return (np.multiply(self.V, np.log(elop(self.V, Va, div))) - self.V + Va).sum()

    def discretiseW(self):
        self.Z = np.asmatrix(np.zeros((self.V.shape[0],self.rank), dtype=np.int)) 
        for i in range(self.rank):
            dgm = discreteGammaMixture(self.W[:,i])
            dgm.fit()
            for j in range(self.V.shape[0]):
                self.Z[j,i] = dgm.Z[j]

        self.HZ = np.copy(self.H)
        
        self.ZT = self.Z.T        
        divl = 0.0
        div = self.div_objectiveHZ()

        iter=0
        while iter < self.max_iter and math.fabs(divl - div) > self.min_change:
            self.div_updateHZ()
               
            divl = div
            div = self.div_objectiveHZ()

            print "hz," + str(iter) + "," + str(div)
            iter = iter + 1
    def __str__(self):
        return self.name

class discreteGammaMixture:

    """
    Fits gamma mixture with fixed interval to data set

    Parameters
    ---------
    
    N - no. of data points

    Z - assignments

    k0 - shape parameter for zeroth assignment
    
    theta0 - scale parameter for zeroth assignment
    
    k1 - shape parameter for first assignment

    theta1 - scale parameter for first assignment

    delta - interval

    """

    def __init__(self,data,max_iter=None,min_change=None):
        
        self.data = data

        self.delta = sp.median(data,axis=0)

        self.Z = np.around(np.divide(data,self.delta))
        self.Z.astype(int)
        self.N = self.data.shape[0]

        if max_iter is None:
            self.max_iter = 100
        else:
            self.max_iter = max_iter

        if min_change is None:
            self.min_change = 1.0e-3
        else:
            self.min_change = min_change

    def fitZero(self):
        zeros = np.extract(self.Z == 0, self.data)

        (self.k0,loc,self.theta0) = ss.gamma.fit(zeros, floc=0.0)

        #self.theta0 = np.sum(zeros,axis=0)/zeros.shape[0]

    def fitOne(self):
        ones = np.extract(self.Z == 1, self.data)
        
        (self.k1,loc,self.theta1) = ss.gamma.fit(ones, floc=0.0)

        self.delta = (self.k1 - 1)*self.theta1

    def assignZ(self):
        log1 = ss.gamma.logpdf(self.data, a=self.k1, loc=0.0, scale=self.theta1)
        
        log0 = ss.gamma.logpdf(self.data, a=self.k0, loc=0.0, scale=self.theta0)
        
        self.Z[log1 < log0] = 0
        
        self.Z[log1 >= log0] = 1

        ZD = np.around(np.divide(self.data,self.delta))

        for i in range(self.N):
            if ZD[i] > 1:
                self.Z[i] = ZD[i]

    def fit(self):
        iter = 0
        delta = 0.0
        while (iter < self.max_iter and delta < self.min_change):
            self.fitZero()

            self.fitOne()

            self.assignZ()
            print str(iter) + " " + str(self.k0) + " " + str(self.theta0) + " " + str(self.k1) + " " + str(self.theta1)
            iter = iter + 1   

class BayesDiscreteNMF(object):
    #G number of genomes
    #S number of sites
    #C number of contigs

    #eta genome composition across contigs
    #cov contig coverages across samples

    def __init__(self,counts,G,max_iter=None,lengths=None,a=None,b=None):
        if max_iter is None:
            self.max_iter = 100
        else:
             self.max_iter = max_iter

        #set read counts per contig per sample
        self.contig_counts = np.copy(counts) #does this need a deep copy?

        #set number of genomes
        self.G = G

        #set number of contigs
        self.C = self.contig_counts.shape[0] 

        #set number of samples
        self.S = self.contig_counts.shape[1] 

        #initialise eta as zeros
        self.eta = np.zeros([self.C, self.G],dtype=np.int32)

        #initialise auxiliary counts
        self.sources = np.zeros((self.C,self.G,self.S),dtype=np.int32)

        #initialise gamma as zeros
        self.gamma = np.zeros([self.G, self.S],dtype=np.float64)

        if lengths is None:
            self.lengths = np.ones(self.C)

        if a is None:
            self.a = 1.0e-3

        #sum constants needed for Poisson sampling
        self.ZERO_PENALTY = 100.0 #penalisation factor for observations given zero rate
        self.MAX_COPY     = 10    #maximum possible copy number

        if b is None:
            self.b = 1.0e3

    def setEta(self,newEta):
        self.eta = np.copy(newEta)

    def setGamma(self,newGamma):
        self.gamma = np.copy(newGamma)

    def update(self): #perform one Gibbs update as per Cemgil algorithm 2
        
        iter = 0
        while (iter < self.max_iter):
            self.updateSources()
            
            self.updateEta()
        
            self.updateGamma()
    
            iter = iter + 1

    def updateSources(self): #update source assignments
        #loop contigs
        for c in range(self.C):
            prob = self.gamma.T*self.eta[c,:]
            row_sums = np.sum(prob,axis=1)
            prob = prob / row_sums[:, np.newaxis]
            
            #loop sites
            for s in range(self.S):
                self.sources[c,:,s] = np.random.multinomial(self.contig_counts[c,s], prob[s,:])

    def calcLogPoissonProb(self,eta,sourceCopy,gammaCopy):
        #loop sites
        dLogProb = 0.0;
        for s in range(self.S):
            rate = eta*gammaCopy[s]
            if rate > 0.0:
                dLogProb += sourceCopy[s]*np.log(rate) - rate
            else :
                dLogProb -= sourceCopy[s]*self.ZERO_PENALTY
        return dLogProb

    def sampleLogProb(self,adLogProbS):
        dP = np.exp(adLogProbS - np.max(adLogProbS))
        dP = dP/np.sum(dP,axis=0)
        return np.flatnonzero(np.random.multinomial(1,dP,1))[0]


    def updateEta(self):
        #need to numpify this in the future

        for g in range(self.G):#loop genomes
            for c in range(self.C):#loop contigs
                probEta = np.zeros(self.MAX_COPY)
                for m in range(self.MAX_COPY):
                    probEta[m] = self.calcLogPoissonProb(m,self.sources[c,g,:],self.gamma[g,:])

                self.eta[c,g] = self.sampleLogProb(probEta)
    
    def updateGamma(self):
        #sum up counts for each genome in each sample
        sourceNorm = self.sources / self.lengths[:,np.newaxis,np.newaxis]
        SigmaG = sourceNorm.sum(axis = 0)
        
        AlphaG = self.a + SigmaG
 
        t1 = self.eta.sum(0).T
        G1 = t1[:,np.newaxis].repeat(self.S, axis=1)
        
        BetaG = self.a/self.b + G1
        BetaG = 1.0/BetaG

        self.gamma = np.random.gamma(AlphaG, BetaG)

def main(args):
    cov_file = ''

    cov    = p.read_csv(args.cov_file, header=0, index_col=0)
    counts = p.read_csv(args.count_file, header=0, index_col=0)

    if args.vanilla_nmf:
        nmf = NMF(np.array(cov.as_matrix(),dtype=np.float64),2)
    
        nmf.factorize()
        nmf.discretiseW()
    
        np.savetxt("Z.csv", nmf.Z, delimiter=',')
        np.savetxt("W.csv", nmf.W, delimiter=',') 
        np.savetxt("HZ.csv", nmf.HZ, delimiter=',')
        np.savetxt("H.csv", nmf.H, delimiter=',')
    
    else:
        bdnmf = BayesDiscreteNMF(np.array(counts.as_matrix(),dtype=np.int32),2)

        max_val = counts.as_matrix().max()
        eta_dim1 = counts.as_matrix().shape[0]
        eta_dim2 = 2
        gamma_dim1 = 2
        gamma_dim2 = counts.as_matrix().shape[1]

        random_state = np.random.RandomState()
        bdnmf.setGamma(np.mat(random_state.gamma(bdnmf.a, bdnmf.b, (gamma_dim1, gamma_dim2))))
        bdnmf.setEta(np.mat(random_state.gamma(bdnmf.a, bdnmf.b, (eta_dim1, eta_dim2))))
        import ipdb; ipdb.set_trace()
        #bdnmf.setGamma(nmf.HZ)
        #bdnmf.setEta(nmf.Z)

        bdnmf.update()
        np.savetxt("eta.csv", bdnmf.eta, delimiter=',')
        np.savetxt("gamma.csv", bdnmf.gamma, delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vanilla_nmf', action='store_true',
            help="Use the regular NMF algorithm, default is the gibbs discrete NMF")
    parser.add_argument('-v', '--cov_file', required=True,
            help="The coverage file")
    parser.add_argument('-c', '--count_file', required=True,
            help="The count file")
    args = parser.parse_args()
    main(args)
