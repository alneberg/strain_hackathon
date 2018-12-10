import sys, getopt
import pandas as p
import numpy as np
import scipy.misc as spm
import scipy as sp
from itertools import product, tee, izip
import itertools as it
import re
from Bio import SeqIO

class Constants(object):
    THRESHOLD = 1000
    KMER_LEN  = 4
    MAX_CCOUNT = 5 
    THETA = 1.0e-7
    DELTA = 10.0
    SAMPLE_SIZE = 1000

class CSample(object):
    #nG number of genomes
    #nS number of sites
    #nC number of contigs

    #aadPi probability distribution of genomes over sites
    #aanEta genome composition across contigs
    #con_counts contig counts across samples
    
    #no. of states   
 
    #assignment of reads to genomes in each site
    def __init__(self, gamma, eta):
        
        self.aadPi = gamma.as_matrix()
        self.nG = self.aadPi.shape[1]
        self.nS = self.aadPi.shape[0]

        self.aanEta = eta.as_matrix()
        self.nC = self.aanEta.shape[1]

        self.lengths = np.zeros(self.nC)
        self.lengths.fill(1.0)
        # scaling coverage by contig lengths

        self.anN = np.zeros(self.nS,dtype=np.int)
        self.anN.fill(Constants.SAMPLE_SIZE)

        self.con_counts = np.zeros((self.nC,self.nS),dtype=np.int)
        
    def sampleReads(self):
        #first sample genomes in each site
        z_counts = np.zeros((self.nS,self.nG),dtype=np.int)
        for s in range(0, self.nS):
            z_counts[s,:] = np.random.multinomial(self.anN[s],self.aadPi[s,:]) 

        adEL = self.aanEta*self.lengths
        row_sums = adEL.sum(axis=1)
        adELP = adEL / row_sums[:, np.newaxis]
        con_countst = np.zeros((self.nS,self.nC),dtype=np.int)
        for s in range(0, self.nS):
            for g in range(0, self.nG):
                con_countst[s,:] += np.random.multinomial(z_counts[s,g],adELP[g,:])
        self.con_counts = con_countst.transpose()
    
    def calcLogPoissonProb(self,adNu,anC):
        #loop sites
        dLogProb = 0.0;    
        for s in range(self.nS):
            dLogProb = anC[s]*np.log(adNu[s]*self.anN[s]) - adNu[s]*self.anN[s]
        
        return dLogProb

    def sampleLogProb(self,adLogProbS):
        dP = np.exp(adLogProbS - np.max(adLogProbS))
        dP = dP/np.sum(dP,axis=0)
        return np.flatnonzero(np.random.multinomial(1,dP,1))[0]


    def sampleEta(self):
        #loop each contig in turn
        dTotalNorm = np.dot(self.aadPi,self.aanEta*np.array(self.lengths))
        #import ipdb; ipdb.set_trace()
        for c in range(self.nC):
            adLogProbS = np.zeros(self.nStates,dtype=np.float64)
            temp = np.delete(dTotalNorm,(c), axis=1)
            dNormSC = np.sum(temp,axis=1)
            for s in range(self.nStates):
                adNu = np.dot(self.aadPi,self.states[s,:])*self.lengths[c]
                adNu = adNu/(dNormSC + adNu)
                dLogProbS = self.calcLogPoissonProb(adNu,self.con_counts[c,:])
                dLogProbS -= Constants.THETA*np.sum(self.states[s,:],axis=0)*self.lengths[c]
                adLogProbS[s] = dLogProbS
            nSS = self.sampleLogProb(adLogProbS)
            self.aanEta[:,c] = self.states[nSS,:] 
            print c 
            print self.states[nSS,:] 
    
    #Sample assignments of reads to genomes
    def sampleGenomeAssignments(self):

        #loop contigs
        for c in range(self.nC):
            dP = self.aadPi*self.aanEta[:,c]
            row_sums = np.sum(dP,axis=1)
            dP = dP / row_sums[:, np.newaxis]
            
            #loop sites
            for s in range(self.nS):
                self.aaanM[c,s,:] = np.random.multinomial(self.con_counts[c,s], dP[s,:])
    
    def samplePi(self):
        for s in range(self.nS):
            dTemp = np.zeros(self.nG,dtype=np.float64)
            for g in range(self.nG):
                for c in range(self.nC):
                    if(self.aanEta[g,c] > 0):
                        dTemp[g] += self.aaanM[c,s,g]/(self.aanEta[g,c]*self.lengths[c])
            dTemp += Constants.DELTA/self.nG
            self.aadPi[s,:] = np.random.dirichlet(dTemp)

def main(argv):
    eta_file = ''
    gamma_file = ''
    output_file = ''
    try:
        opts, args = getopt.getopt(argv,"he:g:o:",["eta_file=","gamma_file=","output_file="])
    except getopt.GetoptError:
        print 'GenModel.py -e <eta_file> -g <gamma_file> -o <ouput_file>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'GenModel.py -e <eta_file> -g <gamma_file> -o <ouput_file>'
            sys.exit()
        elif opt in ("-e", "--eta_file"):
            eta_file = arg
        elif opt in ("-g", "--gamma_file"):
            gamma_file = arg
        elif opt in ("-o", "--output_file"):
            output_file = arg

    eta = p.read_csv(eta_file,index_col=0) 
    pi = p.read_csv(gamma_file,index_col=0)
    import ipdb; ipdb.set_trace()
    sample = CSample(pi, eta)
    sample.sampleReads()
    rownames = eta.columns.values.tolist()
    ca = np.array(sample.con_counts)
    counts_df = p.DataFrame(data=ca,index=rownames,columns=pi.index.tolist()) 
    counts_df.to_csv(output_file) 
    import ipdb; ipdb.set_trace() 
if __name__ == "__main__":
    main(sys.argv[1:])

