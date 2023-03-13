import numpy as np
import scipy.stats

A = np.zeros((8,8))

A[0:4,0:4] = np.ones((4,4))
A[5:9,5:9] = np.ones((4,4))

def p_x_giv_z():
    # Calculate P(X|z): the probability of the graph given a particular clustering structure.
    # This is calculated by integrating out all the internal cluster connection parameters.
    # 

    pass

def p_z():
    # Probability distribution over the number of clusters
    # Input parameters:
    
    pass

# logP = sum([betaln(M1+R+a,M0+M-R+b)-betaln(M1+a,M0+b) ... % Log probability of n belonging
# betaln(r+a,m-r+b)-betaln(a,b)],1)' + log([m; A]); % to existing or new component