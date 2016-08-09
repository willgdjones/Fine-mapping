
# coding: utf-8

# In[1]:

import numpy
import itertools
import math
from copy import copy
import pdb

### Create selection of SNPs
def select_snps(z, subset):
    """
    Given a gene subset and a vector of z-scores, extract out the appropriate z-scores.
    """
    return [z[i] for i in subset]

#example
# for subset in it.combinations(range(len(z1)),3):
#     print subset, select_snps(z1, subset)    



### Select covariance submatrix

def select_cov(cov, subset):
    """
    Given a gene subset and a covariance matrix, extract out the appropriate rows and columns
    """
    return cov[numpy.ix_(subset,subset)]

#example   
#select_cov(LD_tss_1, (0,1,5))

### Calculate Bayes Factor

def calc_BF(z, cov,n,v=0.1):
    """
    Calculate the Bayes factor of a single set of candidate SNPs effect sizes z,
    covariance matrix cov, a prior variance on beta v, and a sample
    size n.
    """
    z = numpy.matrix(z)
    z = z.T
    v_matrix = numpy.matrix(numpy.eye(len(z)) * v)
    coeff = 1. / math.sqrt(numpy.linalg.det((numpy.matrix(numpy.eye(len(z))) + n * v_matrix * numpy.matrix(cov))))
    exponent = 0.5* z.T * numpy.matrix(numpy.linalg.pinv((n*v_matrix).I + cov)) * z
    return numpy.array(math.log(coeff) + exponent)[0][0]

# example
# subset = (0,1,5,8)
# cov = select_cov(LD_tss_1, subset)
# z = select_snps(z1, subset)
# v = np.eye(len(z))/1000
# n = 1000
# calc_BF(z,cov,v,n)

def calc_prior(x,m,prior='binomial'):
    """
    Given the total candidate gene-set size, and the size of a candidate gene-set,
    return the calculated posterior.
    """
    if prior == 'binomial':
        p = 1./m
        l = len(x)
        return p**l * (1-p)**(m-l)
    else:
        return None
    
# example
# calc_prior((1,3,5),30)
    
def calc_posterior(variant_set_BF,prior='binomial'):
    """
    Given gene-set and bayes factor pairs, return the gene-set posterior probability pairs.
    """
    priors = [math.log(calc_prior(x[0],30)) for x in variant_set_BF]  
    log_bayes_factors = [x[1] for x in variant_set_BF]
    unscaled_log_posteriors = [ log_bayes_factors[i] + priors[i] for i in range(len(log_bayes_factors))]
    scaled_log_posteriors = numpy.array(unscaled_log_posteriors) - max(unscaled_log_posteriors)
    scaled_posteriors = [math.exp(x) for x in scaled_log_posteriors]
    calib_factor = sum([math.exp(x) for x in scaled_log_posteriors])
    posteriors = [x/calib_factor for x in [math.exp(x) for x in scaled_log_posteriors]]  
    aug_posteriors = [(variant_set_BF[i][0], posteriors[i]) for i in range(len(posteriors))]   
    aug_posteriors.sort(key=lambda x: x[1], reverse=True)
    
    return aug_posteriors





def calc_variant_set_BFs(data,k,v=0.1,prior='binomial'):
    """
    Calculate variant set posteriors with a binomial prior as normal,
    searching all variant sets up till size k.
    v is the prior variance on beta.
    data has the format (z,LD,n) where z is the effect sizes, 
    LD is the linkage disequilibrium matrix, and n is the 
    number of samples.
    """
    bayes_factors = []
    for i in range(1,k):
        for subset in itertools.combinations(range(len(data[0])),i):
            z = select_snps(data[0], subset)
            cov = select_cov(data[1],subset)
            n = data[2]
            bayes_factors.append((subset, calc_BF(z, cov,n,v)))
    
    bayes_factors.sort(key=lambda x: x[1], reverse=True)
    return bayes_factors



def get_neighbourhood(subset, data):
    
    total_set = set(range(1, len(data[0])))
    out_set = list(total_set - set(subset))

    delete_neighbourhood = []
    add_neighbourhood = []
    change_neighbourhood = []

    for k in subset:
        neighbour = copy(subset)
        neighbour.remove(k)
        delete_neighbourhood.append(neighbour)


    for m in out_set:
        neighbour = copy(subset)
        neighbour.append(m)
        add_neighbourhood.append(neighbour)

    for k in subset:
        for m in out_set:
            neighbour = copy(subset)
            neighbour.remove(k)
            neighbour.append(m)
            change_neighbourhood.append(neighbour)

    neighbourhood = delete_neighbourhood + add_neighbourhood + change_neighbourhood
    return neighbourhood


def calc_neighbourhood_and_get_max(neighbourhood, data, subset_score_hash):
    """
    Given a gene-set neighbourhood, return the neighbour with the highest bayes factor, and add the results of the
    others to the subset_score_hash.
    """
    max_bf = -float('inf')
    max_subset = []

    for subset in neighbourhood:
        z = select_snps(data1[0], subset)
        cov = select_cov(data1[1],subset)
        n = data1[2]

        try:
            subset_bf = subset_score_hash[str(sorted(subset))]
            if subset_bf > max_bf:
                max_bf = subset_bf
                max_subset = subset

        except KeyError:
            z = select_snps(data[0], subset)
            cov = select_cov(data[1],subset)
            n = data[2]
            subset_bf = calc_BF(z, cov,n,v)
            subset_score_hash[str(sorted(subset))] = subset_bf

            if subset_bf > max_bf:
                max_bf = subset_bf
                max_subset = subset
        
    return (max_subset, max_bf)


def stochastic_search(data,start,iterations=30):
    """
    Start a stochastic search with the input data, and a maximum number of iterations.
    Loops terminate the walk.
    """
    subset_score_hash = {'[]':0}
    subset_walk = []
    subset = start

    for i in range(iterations):
        neighbourhood = get_neighbourhood(subset,data)
        max_subset, max_bf = calc_neighbourhood_and_get_max(neighbourhood,data, subset_score_hash)
        if max_subset in [x[0] for x in subset_walk]:
            subset_walk.append((max_subset, max_bf))
            return subset_walk
        else:
            subset_walk.append((max_subset, max_bf))
            subset = max_subset
    
    return subset_walk


# In[8]:

if __name__ == "__main__":
    s_tss_1=numpy.load('../data/raw/summary_stats_g1_tss60.npy')[0]
    s_tss_2=numpy.load('../data/raw/summary_stats_g2_tss60.npy')[0]
    LD_tss_1=numpy.load('../data/raw/LD_g1_TSS60.npy')
    LD_tss_2=numpy.load('../data/raw/LD_g2_TSS60.npy')
    n1 = 10000
    n2 = 1000
    z1 = numpy.array(numpy.divide(s_tss_1['beta'],numpy.sqrt(s_tss_1['var_beta'])))
    z2 = numpy.array(numpy.divide(s_tss_2['beta'],numpy.sqrt(s_tss_2['var_beta'])))
    z1 = numpy.ndarray.flatten(z1)
    z2 = numpy.ndarray.flatten(z2)



    ### Initialise hyper parameters
    k=3
    v=0.1
    data1 = (z1, LD_tss_1, 10000)
    data2 = (z2, LD_tss_2, 1000)

    stochastic_search(data2,[9])


# In[180]:


if __name__ == "__main__":
    get_ipython().magic(u'reset -f')
    import sys
    sys.path.append('/Users/fonz/Projects/Notebooks/Fine-mapping/src')
    import models
    import numpy
    import math

    s_tss_1=numpy.load('../data/raw/summary_stats_g1_tss60.npy')[0]
    s_tss_2=numpy.load('../data/raw/summary_stats_g2_tss60.npy')[0]
    LD_tss_1=numpy.load('../data/raw/LD_g1_TSS60.npy')
    LD_tss_2=numpy.load('../data/raw/LD_g2_TSS60.npy')

    ### Generate z arrays

    n1 = 10000
    n2 = 1000
    z1 = numpy.array(numpy.divide(s_tss_1['beta'],numpy.sqrt(s_tss_1['var_beta'])))
    z2 = numpy.array(numpy.divide(s_tss_2['beta'],numpy.sqrt(s_tss_2['var_beta'])))
    z1 = numpy.ndarray.flatten(z1)
    z2 = numpy.ndarray.flatten(z2)

    ### Initialise hyper parameters
    k=3
    data1 = (z1, LD_tss_1, 10000)
    data2 = (z2, LD_tss_2, 1000)

    ### Calculate variant set Bayes Factors
    set1 = models.bayes_factors.calc_variant_set_BFs(data1,k)
    set2 = models.bayes_factors.calc_variant_set_BFs(data2,k)

    ### Calculate variant set posteriors
    posteriors1 = models.bayes_factors.calc_posterior(set1)
    posteriors2 = models.bayes_factors.calc_posterior(set2)

    posteriors1.sort(key=lambda x: x[1], reverse=True)
    posteriors2.sort(key=lambda x: x[1], reverse=True)

    posteriors2[0:10]

