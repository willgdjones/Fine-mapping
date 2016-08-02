
# coding: utf-8

# # Initialisation

# In[10]:

# %load /Users/fonz/Projects/Notebooks/Fine-mapping/src/models/__init__.py

# import numpy
# import matplotlib.pyplot as plt
# import math
# from scipy import stats
# import pdb
# from sklearn import preprocessing
# import copy
# from unittest import *
# import itertools
# from bidict import bidict


# # Single trait Fine-mapping

# ## Bayes Factor Computation

# ### Derivation of $z$ values

# To start, we assume that the trait $y$ is modelled as:
# 
# $$ y = X\beta + \epsilon $$
# 
# Where $X$ is an $n$x$m$ matrix of values consisting of 0,1,2 denoting whether a SNP is homozygous to the common allele, heterozygous, or homozygous to the rare allele respectively. $n$ denotes the number of samples, and $m$ the number of causitive SNPs.
# 
# We scale $X$ such that $\frac{1}{n}\sum^{n}_{i=1} X_{ij} = 0$, and $\frac{1}{n}\sum^{n}_{i=1} X^2_{ij} = 1$ for $j = 1,2, ... m$. We also scale $y$ such that $\frac{1}{n}\sum^n_{i=1} y_i = 0$ and $\frac{1}{n}\sum^n_{i=1} y_i^2 = 1$.
# 
# We assume $\epsilon$ ~ $N(0, \frac{1}{\tau} I_n)$. We also assume $\beta$ has a prior normal distribution $N(0,\nu \frac{1}{\tau})$. $\nu$ is diagonal, $\beta$ and $\epsilon$ are independent and we assume all SNPs have the same prior variance $\sigma^2 \frac{1}{\tau}$. Therefore $\nu = \sigma^2 I_m$.
# 
# Now given this prior on $\beta$, and using $X$ and $\epsilon$, we can deduce the expectation and mean of $y$.
# 
# $$E(y \: | \: \tau, X) = E(E(y \: | \: \tau,X,\beta)) = E(X \beta) = 0$$ 
# 
# <sub>[ *since* $E(\beta) = 0$ ]</sub>
# 
# $$ Var(y \: | \: \tau, X) = E(Var(y \: | \tau, X, \beta)) + Var(E(y \: | \: \tau, X, \beta)) $$
# 
# <sub>[ *since* $Var(X \: | \: Y) = E(Var(X \: | \: Y)) + Var(E(X \: | \: Y))$ ]</sub>
# 
# $$ = E(\frac{1}{\tau}I_n) + Var(X \beta)$$
# 
# $$ = \frac{1}{\tau}( I_n + X \nu X^T)$$
# 
# Now, since y is a linear transformation of a multivariate normal random vector,
# 
# $$ y \:|\: \tau, X \sim N \left( 0,\frac{1}{\tau}( I_n + X \nu X^T)) \right) $$
# 
# The null distribution is when $\beta = 0$. In which case,
# 
# $$y \:|\: \tau, X \sim N \left( 0,\frac{1}{\tau}I_n \right) $$
# 
# Now consider a new variable $z = \sqrt{\frac{\tau}{n}} X^{T}y$:
# 
# $$ z ~ \sim N \left( 0, \frac{X^T}{n}(I_n + X \nu X^T) X \right)$$
# 
# $$ = N \left( 0, \left(\frac{X^TX}{n} + \frac{X^TX \nu X^TX}{n}\right) \right)$$
# 
# Now let $\Sigma_x = \frac{X^T X}{n}$. Since all column in $X$ are standardised, this is equivalent to the correlation matrix or, more importantantly, the linkage disequilibirum structure of the SNPs which can be derived from the 1000 genomes data.
# 
# Then we have:
# 
# $$ z \sim N(0, \Sigma_x + \Sigma_x n\nu \Sigma_x) $$
# 

# ### Calculation of Bayes Factor 

# The *Bayes Factor* is the ratio of the likelihood functions under the alternative hypothesis, and under the null hypothesis. It is equivalent to the likelihood ratio.
# 
# $P_1(z \:|\: \tau, X)$, the likelihood of $z$ under our alternate hypothesis, i.e. when $\nu \neq 0$ is:
# 
# $$ P_1(z \:|\: \tau, X) = 2\pi^{-\frac{n}{2}} | \Sigma_x + \Sigma_x n\nu \Sigma_x |^{-\frac{1}{2}} \exp\left(-\frac{1}{2}z^T(\Sigma_x + \Sigma_x n\nu \Sigma_x)^{-1}z\right)$$
# 
# $P_0(z \:| \: \tau, X)$, the likelihood of $z$ under the null hypothesis when $\nu = 0$ is:
# 
# $$P_0(z \:| \: \tau, X) = 2\pi^{-\frac{n}{2}} |\Sigma_x|^{-\frac{1}{2}} \exp\left(-\frac{1}{2}z^T(\Sigma_x)^{-1}z\right)$$
# 
# Therefore we calculate the Bayes Factor as:
# 
# $$ \frac{
# | \Sigma_x + n\nu \Sigma_x^2 |^{-\frac{1}{2}} \exp\left(-\frac{1}{2}z^T(\Sigma_x + \Sigma_x n\nu \Sigma_x)^{-1}z\right)
# }{
# |\Sigma_x|^{-\frac{1}{2}} \exp\left(-\frac{1}{2}z^T(\Sigma_x)^{-1}z\right)
# }
# $$
# 
# We assume that $X$ has full column rank, and that $\Sigma_x$ also has rank $m$ and is non-singular. That is to say, we assume that no two snps are in full linkage disequilibrium.
# 
# Using the Woodberry matrix identity:
# 
# $$
# (\Sigma_x + \Sigma_x n\nu \Sigma_x)^{-1} = \Sigma_x^{-1} - ((n\nu)^{-1} + \Sigma_x)^{-1}
# $$
# 
# Therefore the resulting Bayes Factor is:
# 
# $$
# |I_m + n\nu \Sigma_x|^\frac{1}{2} \exp(\frac{1}{2}z^T((n\nu)^{-1} + \Sigma_x)^{-1}z)
# $$
# 
# Crucially, this only depends on inverting matrices of size m, our candidate gene set. Therefore we compute these Bayes Factors using sets of candidate SNPs of size m, and choose the set with the highest calculated Bayes Factor.
# 
# In practice, we recieve $\beta$, $se(\beta)$, and the SNP linkage disequilibrium structure $\Sigma_x$.
# 
# Since both $X$ and $y$ are normalised, 
# 
# $$\beta = \frac{X^T y}{n}$$
# 
# Also, 
# $$\tau = \frac{1}{\sigma^2}, \:\: se(\epsilon) = \frac{\sigma}{\sqrt{n}}$$
# 
# where $\sigma$ is the observed standard deviation of the errors $\epsilon$.
# 
# Therefore:
# $$
# se(\epsilon) = \frac{1}{\sqrt{n\tau}}
# $$
# 
# Therefore we generate the $z$ vector exactly with and $se$ is the standard error:
# 
# $$
# \frac{\beta}{se(\beta)} = \sqrt{\frac{\tau}{n}} X^{T}y = z
# $$
# 
# The Bayes Factor can then be directly calculated using $z$ and $\Sigma_x$.
# 
# 

# ### Calculation of Posterior

# We place a binomial prior on candidate gene sets. If our gene set $G$ has size $m$, we assume that each SNPs has probability  $p = \frac{1}{m}$ of being causal. Therefore the prior probability of a causal gene set with size $l$ is:
# 
# $$
# P(G) = p^l(1-p)^{m-l}
# $$
# 
# Therefore using Bayes Theorem:
# 
# $$
# P(G \: | \: X) = \frac{P(X \: | \: G) \times P(G)}{P(X)}
# $$
# 
# to calculate posterior probabilities of the gene sets where $P(X \: | \: G)$ is calculated from the normalised Bayes Factors.
# 
# However when we calculated the Bayes Factors, these are not exactly the likelihoods. They are however far easier to compute.
# 
# The Bayes Factors we have calculated are equivalent to:
# 
# $$
# \frac{P(X \: | \: G)}{P(X \: | \: G_0)}
# $$
# 
# where $G_0$ is the null hypothesis that no gene-set is casual.
# 
# However, since $P(X \: | \: G_0)$ is a constant for all gene-sets, this is proportional to the likelihood term. Therefore we can normalise to output the posterior probability distributions.

# ### Implementation

# In[1]:

import numpy
import itertools
import math

### Create selection of SNPs
def select_snps(z, subset):
    return [z[i] for i in subset]

#example
# for subset in it.combinations(range(len(z1)),3):
#     print subset, select_snps(z1, subset)    



### Select covariance submatrix

def select_cov(cov, subset):
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
#     pdb.set_trace()
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
    if prior == 'binomial':
        p = 1./m
        l = len(x)
        return p**l * (1-p)**(m-l)
    else:
        return None
    
# example
# calc_prior((1,3,5),30)
    
def calc_posterior(variant_set_BF,prior='binomial'):
    
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



# # Example

# In[3]:

get_ipython().magic(u'reset -f')
if __name__ == "__main__":
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
    z2 = numpy.array(numpy.divide(s_tss_2['beta'],numpy.sqrt(s_tss_1['var_beta'])))
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

