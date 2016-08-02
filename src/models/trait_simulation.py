
# coding: utf-8

# In[3]:

import numpy
import math
from scipy import stats

### Sample genotypes

def simulate_genotype(n,m,geno_dist):
    """
    Simulate a genotype of n samples and m causal SNPs with specified genotype distribution for (0,1,2).
    """
    X=numpy.zeros([n,m])
    for i in range(m):
        X[:,i] = [numpy.random.choice(a=[0,1,2],p=geno_dist) for x in range(n)]
    return numpy.array(X)

###example
# X = simulate_genotype(n=10000,m=30,geno_dist=[0.85,0.1,0.05])

def simulate_traits(X,snp_ratios,beta_var):
    """
    X is genotype information.
    
    beta_var is the total heteritability and must be between 0 and 1.
    
    snp_ratio is a dictionary where dictionary values signify the ratio of how much each snp explains 
    of the total heritability.
    
    For example beta_var = 0.6, snp_ratios = {3:1, 5:1, 7:2} means that the effect of SNP 7 is twice as great as
    snp 3 and 5, whose effects are equal. In total the snps account for 60% of the observed variance.
    
    """
    
    eps_var = 1 - beta_var
    u = math.sqrt(1.0 /sum([x*x for x in snp_ratios.values()]))
    snp_betas = dict([(key,snp_ratios[key]*u) for key in snp_ratios.keys()])
    beta = snp_betas.values()
    snps = snp_betas.keys()
    eps_vector = numpy.array(numpy.random.normal(0,eps_var,X.shape[0]))
    return numpy.add(numpy.dot(X[:,snps], beta), eps_vector)
    
# examples
# y = simulate_traits(X,eps=0.5,snp_group={3: 5, 9: 3})

def build_linear_models(X,y):
    """
    Build univariate linear models for each SNP column in X against the trait y.
    """
    return [stats.linregress(X[:,i],y) for i in range(X.shape[1])]

# example
# models1 = [x for x in build_linear_models(X,y)]

def calc_effect_sizes(models):
    """
    Calculate the effect sizes = beta / se(beta) of individual SNPs towards the traits.
    Takes in a list of linear regression models.
    """
    return [x.slope / x.stderr for x in models]

# example
# z1 = [x.slope / x.stderr for x in models1]



# In[7]:

if __name__ == '__main__':
    get_ipython().magic(u'reset -f')
    import sys
    sys.path.append('/Users/fonz/Projects/Notebooks/Fine-mapping/src')
    import models
    from sklearn import preprocessing
    import numpy
    snp_ratios = [{1: 1}, {1: 1, 3: 2}, {1: 1, 3: 1, 15:2}, {1: 1, 3: 1, 15:1, 25:2}]

    for r in snp_ratios:
        n = 1000
        m = 30

        ### simulate genotypes
        X = models.trait_simulation.simulate_genotype(n,m,geno_dist=[0.85,0.1,0.05])
        ### scale columns
        X = preprocessing.scale(X)

        ### calculate LD matrix
        LD_matrix = numpy.corrcoef(X,rowvar=0)

        ### simulate traits
        y = models.trait_simulation.simulate_traits(X,snp_ratios=r, beta_var=0.2)
        ### scale traits
        y = preprocessing.scale(y)

        t_statistics = models.trait_simulation.build_linear_models(X,y)

        beta = [x.slope for x in t_statistics]
        se_beta = [x.stderr for x in t_statistics]

        ###calcuate z

        z =  numpy.divide(beta, se_beta)

        simulated_effectsize_data = ([x*numpy.sqrt(n) for x in beta], LD_matrix, n)

        gene_set_BFs = models.bayes_factors.calc_variant_set_BFs(simulated_effectsize_data,k=5,v=0.01)

        gene_set_posteriors = models.bayes_factors.calc_posterior(gene_set_BFs)
        print r, gene_set_posteriors[0:5]

