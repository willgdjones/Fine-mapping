
# coding: utf-8

# # Colocalisation

# Is it possible to ascertain whether two traits are due to the same causal variant?
# 
# This is the aim of colocalisation. This is calculated by independently computing the gene-set probabilities, and then calculating the cartesian product with a multiplication operation to form a colocalisation matrix. Summing along the diagonal entries gives the total evidence for colocalisation.
# 
# Suppose the posterior probabilities for each trait $\mathbf{a}$ and $\mathbf{b}$ with respect to the same genotype are the vectors:
# 
# $$
# \mathbf{a}
# =
# \begin{pmatrix}
#  & a_{1}  & \\
#  & \vdots & \\
#  & a_{i}  & \\
#  & \vdots & \\
#  & a_{M}  &
# \end{pmatrix}
# ,
# \quad
# \mathbf{b}
# =
# \begin{pmatrix}
#  & b_{1}  & \\
#  & \vdots & \\
#  & b_{i}  & \\
#  & \vdots & \\
#  & b_{M}  &
# \end{pmatrix}
# $$
# 
# where $a_i$ and $b_i$ denotes the posterior probability that the $i$th candidate gene set is causal to trait $\mathbf{a}$ and $\mathbf{b}$ respectively
# 
# The colocalisation matrix is defined as the Cartesian (also known as dyadic) product of these two vectors $\mathbf{a}  \unicode{x25E6}  \mathbf{b}$. 
# 
# $$
# \mathbf{a} \unicode{x25E6} \mathbf{b}
# =
# \begin{pmatrix}
#  a_{1}b_{1} &        &   \dots    &        &  a_{1}b_{M}  \\
#             & \ddots &            &        &              \\
#  \vdots     &        & a_{i}b_{i} &        &  \vdots      \\
#             &        &            & \ddots &              \\
#  a_{M}b_{1} &        &   \dots    &        & a_{M}b_{M}  
# \end{pmatrix}
# $$
# 
# an $M \times M$ matrix.
# 
# The sum $\sum^{M}_{i=1}a_{i}b_{i}$ along the diagonal is the total evidence for colocalisation.
# 
# 
# 

# In[ ]:

import numpy
import bayes_factors
import trait_simulation
import itertools

def is_colocalised(X, LD_matrix ,trait1, trait2,db=0):
    """
    With respect to a shared genotype X, Determine whether trait1 and trait2 are colocalised
    given an LD matric LD_matrix
    I.e. whether there is evidence that they share a genetic basis.
    """

    ### Get number of samples
    n = X.shape[0]
    
    ### generate individual linear models
    models1 = trait_simulation.build_linear_models(X,trait1)
    models2 = trait_simulation.build_linear_models(X,trait2)

    ### pull out slope and standard error terms.
    beta1 = [x.slope for x in models1]
    se_beta1 = [x.stderr for x in models1]

    beta2 = [x.slope for x in models2]
    se_beta2 = [x.stderr for x in models2]

    ### calculate z scores
    simulated_effectsize_data1 = ([x*numpy.sqrt(n) for x in beta1], LD_matrix, n)
    simulated_effectsize_data2 = ([x*numpy.sqrt(n) for x in beta2], LD_matrix, n)

    ### generate the gene set Bayes Factors
    gene_set_BFs1 = bayes_factors.calc_variant_set_BFs(simulated_effectsize_data1,k=4,v=0.01)
    gene_set_BFs2 = bayes_factors.calc_variant_set_BFs(simulated_effectsize_data2,k=4,v=0.01)
    

    ### calculate the posteriors
    gene_set_posteriors1 = bayes_factors.calc_posterior(gene_set_BFs1)
    gene_set_posteriors2 = bayes_factors.calc_posterior(gene_set_BFs2)
    
    if db == 1: 
        
        print gene_set_BFs1[0:10]
        print gene_set_BFs2[0:10]
        
        print gene_set_posteriors1[0:10]
        print gene_set_posteriors2[0:10]


    ### sort by posterior size
    gene_set_posteriors1.sort(key=lambda x: x[0], reverse=False)
    gene_set_posteriors2.sort(key=lambda x: x[0], reverse=False)

    ### select just toe posteriors
    posteriors1 = [x[1] for x in gene_set_posteriors1]
    posteriors2 = [x[1] for x in gene_set_posteriors2]

    ### generate cartesian product from the posteriors
    cart_product = list(itertools.product(posteriors1,posteriors2))

    gene_set_len1 = len(gene_set_posteriors1)
    gene_set_len2 = len(gene_set_posteriors2)

    ### calculate colocalisation posteriors with a specificed scoring function.
    colocalisations = numpy.array(map(lambda x: min(x[0],x[1]), cart_product)).reshape(gene_set_len1,gene_set_len2)


    ### pull out sorted set list
    sorted_setlist1 = [x[0] for x in gene_set_posteriors1]
    sorted_setlist2 = [x[0] for x in gene_set_posteriors2]

    if db == 1:
        
        ###  create bidirectional map from gene_set to positon in colocalisation array
        setlist_1map = bidict([(sorted_setlist1[i],i) for i in range(len(sorted_setlist1))])
        setlist_2map = bidict([(sorted_setlist1[i],i) for i in range(len(sorted_setlist2))])

        bf_1map = dict(gene_set_BFs1)
        bf_2map = dict(gene_set_BFs2)


        posterior1_map = dict(gene_set_posteriors1)
        posterior2_map = dict(gene_set_posteriors2)
        pdb.set_trace()

    ### output total evidence for colocalisation
    return sum([colocalisations[i][i] for i in range(colocalisations.shape[0])])


# In[2]:

if __name__ == '__main__':
    get_ipython().magic(u'reset -f')
    import numpy
    from sklearn import preprocessing
    import sys
    sys.path.append('/Users/fonz/Projects/Notebooks/Fine-mapping/src')
    import models
    

    gene_ratio_sets = [({8:1},{8:1}),
                 ({8:1},{8:2}),
                 ({8:1},{10:1}),
                 ({8:1, 10:1},{8:1, 10:1}),
                 ({8:1, 10:1},{8:1, 10:2}),
                 ({8:1, 10:1},{8:1, 15:1}),
                 ({8:1, 10:1, 12:1},{8:1, 10:1, 12:1}),
                 ({8:1, 10:1, 12:1},{8:1, 10:1, 12:2}),
                 ({8:1, 10:1, 12:1},{8:1, 10:1, 15:1}),
                ]
    ### set sample size
    n = 10000

    ### simulate genotypes and scale columns
    X = preprocessing.scale(models.trait_simulation.simulate_genotype(n, 30, (0.85, 0.1, 0.05)))

    ### calculate LD matrix
    LD_matrix = numpy.corrcoef(X,rowvar=0)

    for gr in gene_ratio_sets:        

        ### simulate two traits and scale columns
        y1 = preprocessing.scale(models.trait_simulation.simulate_traits(X, snp_ratios=gr[0], beta_var=0.2))
        y2 = preprocessing.scale(models.trait_simulation.simulate_traits(X, snp_ratios=gr[0], beta_var=0.2))

        print gr, models.colocalisation.is_colocalised(X,LD_matrix,y1,y2) > 0.6

